"""
RAG-based retrieval system for SAT knowledge matching.
Uses sentence transformers for embeddings and FAISS for vector search.
"""

import os
# Fix for OpenMP threading issues on macOS (must be before other imports)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'

import json
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Optional


class SATKnowledgeRetriever:
    """Retrieval system for matching student questions to SAT knowledge points."""
    
    def __init__(self, knowledge_base_path: str = "sat_knowledge_base.json", 
                 model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "faiss_index.faiss",
                 index_meta_path: str = "faiss_index_meta.json"):
        """
        Initialize the RAG system.
        
        Args:
            knowledge_base_path: Path to JSON file containing SAT knowledge base
            model_name: Name of sentence transformer model for embeddings
        """
        self.knowledge_base_path = knowledge_base_path
        self.model_name = model_name
        self.index_path = index_path
        self.index_meta_path = index_meta_path
        self.model = None
        self.knowledge_base = []
        self.embeddings = None
        self.index = None
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.metric = "ip"  # inner product over normalized vectors ~= cosine similarity
        self.hnsw_m = 32
        self.hnsw_ef_construction = 200
        self.hnsw_ef_search = 128
        
    def _kb_fingerprint(self) -> str:
        """Hash the knowledge base file so we can detect if the on-disk index is stale."""
        h = hashlib.sha1()
        with open(self.knowledge_base_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def load_knowledge_base(self):
        """Load SAT knowledge base from JSON file."""
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        print(f"Loaded {len(self.knowledge_base)} knowledge points.")
        
    def initialize_model(self):
        """Initialize the sentence transformer model."""
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully.")
        
    @staticmethod
    def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, eps)

    def _build_hnsw_index(self, vectors: np.ndarray) -> faiss.Index:
        """
        Build a FAISS HNSW index.

        We use inner-product similarity on L2-normalized vectors to approximate cosine similarity.
        """
        d = vectors.shape[1]
        index = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = self.hnsw_ef_search
        index.add(vectors)
        return index

    def create_embeddings(self):
        """Create embeddings for all knowledge points and build the FAISS index."""
        if not self.model:
            self.initialize_model()
            
        if not self.knowledge_base:
            self.load_knowledge_base()
            
        print("Creating embeddings for knowledge base...")
        
        # Combine topic, concept, and content for better retrieval
        texts = []
        for kb_item in self.knowledge_base:
            text = f"{kb_item['topic']}. {kb_item['concept']}. {kb_item['content']}"
            texts.append(text)
            
        # Generate embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = np.array(self.embeddings).astype('float32')
        self.embeddings = self._l2_normalize(self.embeddings).astype("float32")
        
        # Create FAISS index (HNSW for speed/quality tradeoff)
        self.dimension = self.embeddings.shape[1]
        self.index = self._build_hnsw_index(self.embeddings)
        
        print(f"Created FAISS index with {self.index.ntotal} vectors.")
        
    def save_index(self):
        """Save the FAISS index and metadata for faster loading."""
        if self.index is None:
            raise ValueError("Index is not built; call create_embeddings() first.")

        faiss.write_index(self.index, self.index_path)
        meta = {
            "knowledge_base_path": self.knowledge_base_path,
            "knowledge_base_fingerprint": self._kb_fingerprint(),
            "knowledge_base_size": len(self.knowledge_base),
            "dimension": self.dimension,
            "index_type": "hnsw",
            "metric": self.metric,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "model_name": self.model_name
        }
        with open(self.index_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Index saved to {self.index_path} (+ meta: {self.index_meta_path})")
        
    def load_index(self) -> bool:
        """Load pre-computed FAISS index if present and not stale."""
        if not (os.path.exists(self.index_path) and os.path.exists(self.index_meta_path)):
            return False

        try:
            with open(self.index_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected_fp = meta.get("knowledge_base_fingerprint")
            current_fp = self._kb_fingerprint()
            if expected_fp != current_fp:
                print("Saved index is stale (knowledge base changed). Rebuilding index...")
                return False

            print(f"Loading pre-computed index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            self.dimension = int(meta.get("dimension", self.dimension))

            # Load KB so that index ids map to items
            self.load_knowledge_base()

            # Ensure efSearch matches config
            if hasattr(self.index, "hnsw"):
                self.index.hnsw.efSearch = self.hnsw_ef_search

            print(f"Loaded index with {self.index.ntotal} vectors.")
            return True
        except Exception as e:
            print(f"Failed to load index; rebuilding. Reason: {e}")
            return False
        
    def retrieve(self, query: str, top_k: int = 5, context: Optional[str] = None) -> List[Dict]:
        """
        Retrieve most relevant knowledge points for a query.
        
        Args:
            query: Student question or dialogue
            top_k: Number of results to return
            context: Optional conversation context (prior turns) to improve retrieval
            
        Returns:
            List of dictionaries containing knowledge points with relevance scores
        """
        if not self.model:
            self.initialize_model()
            
        if self.index is None:
            if not self.load_index():
                self.create_embeddings()
                self.save_index()
                
        # Encode query (optionally with prior conversation context)
        if context and context.strip():
            full_query = f"Conversation context:\n{context.strip()}\n\nCurrent question:\n{query.strip()}"
        else:
            full_query = query

        query_embedding = self.model.encode([full_query])
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = self._l2_normalize(query_embedding).astype("float32")
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.knowledge_base):
                continue
            kb_item = self.knowledge_base[idx].copy()
            # With IP over normalized vectors, score is cosine similarity in [-1, 1]
            kb_item['relevance_score'] = float(score)
            kb_item['rank'] = i + 1
            results.append(kb_item)
            
        return results
        
    def initialize(self, force_rebuild: bool = False):
        """
        Initialize the complete system.
        
        Args:
            force_rebuild: If True, rebuild index even if saved version exists
        """
        if not force_rebuild and self.load_index():
            self.initialize_model()  # Still need model for new queries
            return
            
        self.load_knowledge_base()
        self.initialize_model()
        self.create_embeddings()
        self.save_index()


def main():
    """Demo the retrieval system."""
    retriever = SATKnowledgeRetriever()
    retriever.initialize()
    
    # Example queries
    test_queries = [
        "How do I solve quadratic equations?",
        "What's the difference between mean and median?",
        "How do I identify the main idea in a passage?",
        "When should I use a comma?",
        "How do I solve systems of equations?"
    ]
    
    print("\n" + "="*60)
    print("SAT Knowledge Matching System - Demo")
    print("="*60 + "\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)
        results = retriever.retrieve(query, top_k=3)
        for result in results:
            print(f"\n[{result['section']}] {result['topic']} - {result['concept']}")
            print(f"Relevance: {result['relevance_score']:.3f}")
            print(f"Content: {result['content'][:150]}...")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
