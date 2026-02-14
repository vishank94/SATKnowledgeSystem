"""
Streamlit interface for SAT Knowledge Matching System.
"""

import os
# Fix for OpenMP threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'

import streamlit as st
import sys
import re
from rag_system import SATKnowledgeRetriever
# from answer_generator import AnswerGenerator  # Commented out - answer generation disabled

# Page configuration
st.set_page_config(
    page_title="SAT Knowledge Matching System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'reset_session' not in st.session_state:
    st.session_state.reset_session = False

# Title and description
st.title("📚 SAT Knowledge Matching System")
st.markdown("""
This system uses RAG (Retrieval-Augmented Generation) to match student questions 
with relevant SAT curriculum knowledge points.
""")

# Helper function to highlight keywords
def highlight_keywords(text, query, keywords=None):
    """Highlight relevant keywords in text."""
    if not text:
        return text
    
    # Extract keywords from query if not provided
    if keywords is None:
        # Simple keyword extraction: remove stop words and get meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those'}
        query_words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in query_words if w not in stop_words and len(w) > 2]
    
    # Also include topic/concept keywords from the result itself
    highlighted_text = text
    all_keywords = set(keywords)
    
    # Add common SAT terms that might be relevant
    sat_keywords = ['equation', 'formula', 'solve', 'factor', 'graph', 'function', 'passage', 'main idea', 'tone', 'grammar', 'punctuation', 'comma', 'semicolon', 'verb', 'subject', 'pronoun']
    for kw in sat_keywords:
        if kw.lower() in query.lower():
            all_keywords.add(kw.lower())
    
    # Highlight each keyword
    for keyword in all_keywords:
        if len(keyword) > 2:  # Only highlight meaningful words
            # Case-insensitive highlighting
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted_text = pattern.sub(
                lambda m: f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">{m.group()}</mark>',
                highlighted_text
            )
    
    return highlighted_text

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Number of results to return", min_value=1, max_value=10, value=5)
    force_rebuild = st.checkbox("Force rebuild index", value=False)
        
    # if st.button("🔄 Reset Session", type="secondary", key="reset_btn"):
    #     st.session_state.conversation_history = []
    #     st.session_state.last_results = []
    #     st.session_state.reset_session = True
    #     st.success("Session reset! Conversation history cleared.")
    #     st.rerun()
    
    # Handle reset flag
    # if st.session_state.reset_session:
    #     st.session_state.reset_session = False
        
    # LLM Configuration - COMMENTED OUT
    # st.header("🤖 Answer Generation")
    # use_llm = st.checkbox("Enable LLM answer generation", value=True, 
    #                       help="Uses open-source FLAN-T5 model (no API key required)")
    # if use_llm and 'answer_generator' not in st.session_state:
    #     with st.spinner("Loading open-source LLM model (first time may take a minute)..."):
    #         try:
    #             st.session_state.answer_generator = AnswerGenerator()
    #             if st.session_state.answer_generator.is_available():
    #                 st.success("LLM ready! Using FLAN-T5 model.")
    #             else:
    #                 st.warning("LLM model could not be loaded. Using fallback answer generation.")
    #                 st.session_state.answer_generator = None
    #         except Exception as e:
    #             st.error(f"Could not initialize LLM: {e}")
    #             st.session_state.answer_generator = None
    # elif not use_llm:
    #     st.session_state.answer_generator = None
    
    # st.markdown("---")
    
    if st.button("Initialize System"):
        with st.spinner("Initializing RAG system..."):
            try:
                retriever = SATKnowledgeRetriever()
                retriever.initialize(force_rebuild=force_rebuild)
                st.session_state.retriever = retriever
                st.session_state.initialized = True
                st.success("System initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing system: {str(e)}")
                st.session_state.initialized = False

    st.markdown("---")

    st.header("💬 Conversation History")
    
    if st.session_state.conversation_history:
        st.markdown(f"**{len(st.session_state.conversation_history)} previous queries**")
        with st.expander("View History"):
            for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
                q = entry[0] if isinstance(entry, (list, tuple)) else entry
                st.markdown(f"{i}. {q[:60]}...")
    else:
        st.info("No conversation history yet.")

# Main content area
if not st.session_state.initialized:
    st.info("👈 Please initialize the system using the sidebar.")
    st.markdown("""
    ### How to use:
    1. Click "Initialize System" in the sidebar
    2. Wait for the system to load (first time may take a minute)
    3. Enter your question in the text area below
    4. View the matched knowledge points
    """)
else:
    # Query input
    st.header("💬 Enter Your Question")
    
    query = st.text_area(
        "Enter your question",
        height=100,
        placeholder="Enter your SAT-related question here...",
        label_visibility="collapsed"
    )
    
    if st.button("🔍 Search Knowledge Base", type="primary"):
        if query.strip():
            with st.spinner("Searching knowledge base..."):
                try:
                    # Build context from conversation history - COMMENTED OUT
                    # context = None
                    # if st.session_state.conversation_history:
                    #     # Use last 3 queries as context
                    #     recent_queries = [q for q, _ in st.session_state.conversation_history[-3:]]
                    #     context = "\n".join(recent_queries)
                    
                    # Retrieve without context (conversation history not used for retrieval)
                    results = st.session_state.retriever.retrieve(query, top_k=top_k, context=None)
                    
                    # Generate answer using LLM if available - COMMENTED OUT
                    # generated_answer = None
                    # if st.session_state.get('answer_generator') and st.session_state.answer_generator:
                    #     with st.spinner("Generating answer..."):
                    #         try:
                    #             generated_answer = st.session_state.answer_generator.generate_answer(
                    #                 query, results, context
                    #             )
                    #         except Exception as e:
                    #             st.warning(f"Could not generate answer: {e}")
                    
                    # Store in conversation history (query, results)
                    st.session_state.conversation_history.append((query, results))
                    st.session_state.last_results = results
                    
                    # Display generated answer if available - COMMENTED OUT
                    # if generated_answer:
                    #     st.header("💡 Generated Answer")
                    #     st.info(generated_answer)
                    #     st.markdown("---")
                    
                    st.header("📖 Matching Knowledge Points")
                    st.markdown(f"Found {len(results)} relevant knowledge point(s):\n")
                    
                    # Extract keywords from query for highlighting
                    query_keywords = re.findall(r'\b\w+\b', query.lower())
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those'}
                    keywords = [w for w in query_keywords if w not in stop_words and len(w) > 2]
                    
                    for i, result in enumerate(results, 1):
                        # Highlight keywords in content
                        highlighted_content = highlight_keywords(result['content'], query, keywords)
                        highlighted_topic = highlight_keywords(result['topic'], query, keywords)
                        highlighted_concept = highlight_keywords(result['concept'], query, keywords)
                        
                        with st.expander(
                            f"**{i}. [{result['section']}] {result['topic']} - {result['concept']}** "
                            f"(Relevance: {result['relevance_score']:.3f})",
                            expanded=(i == 1)
                        ):
                            st.markdown(f"**Topic:** {highlighted_topic}", unsafe_allow_html=True)
                            st.markdown(f"**Concept:** {highlighted_concept}", unsafe_allow_html=True)
                            st.markdown(f"**Difficulty:** {result['difficulty']}")
                            st.markdown(f"**Content:**")
                            st.markdown(highlighted_content, unsafe_allow_html=True)
                            st.markdown(f"*Knowledge Point ID: {result['id']}*")
                except Exception as e:
                    st.error(f"Error retrieving results: {str(e)}")
        else:
            st.warning("Please enter a question.")
    
    # Show conversation history if available
    if st.session_state.conversation_history:
        st.markdown("---")
        st.header("📜 Conversation History")
        with st.expander("View Full Conversation", expanded=False):
            for idx, entry in enumerate(st.session_state.conversation_history, 1):
                # Handle both old format (with answer) and new format (without answer)
                if len(entry) == 3:
                    q, res, _ = entry  # Ignore answer if present
                elif len(entry) == 2:
                    q, res = entry
                else:
                    q, res = entry[0], entry[1] if len(entry) > 1 else []
                
                st.markdown(f"**Query {idx}:** {q}")
                
                # Answer display - COMMENTED OUT
                # if answer:
                #     st.markdown("**Answer:**")
                #     st.info(answer)
                
                if res:
                    st.markdown(f"*Found {len(res)} knowledge point(s)*")
                    # Show top 3 knowledge points (can't nest expanders, so show directly)
                    for i, result in enumerate(res[:3], 1):  # Show top 3
                        st.markdown(f"**{i}. [{result['section']}] {result['topic']} - {result['concept']}**")
                        st.markdown(f"*{result['content'][:100]}...*")
                    if len(res) > 3:
                        st.markdown(f"*... and {len(res) - 3} more knowledge point(s)*")
                
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>SAT Knowledge Matching System | Built with RAG (Retrieval-Augmented Generation)</p>
</div>
""", unsafe_allow_html=True)
