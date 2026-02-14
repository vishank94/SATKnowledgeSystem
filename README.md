# SAT Knowledge Matching System

A RAG-based system that matches student questions with relevant SAT curriculum knowledge points.

## Demo:
[![Watch the video](https://github.com/vishank94/SATKnowledgeSystem/blob/main/SATKnowledgeSystemThumbnail.png)](https://drive.google.com/file/d/1hEpPvmcOPjM0NV3I0cuu-uiClpBP2YT8/view?usp=sharing)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser. First-time setup will:
- Download the embedding model (~80MB)
- Build the search index (~30-60 seconds)

### 3. Use the System

1. Click "Initialize System" in the sidebar
2. Enter your SAT-related question
3. Click "Search Knowledge Base"
4. View matching knowledge points with highlighted keywords

## Features

- 🎯 Semantic search over 100+ SAT knowledge points
- 🔍 Fast HNSW-based retrieval
- 🎨 Keyword highlighting in results
- 💬 Conversation history tracking

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Troubleshooting

**macOS users**: If you encounter segmentation faults, use:
```bash
./run.sh
```

**First run is slow**: Normal - the system downloads models and builds the index. Subsequent runs are much faster.

## Project Structure

```
SATKnowledge/
├── app.py                      # Streamlit web interface
├── rag_system.py              # Core RAG retrieval system
├── sat_knowledge_base.json    # SAT knowledge base (100+ points)
├── requirements.txt           # Python dependencies
├── run.sh                     # Helper script for macOS
└── README.md                  # This file
```

## Knowledge Base

The system includes 100+ knowledge points covering:
- **Math**: Algebra, Geometry, Functions, Statistics
- **Reading**: Comprehension, Main Idea, Inference
- **Writing**: Grammar, Punctuation, Sentence Structure
