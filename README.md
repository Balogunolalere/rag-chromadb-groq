# RAG with ChromaDB and Groq

This project implements a Retrieval-Augmented Generation (RAG) system using ChromaDB for vector storage and Groq for language model inference. It allows users to upload PDF or TXT documents, create a searchable knowledge base, and query documents using natural language.

## Features

- Document upload and processing (PDF and TXT)
- Text chunking with sliding window or semantic methods
- Vector storage using ChromaDB
- Language model inference using Groq
- Streamlit-based web interface
- Query history tracking

## Prerequisites

- Python 3.8+
- Groq API key
- ChromaDB
- Streamlit
- PyPDF2
- spaCy (with en_core_web_lg model)
- langdetect

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Balogunolalere/rag-chromadb-groq.git
   cd rag-chromadb-groq
   ```

2. Create virtual environment and install dependencies:
   ```
   virtualenv env && source env/bin/activate && pip install -r requirements.txt
   ```

3. Download the spaCy model:
   ```
   python -m spacy download en_core_web_lg
   ```

4. Set up environment variables:
   Copy `.env.example` to `.env` and fill in your Groq API key.

## Usage

Run the Streamlit app:

```
streamlit run main.py
```

Navigate to the provided URL in your web browser to use the application.

![example](https://github.com/Balogunolalere/rag-chromadb-groq/blob/main/Screenshot%20from%202024-09-16%2023-23-48.png?raw=true)

![example](https://github.com/Balogunolalere/rag-chromadb-groq/blob/main/Screenshot%20from%202024-09-16%2023-23-55.png?raw=true)