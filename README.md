# 🇹🇷 Turkish News Vector Store System

A Python-based document retrieval system that uses vector embeddings to search through Turkish news articles. The system supports both FAISS and Annoy vector stores for efficient similarity search.
## Features

This API assigns an LLM as a "Tech Assistant" and enforces the following checks:

1. **Multi-format Support:** Loads and processes .txt files with UTF-8 encoding
2. **Turkish Character Support:** Properly handles Turkish characters (ı, ğ, ü, ş, ö, ç)
3. **Dual Vector Store:**  Implements both FAISS and Annoy indexing
4. **Smart Text Splitting:** Chunks documents intelligently with configurable overlap
5. **Semantic Search:** Uses OpenAI embeddings for similarity-based retrieval
6. **Persistent Storage:** Save and load FAISS indices for faster subsequent runs

## Installation and Running

### 1. Clone the Repository
```bash
git clone https://github.com/AbdulSametTurkmenoglu/compare_vector_search_libraies.git
cd compare_vector_search_libraies
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Copy the `.env.example` file and create a new file named `.env`:
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Now open the `.env` file and enter your own `OPENAI_API_KEY`.

#### Configuration
```python
class Config:
    CHUNK_SIZE = 500           # Size of text chunks
    CHUNK_OVERLAP = 50         # Overlap between chunks
    TOP_K_RESULTS = 3          # Number of results to retrieve
    EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
```


#### Usage
```bash
python main.py
```
