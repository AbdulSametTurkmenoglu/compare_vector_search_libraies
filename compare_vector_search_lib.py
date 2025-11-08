import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Annoy
from langchain_core.documents import Document


class Config:
    """Centralized configuration management"""
    SCRIPT_DIR = Path(__file__).parent.absolute()
    DOC_DIRECTORY = SCRIPT_DIR / "data"
    FAISS_INDEX_PATH = SCRIPT_DIR / "faiss_storage_langchain"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 3
    EMBEDDING_MODEL = "text-embedding-3-small"


class DocumentProcessor:
    """Handles document loading and text splitting"""

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE,
                 chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_documents(self, directory_path: Path) -> List[Document]:
        """
        Load all .txt files from directory with UTF-8 encoding.
        Handles Turkish characters properly.
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        print(f" Loading documents from: {directory_path}")

        try:
            loader = DirectoryLoader(
                str(directory_path),
                glob="**/*.txt",
                show_progress=True,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
            )
            documents = loader.load()

            if not documents:
                print(f"  No .txt files found in {directory_path}")
                return []

            print(f" Loaded {len(documents)} document(s)")
            return documents

        except Exception as e:
            print(f" Error loading documents: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        if not documents:
            return []

        print(f"️  Splitting documents (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        split_docs = self.text_splitter.split_documents(documents)
        print(f" Created {len(split_docs)} chunks from {len(documents)} document(s)")
        return split_docs

    def load_and_split(self, directory_path: Path) -> List[Document]:
        """Complete pipeline: load and split documents"""
        documents = self.load_documents(directory_path)
        return self.split_documents(documents)


class VectorStoreManager:
    """Manages FAISS and Annoy vector stores"""

    def __init__(self, embedding_model: OpenAIEmbeddings):
        self.embeddings = embedding_model

    def create_faiss_store(self, docs: List[Document],
                           save_path: Optional[Path] = None) -> FAISS:
        if not docs:
            raise ValueError("No documents provided for FAISS store creation")

        print(f" Creating FAISS vector store...")
        db = FAISS.from_documents(docs, self.embeddings)

        if save_path:
            self.save_faiss_store(db, save_path)

        return db

    def save_faiss_store(self, db: FAISS, save_path: Path) -> None:
        """Save FAISS index to disk"""
        print(f" Saving FAISS index to: {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        db.save_local(str(save_path))
        print(" FAISS index saved successfully")

    def load_faiss_store(self, load_path: Path) -> FAISS:
        """Load pre-built FAISS vector store"""
        if not load_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {load_path}")

        print(f" Loading FAISS index from: {load_path}")
        db = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(" FAISS index loaded successfully")
        return db

    def create_annoy_store(self, docs: List[Document]) -> Annoy:
        """Create in-memory Annoy vector store"""
        if not docs:
            raise ValueError("No documents provided for Annoy store creation")

        print(" Creating Annoy vector store...")
        db = Annoy.from_documents(docs, self.embeddings)
        print(" Annoy store created successfully")
        return db


class QueryEngine:
    """Handles querying vector stores"""

    @staticmethod
    def query(store, query: str, k: int = Config.TOP_K_RESULTS) -> List[Document]:
        """Perform similarity search"""
        print(f"\n Query: '{query}'")
        print(f"   Retrieving top {k} result(s)...")

        try:
            results = store.similarity_search(query, k=k)

            if results:
                print(f" Found {len(results)} result(s)\n")
                for i, doc in enumerate(results, 1):
                    print(f"{'─' * 60}")
                    print(f"Result #{i}")
                    print(f"{'─' * 60}")
                    print(doc.page_content)
                    if hasattr(doc, 'metadata') and doc.metadata:
                        print(f"\n Source: {doc.metadata.get('source', 'Unknown')}")
                    print()
            else:
                print(" No results found")

            return results

        except Exception as e:
            print(f" Query error: {e}")
            return []


def main():
    """Main application workflow"""

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print(" Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    print("=" * 70)
    print("🇹🇷 Turkish News Vector Store System")
    print("=" * 70)

    try:
        print("\n Initializing components...")
        embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        processor = DocumentProcessor()
        vector_manager = VectorStoreManager(embeddings)
        query_engine = QueryEngine()

        documents = processor.load_and_split(Config.DOC_DIRECTORY)

        if not documents:
            print("\n  No documents loaded. Please check your data directory.")
            print(f"   Expected path: {Config.DOC_DIRECTORY}")
            sys.exit(1)

        print("\n" + "=" * 70)
        print(" FAISS WORKFLOW")
        print("=" * 70)

        faiss_db = vector_manager.create_faiss_store(
            documents,
            save_path=Config.FAISS_INDEX_PATH
        )

        query_engine.query(faiss_db, "Meta ne geliştirdi?")  # Turkish: What did Meta develop?
        query_engine.query(faiss_db, "Dünya haberlerinde neler oluyor?")  # What's happening in world news?

        print("\n" + "=" * 70)
        print(" ANNOY WORKFLOW")
        print("=" * 70)

        annoy_db = vector_manager.create_annoy_store(documents)
        query_engine.query(annoy_db, "Teknoloji haberleri neler?")  # What are the tech news?

        print("\n" + "=" * 70)
        print(" All workflows completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()