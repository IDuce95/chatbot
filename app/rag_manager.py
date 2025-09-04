import os
from typing import Any, Dict, List

import toml
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .rag_metrics import RAGMetrics


class RAGManager:
    def __init__(self, config_path: str = "config.toml"):
        self.config = toml.load(config_path)

        self.docs_path = self.config["rag"]["docs_directory"]
        self.db_path = self.config["database"]["persist_directory"]
        self.faiss_path = self.config["database"]["faiss_index_path"]
        self.vectorstore_type = self.config["rag"]["vectorstore_type"]
        self.chunk_size = self.config["rag"]["chunk_size"]
        self.chunk_overlap = self.config["rag"]["chunk_overlap"]
        self.max_retrieved_docs = self.config["rag"]["max_retrieved_docs"]

        self._load_environment()
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        self.vectorstore = None
        self.metrics = RAGMetrics()
        self.setup_vectorstore()

    def setup_vectorstore(self):
        try:
            if self.vectorstore_type == "chroma":
                self.setup_chromadb()
            elif self.vectorstore_type == "faiss":
                self.setup_faiss()
            else:
                raise ValueError(f"Unsupported vectorstore type: {self.vectorstore_type}")
        except Exception as e:
            print(f"Error setting up vectorstore: {e}")
            raise

    def setup_chromadb(self):
        try:
            if os.path.exists(self.db_path):
                self.vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
                print(f"Loaded existing ChromaDB from {self.db_path}")
            else:
                print(f"Creating new ChromaDB at {self.db_path}")
                self.vectorstore = None
        except Exception as e:
            print(f"Error setting up ChromaDB: {e}")
            raise

    def setup_faiss(self):
        try:
            faiss_index_file = os.path.join(self.faiss_path, "index.faiss")
            if os.path.exists(faiss_index_file):
                self.vectorstore = FAISS.load_local(
                    self.faiss_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing FAISS index from {self.faiss_path}")
            else:
                print(f"Creating new FAISS index at {self.faiss_path}")
                self.vectorstore = None
        except Exception as e:
            print(f"Error setting up FAISS: {e}")
            raise

    def _load_environment(self):
        import os
        from pathlib import Path

        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

    def load_pdf_documents(self) -> List[Document]:
        documents = []
        pdf_files = [f for f in os.listdir(self.docs_path) if f.endswith('.pdf')]

        print(f"Loading {len(pdf_files)} PDF files...")

        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(self.docs_path, pdf_file)
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata['source_file'] = pdf_file
                    doc.metadata['file_type'] = 'pdf'

                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {pdf_file}")

            except Exception as e:
                print(f"  Error loading {pdf_file}: {e}")
                continue

        print(f"Total loaded: {len(documents)} document pages")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, documents: List[Document]):
        if not documents:
            print("No documents to process")
            return

        chunks = self.split_documents(documents)

        if self.vectorstore_type == "chroma":
            print("Creating embeddings and populating ChromaDB...")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.db_path
            )
            print(f"ChromaDB saved to {self.db_path}")

        elif self.vectorstore_type == "faiss":
            print("Creating embeddings and populating FAISS...")
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            self.vectorstore.save_local(self.faiss_path)
            print(f"FAISS index saved to {self.faiss_path}")

        else:
            raise ValueError(f"Unsupported vectorstore type: {self.vectorstore_type}")

    def search_documents(self, query: str, k: int = None) -> List[Document]:
        if k is None:
            k = self.max_retrieved_docs

        if not self.vectorstore:
            print("Vectorstore not initialized")
            return []

        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def search_documents_with_scores(self, query: str, k: int = None):
        if k is None:
            k = self.max_retrieved_docs

        if not self.vectorstore:
            print("Vectorstore not initialized")
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            retrieved_docs = []
            for i, (doc, distance) in enumerate(results):
                doc_info = {
                    'id': f"{doc.metadata.get('source_file', 'unknown')}_{i}",
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': 1.0 - distance,
                    'distance': distance
                }
                retrieved_docs.append(doc_info)

            self.metrics.evaluate_retrieval(query, retrieved_docs)

            return results
        except Exception as e:
            print(f"Error searching documents with scores: {e}")
            return []

    def get_context_with_relevance(self, query: str, k: int = None, max_distance: float = 0.4):
        if k is None:
            k = self.max_retrieved_docs

        results = self.search_documents_with_scores(query, k)
        if not results:
            return "", False, 0, []

        relevant_docs = []
        retrieved_docs_info = []

        for i, (doc, distance) in enumerate(results):
            doc_info = {
                'id': f"{doc.metadata.get('source_file', 'unknown')}_{i}",
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': 1.0 - distance,
                'distance': distance
            }
            retrieved_docs_info.append(doc_info)

            if distance < max_distance:
                relevant_docs.append(doc)

        if not relevant_docs:
            return "", False, 0, []

        context_parts = []
        source_files = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('source_file', 'unknown')
            source_files.append(source)
            content = doc.page_content.strip()
            context_parts.append(f"[Doc {i} - {source}]\n{content}")

        context = "\n\n".join(context_parts)

        return context, True, len(relevant_docs), source_files

    def initialize_database(self):
        print("Initializing RAG database...")
        documents = self.load_pdf_documents()
        if documents:
            self.create_vectorstore(documents)
            print("RAG database initialized successfully!")
        else:
            print("No documents found to initialize database")

    def get_stats(self) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"status": "not_initialized", "documents": 0}

        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "status": "ready",
                "documents": count,
                "db_path": self.db_path
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_metrics_summary(self) -> Dict[str, Any]:
        return self.metrics.get_session_summary()

    def export_metrics(self, filepath: str = "rag_metrics.json"):
        self.metrics.export_metrics(filepath)
        print(f"Metrics exported to {filepath}")

    def clear_metrics(self):
        self.metrics.clear_session()
        print("Metrics session cleared")


if __name__ == "__main__":
    rag = RAGManager()
    rag.initialize_database()
