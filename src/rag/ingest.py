"""
src/rag/ingest.py
=================
DAY 11 — Document Ingestion & FAISS Vector Store
-------------------------------------------------
Pipeline:
  PDF files in data/rag_docs/
    → pypdf text extraction (page-by-page)
    → RecursiveCharacterTextSplitter (chunk_size=512, overlap=64)
    → sentence-transformers/all-MiniLM-L6-v2 embeddings  (local, free)
    → FAISS index with rich metadata
    → persisted to vector_store/faiss_index/

WHY each choice
───────────────
all-MiniLM-L6-v2 : 22 MB, runs on CPU in <100 ms per batch. Sufficient
  for domain-specific retrieval on English environmental documents.
  Outperforms TF-IDF on semantic queries ("health effects of PM2.5"
  matches "particulate matter respiratory impact").

FAISS IndexFlatL2 : exact nearest-neighbour, no approximation error.
  Fast enough for corpora <100k chunks on a single machine. Switch to
  IndexIVFFlat only when the corpus exceeds ~500k documents.

Metadata stored per chunk: source filename, page number, document type,
  ingestion date. This lets the LLM agent cite exactly which page and
  document a retrieved passage came from — critical for health advisories
  that must reference WHO/EPA standards.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


# ── lazy imports so the module loads even if optional deps are missing ────────
def _require(pkg: str, install: str) -> Any:
    try:
        import importlib

        return importlib.import_module(pkg)
    except ImportError:
        raise ImportError(
            f"Required package missing. Install with: pip install {install}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_PATH = Path("vector_store/faiss_index")

# Air quality domain test queries (Day 11 validation suite)
TEST_QUERIES = [
    "What are the WHO guidelines for PM2.5 annual exposure?",
    "What AQI level is considered unhealthy for sensitive groups?",
    "How does NO2 affect respiratory health in urban areas?",
    "What are the health effects of long-term ozone exposure?",
    "How is the EPA AQI calculated for PM10?",
    "What emission sources contribute most to urban PM2.5?",
    "What is the safe daily exposure limit for carbon monoxide?",
    "How do temperature inversions affect air quality in cities?",
    "What are the recommended actions during a Very Unhealthy AQI event?",
    "What populations are most vulnerable to air pollution health effects?",
]


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION CLASS
# ─────────────────────────────────────────────────────────────────────────────


class AirSentinelIngestor:
    """
    Reads PDF documents, chunks them, embeds with a local model,
    and stores results in a persisted FAISS index.

    Parameters
    ----------
    docs_dir      : Path  Folder containing PDF files to ingest.
    faiss_path    : Path  Directory where the FAISS index will be saved.
    embed_model   : str   HuggingFace model name for sentence embeddings.
    chunk_size    : int   Max characters per chunk.
    chunk_overlap : int   Overlap between consecutive chunks (context continuity).
    """

    def __init__(
        self,
        docs_dir: Path = Path("data/rag_docs"),
        faiss_path: Path = FAISS_PATH,
        embed_model: str = EMBED_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.faiss_path = Path(faiss_path)
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embeddings = None  # lazy-loaded
        self._vector_store = None  # lazy-loaded after ingest or load

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, force_reingest: bool = False) -> None:
        """
        Full ingestion pipeline.
        If the FAISS index already exists and force_reingest=False, loads it.
        """
        if self.faiss_path.exists() and not force_reingest:
            logger.info(
                f"FAISS index found at {self.faiss_path} — loading existing index."
            )
            self._load_index()
            return

        logger.info(f"Starting document ingestion from {self.docs_dir}")
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(
                f"No PDF files found in {self.docs_dir}. "
                "Place WHO/EPA guideline PDFs there and re-run."
            )
            # Create a placeholder document so the pipeline doesn't crash
            self._create_placeholder_index()
            return

        docs = self._load_pdfs(pdf_files)
        chunks = self._split_documents(docs)
        self._build_faiss_index(chunks)
        logger.success(f"Ingestion complete. {len(chunks)} chunks indexed.")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        use_mmr: bool = False,
        fetch_k: int = 20,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant document chunks for a query.

        Parameters
        ----------
        query   : str   Natural-language query.
        k       : int   Number of chunks to return.
        use_mmr : bool  If True, use Maximal Marginal Relevance to reduce
                        redundancy in results. Useful when multiple chunks
                        from the same document dominate the top-k.
        fetch_k : int   Candidate pool size for MMR (must be > k).

        Returns
        -------
        List of dicts with keys: content, metadata, score
        """
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialised. Call run() first.")

        if use_mmr:
            results = self._vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k
            )
        else:
            results = self._vector_store.similarity_search_with_score(query, k=k)
            results = [(doc, score) for doc, score in results]

        output = []
        for item in results:
            if isinstance(item, tuple):
                doc, score = item
                output.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                    }
                )
            else:
                output.append(
                    {
                        "content": item.page_content,
                        "metadata": item.metadata,
                        "score": None,
                    }
                )
        return output

    def run_test_queries(self) -> None:
        """Run the 10-query validation suite and log results."""
        logger.info("Running Day 11 test query suite …")
        for i, query in enumerate(TEST_QUERIES, 1):
            results = self.similarity_search(query, k=2)
            top_source = (
                results[0]["metadata"].get("source", "N/A") if results else "none"
            )
            logger.info(f"  Q{i:02d}: '{query[:55]}…' → top doc: {top_source}")
        logger.success("Test suite complete.")

    def get_vector_store(self):
        """Return the raw LangChain FAISS vector store (for agent tool use)."""
        return self._vector_store

    # ── Private methods ───────────────────────────────────────────────────────

    def _get_embeddings(self):
        """Lazy-load embeddings model (downloads ~22 MB on first run)."""
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info(f"Loading embedding model: {self.embed_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embed_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _load_pdfs(self, pdf_files: list[Path]) -> list[dict]:
        """
        Extract text from PDFs page-by-page using pypdf.
        Returns a list of dicts: {content, metadata}

        WHY page-by-page?
        Each page is a natural semantic unit in regulatory documents.
        Chunking across page boundaries sometimes splits tables or
        numbered guidelines — page-level extraction prevents this.
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pip install pypdf")

        documents = []
        for pdf_path in pdf_files:
            logger.info(f"  Extracting: {pdf_path.name}")
            try:
                reader = PdfReader(str(pdf_path))
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    text = text.strip()
                    if len(text) < 50:  # skip nearly-blank pages (headers/footers)
                        continue
                    documents.append(
                        {
                            "content": text,
                            "metadata": {
                                "source": pdf_path.name,
                                "page": page_num + 1,
                                "doc_type": self._classify_doc_type(pdf_path.name),
                                "date": datetime.now(timezone.utc).isoformat(),
                            },
                        }
                    )
            except Exception as exc:
                logger.warning(f"  Failed to read {pdf_path.name}: {exc}")

        logger.info(f"Extracted {len(documents)} pages from {len(pdf_files)} PDFs.")
        return documents

    def _classify_doc_type(self, filename: str) -> str:
        """Infer document type from filename for metadata tagging."""
        fname = filename.lower()
        if "who" in fname:
            return "WHO_Guideline"
        if "epa" in fname:
            return "EPA_Standard"
        if "news" in fname:
            return "News_Article"
        if "city" in fname:
            return "City_Report"
        return "Environmental_Document"

    def _split_documents(self, documents: list[dict]) -> list:
        """
        Chunk documents using RecursiveCharacterTextSplitter.

        RecursiveCharacterTextSplitter tries to split on paragraph breaks
        ('\n\n') first, then sentences ('\n'), then words (' '), then
        characters — preserving semantic coherence as much as possible.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        for doc in documents:
            raw_chunks = splitter.split_text(doc["content"])
            for i, chunk_text in enumerate(raw_chunks):
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={**doc["metadata"], "chunk_index": i},
                    )
                )

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages.")
        return chunks

    def _build_faiss_index(self, chunks: list) -> None:
        """Build and persist the FAISS vector index."""
        from langchain_community.vectorstores import FAISS

        embeddings = self._get_embeddings()
        logger.info("Building FAISS index (this may take a minute on CPU) …")

        self._vector_store = FAISS.from_documents(chunks, embeddings)

        self.faiss_path.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(self.faiss_path))
        logger.success(f"FAISS index saved to {self.faiss_path}")

        # Also save metadata manifest for auditing
        manifest = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "num_chunks": len(chunks),
            "embed_model": self.embed_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        with open(self.faiss_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_index(self) -> None:
        """Load an existing FAISS index from disk."""
        from langchain_community.vectorstores import FAISS

        embeddings = self._get_embeddings()
        self._vector_store = FAISS.load_local(
            str(self.faiss_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS index loaded from {self.faiss_path}")

    def _create_placeholder_index(self) -> None:
        """
        Create a minimal index from synthetic text so downstream
        tools don't crash when no PDFs are present yet.
        """
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        placeholder_docs = [
            Document(
                page_content=(
                    "The WHO Air Quality Guidelines state that the annual mean "
                    "PM2.5 concentration should not exceed 5 µg/m³. "
                    "The 24-hour mean should not exceed 15 µg/m³."
                ),
                metadata={
                    "source": "placeholder",
                    "page": 1,
                    "doc_type": "WHO_Guideline",
                    "date": datetime.now(timezone.utc).isoformat(),
                },
            ),
            Document(
                page_content=(
                    "The EPA AQI for PM2.5 ranges from 0 (Good) to 500 (Hazardous). "
                    "An AQI above 150 is considered Unhealthy and sensitive groups "
                    "should limit prolonged outdoor exertion."
                ),
                metadata={
                    "source": "placeholder",
                    "page": 1,
                    "doc_type": "EPA_Standard",
                    "date": datetime.now(timezone.utc).isoformat(),
                },
            ),
        ]
        embeddings = self._get_embeddings()
        self._vector_store = FAISS.from_documents(placeholder_docs, embeddings)
        self.faiss_path.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(self.faiss_path))
        logger.warning(
            "Placeholder FAISS index created. Add real PDFs to data/rag_docs/ and re-run."
        )


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingestor = AirSentinelIngestor()
    ingestor.run(force_reingest=False)
    ingestor.run_test_queries()
