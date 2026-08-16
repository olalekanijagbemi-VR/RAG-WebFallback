"""
RAG_Web_Fallback.py
========================================================================
Multi-Agent RAG system with Web Fallback (Google Gemini free tier)
========================================================================

A single-file Streamlit application that:
  1. Ingests PDF / TXT / DOCX / CSV documents and chunks them.
  2. Builds a HYBRID index: scikit-learn NearestNeighbors (semantic) + 
     Pure Python BM25 (keyword), combined with weighted scoring 
     (60% semantic / 40% keyword).
  3. Uses a lightweight "Router Agent" (keyword heuristics) to decide,
     per query, whether to answer from the DOCUMENT index, the WEB
     (DuckDuckGo), or BOTH — with an explicit confidence score.
  4. Falls back to DuckDuckGo web search for time-sensitive / missing
     information.
  5. Generates a cited answer with Google Gemini (gemini-2.5-flash).

==========================================================================
FIX: Pure Python BM25 implementation (no external library dependency)
==========================================================================
"""

import os
import re
import time
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

# --- Vector / keyword search (Pure Python BM25) ---
from sklearn.neighbors import NearestNeighbors

# --- Chunking (pure-python, no torch) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Document parsers ---
import pypdf
import docx

# --- Env / secrets ---
from dotenv import load_dotenv

# --- Gemini SDK ---
from google import genai
from google.genai import types

# --- Web fallback ---
from ddgs import DDGS


# ==============================================================================
# CONFIG
# ==============================================================================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RAG_Web_Fallback")

APP_TITLE = "Multi-Agent RAG with Web Fallback"

CHUNK_SIZE_TOKENS = 1500
CHUNK_OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 4

EMBED_MODEL = "models/gemini-embedding-001"
GEN_MODEL = "models/gemini-3.5-flash"

SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

TOP_K_DOCS = 5
WEB_MAX_RESULTS = 5

TIME_SENSITIVE_KEYWORDS = [
    "today", "current", "currently", "latest", "news", "weather", "now",
    "this week", "this month", "this year", "recent", "recently", "live",
    "score", "stock price", "price of", "update", "breaking",
    "right now", "up to date", "real-time", "real time", "forecast",
    "who won", "what happened", "trending",
    # General knowledge triggers (add these)
    "where is", "what is", "who is", "capital of", "population of",
    "which country", "largest", "smallest", "tallest", "longest",
    "located in", "located at", "geography", "city", "country",
    "when was", "how to", "how does", "meaning of",
]

SYSTEM_PROMPT = """You are a precise, careful research assistant.

Rules:
1. Only use information contained in the SOURCES provided below.
2. Cite every factual claim using the matching [Source X] notation.
3. If the sources do not contain enough information to answer confidently,
   say so plainly instead of guessing.
4. If sources conflict, point out the discrepancy rather than picking one silently.
5. Be concise, well-organized, and use markdown formatting where useful.
"""


# ==============================================================================
# PURE PYTHON BM25 IMPLEMENTATION (No external library)
# ==============================================================================
class PureBM25:
    """
    Pure Python BM25 implementation.
    No external dependencies - works with any Python version.
    """
    
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with a corpus of tokenized documents.
        
        Args:
            corpus: List of tokenized documents (list of lists of tokens)
            k1: BM25 parameter (default 1.5)
            b: BM25 parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        
        # Calculate term frequencies across documents (IDF)
        self.idf = {}
        doc_freq = Counter()
        for doc in corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freq[term] += 1
        
        for term, freq in doc_freq.items():
            self.idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
        
        # Pre-compute document term frequencies for speed
        self.doc_term_freqs = [Counter(doc) for doc in corpus]
    
    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Get BM25 scores for a query against all documents."""
        scores = []
        for doc_idx in range(self.doc_count):
            score = 0.0
            doc_term_freq = self.doc_term_freqs[doc_idx]
            doc_len = self.doc_lengths[doc_idx]
            
            for term in query_tokens:
                if term not in self.idf:
                    continue
                idf = self.idf[term]
                tf = doc_term_freq.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores


# ==============================================================================
# API KEY / CLIENT
# ==============================================================================
def get_api_key() -> Optional[str]:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def get_client() -> Optional[genai.Client]:
    if st.session_state.get("client") is not None:
        return st.session_state.client
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        st.session_state.client = client
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None


# ==============================================================================
# EMBEDDINGS (Gemini, PyTorch-free)
# ==============================================================================
class GeminiEmbeddings:
    """Thin wrapper around the Gemini embedding API with basic retry logic."""

    def __init__(self, client: genai.Client, model: str = EMBED_MODEL):
        self.client = client
        self.model = model

    def _embed_one(self, text: str, task_type: str, max_retries: int = 3) -> List[float]:
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                return resp.embeddings[0].values
            except Exception as e:
                last_err = e
                logger.warning(f"Embedding attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Embedding failed after {max_retries} attempts: {last_err}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t, "RETRIEVAL_DOCUMENT") for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text, "RETRIEVAL_QUERY")


# ==============================================================================
# DOCUMENT LOADING
# ==============================================================================
def load_pdf(file) -> str:
    reader = pypdf.PdfReader(file)
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as e:
            logger.warning(f"Failed to extract text from PDF page {i}: {e}")
    return "\n".join(pages_text)


def load_docx(file) -> str:
    document = docx.Document(file)
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def load_txt(file) -> str:
    raw = file.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return raw


def load_csv(file) -> str:
    df = pd.read_csv(file)
    return df.to_string(index=False)


def parse_uploaded_file(file) -> Tuple[str, str]:
    ext = file.name.split(".")[-1].lower()
    if ext == "pdf":
        return load_pdf(file), ext
    elif ext == "docx":
        return load_docx(file), ext
    elif ext == "txt":
        return load_txt(file), ext
    elif ext == "csv":
        return load_csv(file), ext
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


# ==============================================================================
# CHUNKING
# ==============================================================================
def chunk_document(text: str, filename: str, filetype: str) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN,
        chunk_overlap=CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN,
        separators=["\n\n", "\n", ". ", ""],
    )
    raw_chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(raw_chunks):
        if not chunk.strip():
            continue
        docs.append({
            "id": f"{filename}::chunk_{i}",
            "text": chunk.strip(),
            "metadata": {
                "filename": filename,
                "filetype": filetype,
                "chunk_index": i,
            },
        })
    return docs


# ==============================================================================
# HYBRID VECTOR STORE (scikit-learn NearestNeighbors + Pure BM25)
# ==============================================================================
class HybridVectorStore:
    def __init__(self, embeddings: GeminiEmbeddings):
        self.embeddings = embeddings
        self.nn: Optional[NearestNeighbors] = None
        self.vectors: Optional[np.ndarray] = None
        self.chunks: List[Dict] = []
        self.bm25: Optional[PureBM25] = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def is_ready(self) -> bool:
        return self.nn is not None and len(self.chunks) > 0

    def add_documents(self, docs: List[Dict]) -> None:
        if not docs:
            return
        texts = [d["text"] for d in docs]
        vectors = self.embeddings.embed_documents(texts)
        arr = np.array(vectors, dtype="float32")
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / norms

        if self.nn is None:
            self.nn = NearestNeighbors(n_neighbors=min(10, len(arr)), metric='cosine')
            self.vectors = arr
        else:
            self.vectors = np.vstack([self.vectors, arr])
        
        self.nn.fit(self.vectors)
        self.chunks.extend(docs)

        # Rebuild BM25 over the full corpus using pure Python BM25
        tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = PureBM25(tokenized_corpus)

    def search(self, query: str, top_k: int = TOP_K_DOCS) -> List[Dict]:
        if not self.is_ready():
            return []

        # --- Semantic (cosine similarity via NearestNeighbors) ---
        q_vec = np.array([self.embeddings.embed_query(query)], dtype="float32")
        # Normalize query vector
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        
        k = min(top_k * 3, len(self.chunks))
        distances, indices = self.nn.kneighbors(q_vec, n_neighbors=k)
        
        # Convert distances to similarity scores (cosine)
        sem_map: Dict[int, float] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.chunks):
                continue
            similarity = 1.0 - dist
            sem_map[int(idx)] = float(similarity)

        # --- Keyword (Pure BM25) ---
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1.0
        kw_map = {i: float(s) / max_bm25 for i, s in enumerate(bm25_scores)}

        # --- Weighted combination ---
        all_idx = set(sem_map.keys()) | set(kw_map.keys())
        results = []
        for idx in all_idx:
            sem_s = sem_map.get(idx, 0.0)
            kw_s = kw_map.get(idx, 0.0)
            combined = SEMANTIC_WEIGHT * sem_s + KEYWORD_WEIGHT * kw_s
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "semantic_score": sem_s,
                "keyword_score": kw_s,
                "confidence": combined,
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results[:top_k]


# ==============================================================================
# ROUTER AGENT
# ==============================================================================
@dataclass
class RouteDecision:
    route: str          # "documents" | "web" | "both"
    confidence: float   # 0.0 - 1.0
    reason: str


def router_agent(query: str, docs_ready: bool) -> RouteDecision:
    q_lower = query.lower()
    matched = [kw for kw in TIME_SENSITIVE_KEYWORDS if kw in q_lower]

    if not docs_ready:
        return RouteDecision(
            route="web",
            confidence=0.90,
            reason="No documents indexed yet; routing straight to web search.",
        )

    if matched:
        confidence = min(0.60 + 0.10 * len(matched), 0.95)
        return RouteDecision(
            route="both",
            confidence=confidence,
            reason=f"Time-sensitive keyword(s) detected: {', '.join(matched)}. "
                   f"Checking both documents and the web.",
        )

    return RouteDecision(
        route="documents",
        confidence=0.75,
        reason="No time-sensitive keywords found; answering from indexed documents.",
    )


# ==============================================================================
# WEB FALLBACK (DuckDuckGo)
# ==============================================================================
def web_search(query: str, max_results: int = WEB_MAX_RESULTS) -> List[Dict]:
    results = []
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        n = max(len(hits), 1)
        for rank, hit in enumerate(hits):
            relevance = max(0.05, 1.0 - (rank / n) * 0.8)
            results.append({
                "title": hit.get("title", "Untitled"),
                "text": hit.get("body", ""),
                "url": hit.get("href", ""),
                "confidence": relevance,
            })
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        st.warning(f"⚠️ Web search failed: {e}")
    return results


# ==============================================================================
# ANSWER GENERATION
# ==============================================================================
def build_context(doc_results: List[Dict], web_results: List[Dict]) -> Tuple[str, List[Dict]]:
    sources = []
    context_parts = []
    idx = 1

    for r in doc_results:
        label = f"Source {idx}"
        meta = r["metadata"]
        context_parts.append(
            f"[{label}] (Document: {meta['filename']}, confidence: {r['confidence']:.0%})\n{r['text']}"
        )
        sources.append({
            "label": label,
            "type": "document",
            "confidence": r["confidence"],
            "filename": meta["filename"],
            "text": r["text"][:600],
        })
        idx += 1

    for r in web_results:
        label = f"Source {idx}"
        context_parts.append(
            f"[{label}] (Web: {r['title']} — {r['url']}, confidence: {r['confidence']:.0%})\n{r['text']}"
        )
        sources.append({
            "label": label,
            "type": "web",
            "confidence": r["confidence"],
            "title": r["title"],
            "url": r["url"],
            "text": r["text"][:600],
        })
        idx += 1

    return "\n\n".join(context_parts), sources


def generate_answer(client: genai.Client, query: str, context: str) -> str:
    if not context.strip():
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "No sources were retrieved for this question. Tell the user honestly "
            "that no relevant documents or web results were found, and only answer "
            "from general knowledge if you clearly label it as such (no citations).\n\n"
            f"Question: {query}"
        )
    else:
        prompt = (
            f"{SYSTEM_PROMPT}\n\nSOURCES:\n{context}\n\n"
            f"QUESTION: {query}\n\nANSWER (with [Source X] citations):"
        )

    try:
        resp = client.models.generate_content(model=GEN_MODEL, contents=prompt)
        return resp.text or "⚠️ The model returned an empty response."
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"⚠️ Error generating answer: {e}"


# ==============================================================================
# STREAMLIT APP
# ==============================================================================
def init_session_state():
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "indexed_files": [],
        "client": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def process_uploaded_files(files, client: genai.Client) -> int:
    if st.session_state.vector_store is None:
        st.session_state.vector_store = HybridVectorStore(GeminiEmbeddings(client))

    all_docs = []
    for f in files:
        if f.name in st.session_state.indexed_files:
            continue
        try:
            text, ext = parse_uploaded_file(f)
            if not text.strip():
                st.warning(f"No extractable text found in {f.name}; skipping.")
                continue
            docs = chunk_document(text, f.name, ext)
            all_docs.extend(docs)
            st.session_state.indexed_files.append(f.name)
        except Exception as e:
            logger.exception(f"Failed to process {f.name}")
            st.error(f"Failed to process {f.name}: {e}")

    if all_docs:
        with st.spinner(f"Embedding {len(all_docs)} chunks with Gemini..."):
            try:
                st.session_state.vector_store.add_documents(all_docs)
            except Exception as e:
                logger.exception("Failed to build index")
                st.error(f"Failed to build index: {e}")
                return 0
    return len(all_docs)


def render_sources(sources: List[Dict]):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            icon = "📄" if s["type"] == "document" else "🌐"
            header = f"{icon} **{s['label']}** — confidence {s['confidence']:.0%}"
            if s["type"] == "document":
                header += f" — *{s['filename']}*"
            else:
                header += f" — [{s['title']}]({s['url']})"
            st.markdown(header)
            st.caption(s["text"])
            st.markdown("---")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
    init_session_state()

    st.title("🔎 " + APP_TITLE)
    st.caption(
        "Hybrid document RAG (scikit-learn NearestNeighbors + BM25) with an automatic web-search "
        "fallback, powered by Google Gemini."
    )

    client = get_client()

    # ---------------------------------------------------------------- Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        if client:
            st.success("Gemini API connected")
        else:
            st.error("Gemini API key not set")
            st.info(
                "Set `GEMINI_API_KEY` as an environment variable, in a `.env` "
                "file, or in `.streamlit/secrets.toml`."
            )

        vs: Optional[HybridVectorStore] = st.session_state.vector_store
        n_chunks = len(vs.chunks) if vs else 0
        col1, col2 = st.columns(2)
        col1.metric("Indexed chunks", n_chunks)
        col2.metric("Indexed files", len(st.session_state.indexed_files))

        if st.session_state.indexed_files:
            with st.expander("Indexed files"):
                for fn in st.session_state.indexed_files:
                    st.write(f"- {fn}")

        st.divider()
        st.header("📄 Upload Documents")
        uploaded = st.file_uploader(
            "PDF, TXT, DOCX, CSV",
            type=["pdf", "txt", "docx", "csv"],
            accept_multiple_files=True,
        )
        if st.button("Process documents", type="primary", use_container_width=True, disabled=not uploaded):
            if not client:
                st.error("Cannot process documents without a valid Gemini API key.")
            else:
                added = process_uploaded_files(uploaded, client)
                if added:
                    st.success(f"Indexed {added} new chunk(s).")
                else:
                    st.info("No new chunks added (already indexed, empty, or failed).")

        st.divider()
        with st.expander("Advanced settings"):
            st.write(f"Chunk size: ~{CHUNK_SIZE_TOKENS} tokens, overlap: {CHUNK_OVERLAP_TOKENS} tokens")
            st.write(f"Semantic weight: {SEMANTIC_WEIGHT} · Keyword weight: {KEYWORD_WEIGHT}")
            st.write(f"Generation model: `{GEN_MODEL}`")
            st.write(f"Embedding model: `{EMBED_MODEL}`")
            c1, c2 = st.columns(2)
            if c1.button("Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
            if c2.button("Reset KB", use_container_width=True):
                st.session_state.vector_store = None
                st.session_state.indexed_files = []
                st.rerun()

    # ------------------------------------------------------------- Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("route_info"):
                ri = msg["route_info"]
                st.caption(f"🧭 Route: **{ri['route']}** ({ri['confidence']:.0%} confidence) — {ri['reason']}")
            if msg.get("sources"):
                render_sources(msg["sources"])

    # ------------------------------------------------------------------- Input
    query = st.chat_input("Ask a question...")
    if query:
        if not client:
            st.error("Please configure `GEMINI_API_KEY` before asking questions.")
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            vs = st.session_state.vector_store
            docs_ready = vs.is_ready() if vs else False
            decision = router_agent(query, docs_ready)
            st.caption(f"🧭 Route: **{decision.route}** ({decision.confidence:.0%} confidence) — {decision.reason}")

            doc_results: List[Dict] = []
            web_results: List[Dict] = []

            if decision.route in ("documents", "both") and docs_ready:
                with st.spinner("Searching documents (hybrid nearest-neighbors + BM25)..."):
                    try:
                        doc_results = vs.search(query, top_k=TOP_K_DOCS)
                    except Exception as e:
                        logger.exception("Document search failed")
                        st.warning(f"Document search failed: {e}")

            if decision.route in ("web", "both"):
                with st.spinner("Searching the web (DuckDuckGo)..."):
                    web_results = web_search(query)

            context, sources = build_context(doc_results, web_results)

            with st.spinner("Generating answer with Gemini..."):
                answer = generate_answer(client, query, context)

            st.markdown(answer)
            if sources:
                render_sources(sources)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "route_info": {
                    "route": decision.route,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                },
                "sources": sources,
            })


if __name__ == "__main__":
    main()