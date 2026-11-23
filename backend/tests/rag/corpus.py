"""Deterministic synthetic document corpus for M6-E RAG regression tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.app.rag.chunking import chunk_extraction
from backend.app.rag.extraction import extract_txt
from backend.app.rag.ingestion import ingest_document_bytes
from backend.app.rag.persistence import persist_document_chunks
from backend.app.rag.vector_store import VectorStore


@dataclass(frozen=True)
class CorpusDocument:
    company_id: str
    filename: str
    title: str
    doc_type: str
    text: str
    source_reference: str
    vector: tuple[float, ...]


# Vector basis: company axes 0-2, topic axes 3-7.
# Document vectors are normalized by the helper below.
CORPUS_DOCUMENTS: list[CorpusDocument] = [
    CorpusDocument(
        "aurora-retail",
        "aurora_overview.txt",
        "Aurora Retail Solutions - Company Overview",
        "overview",
        "Aurora Retail Solutions is a fictional mid-market retail company. Synthetic data only. "
        "In 2023 revenue was 60.0 million USD and EBITDA was 9.6 million USD. Operating margin was 16.0 percent. "
        "Cash was 8.0 million USD and total debt was 10.0 million USD. Working capital was 10.0 percent of revenue. "
        "No audited 2024 financial statements are available. No segment-level gross margin is disclosed.",
        "synthetic-corpus/aurora-retail/aurora_overview.txt",
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "aurora-retail",
        "aurora_financials.txt",
        "Aurora Retail Solutions - Financial Performance",
        "financial_summary",
        "Financial performance: 2021 revenue 50.0 million USD, 2022 revenue 53.0 million USD after restatement, "
        "and 2023 revenue 60.0 million USD. EBITDA was 6.0, 7.5, and 9.6 million USD. Operating income was 5.0, 6.4, "
        "and 8.4 million USD. Debt was 15.0, 12.0, and 10.0 million USD. Cash was 5.0, 6.0, and 8.0 million USD. "
        "Working capital increased 0.8 million USD in 2023. Missing information: segment-level gross margin.",
        "synthetic-corpus/aurora-retail/aurora_financials.txt",
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "aurora-retail",
        "aurora_management.txt",
        "Aurora Retail Solutions - Management Discussion",
        "management_discussion",
        "Management discussion: store expansion, inventory optimization, and digital sales contributed to 2023 growth. "
        "Operating cash flow was 10.0 million USD and capital expenditure was 1.8 million USD. "
        "Working capital increased 0.8 million USD. No 2024 guidance is provided. "
        + ("Aurora management discussion continues with additional operating detail. " * 90),
        "synthetic-corpus/aurora-retail/aurora_management.txt",
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "aurora-retail",
        "aurora_risks.txt",
        "Aurora Retail Solutions - Risk Assessment",
        "risk_memo",
        "Risk assessment: inventory obsolescence, supplier concentration, and interest-rate exposure. "
        "Total debt was 10.0 million USD at an estimated 6.0 percent. No hedging disclosure is provided. "
        "No audited 2024 financial statements are available.",
        "synthetic-corpus/aurora-retail/aurora_risks.txt",
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ),
    CorpusDocument(
        "aurora-retail",
        "aurora_supplemental.txt",
        "Aurora Retail Solutions - Supplemental Notes",
        "supplemental",
        "Supplemental note: 2022 revenue was 54.0 million USD, while the financial summary states 53.0 million USD. "
        "This discrepancy is unresolved and both figures remain documented. Synthetic data only.",
        "synthetic-corpus/aurora-retail/aurora_supplemental.txt",
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "borealis-industrial",
        "borealis_overview.txt",
        "Borealis Industrial Components - Company Overview",
        "overview",
        "Borealis Industrial Components is a fictional cyclical manufacturing company. Synthetic data only. "
        "In 2023 revenue was 85.0 million USD and EBITDA was 17.0 million USD. Operating margin was 20.0 percent. "
        "Cash was 5.0 million USD and total debt was 0.0 million USD. No audited 2024 financial statements are available.",
        "synthetic-corpus/borealis-industrial/borealis_overview.txt",
        (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "borealis-industrial",
        "borealis_financials.txt",
        "Borealis Industrial Components - Financial Performance",
        "financial_summary",
        "Financial performance: 2021 revenue 80.0 million USD, 2022 revenue 72.0 million USD, and 2023 revenue 85.0 million USD. "
        "EBITDA was 14.4, 5.76, and 17.0 million USD. Operating income was 11.2, -1.44, and 13.6 million USD. "
        "Debt was 30.0, 25.0, and 0.0 million USD. Cash was 3.0, 2.0, and 5.0 million USD. "
        "Missing information: segment-level revenue.",
        "synthetic-corpus/borealis-industrial/borealis_financials.txt",
        (0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "borealis-industrial",
        "borealis_management.txt",
        "Borealis Industrial Components - Management Discussion",
        "management_discussion",
        "Management discussion: cyclical demand, capacity utilization, and cost reduction affected 2022 results. "
        "Operating cash flow was 15.0 million USD and capital expenditure was 4.25 million USD in 2023. "
        "No 2024 guidance is provided.",
        "synthetic-corpus/borealis-industrial/borealis_management.txt",
        (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "borealis-industrial",
        "borealis_risks.txt",
        "Borealis Industrial Components - Risk Assessment",
        "risk_memo",
        "Risk assessment: cyclical demand, customer concentration, commodity prices, and fixed-cost leverage. "
        "A 2022 impairment reduced operating income. No hedging disclosure is provided. "
        "No audited 2024 financial statements are available.",
        "synthetic-corpus/borealis-industrial/borealis_risks.txt",
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ),
    CorpusDocument(
        "borealis-industrial",
        "borealis_supplemental.txt",
        "Borealis Industrial Components - Supplemental Notes",
        "supplemental",
        "Supplemental note: 2022 EBITDA was 6.0 million USD, while the financial summary states 5.76 million USD. "
        "This discrepancy is unresolved and both figures remain documented. Synthetic data only.",
        "synthetic-corpus/borealis-industrial/borealis_supplemental.txt",
        (0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "cedar-health",
        "cedar_overview.txt",
        "Cedar Health Analytics - Company Overview",
        "overview",
        "Cedar Health Analytics is a fictional health-technology analytics company. Synthetic data only. "
        "In 2023 revenue was 30.0 million EUR and EBITDA was 9.6 million EUR. Operating margin was 32.0 percent. "
        "Cash was 15.0 million EUR and total debt was 2.0 million EUR. Minority interest was 1.0 million EUR. "
        "No audited 2024 financial statements are available.",
        "synthetic-corpus/cedar-health/cedar_overview.txt",
        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "cedar-health",
        "cedar_financials.txt",
        "Cedar Health Analytics - Financial Performance",
        "financial_summary",
        "Financial performance: 2021 revenue 10.0 million EUR, 2022 revenue 18.0 million EUR, and 2023 revenue 30.0 million EUR. "
        "EBITDA was 2.5, 5.04, and 9.6 million EUR. Operating income was 2.2, 4.5, and 8.7 million EUR. "
        "Debt was 2.0 million EUR in all three years. Cash was 3.0, 8.0, and 15.0 million EUR. "
        "Missing information: segment-level margin.",
        "synthetic-corpus/cedar-health/cedar_financials.txt",
        (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "cedar-health",
        "cedar_management.txt",
        "Cedar Health Analytics - Management Discussion",
        "management_discussion",
        "Management discussion: customer adoption, recurring revenue, and product investment supported growth. "
        "Operating cash flow was 9.5 million EUR and capital expenditure was 1.8 million EUR in 2023. "
        "No 2024 guidance is provided.",
        "synthetic-corpus/cedar-health/cedar_management.txt",
        (0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    ),
    CorpusDocument(
        "cedar-health",
        "cedar_risks.txt",
        "Cedar Health Analytics - Risk Assessment",
        "risk_memo",
        "Risk assessment: customer concentration, regulatory changes, and currency exposure. "
        "Minority interest was 1.0 million EUR in 2022 and 2023. No hedging disclosure is provided. "
        "No audited 2024 financial statements are available.",
        "synthetic-corpus/cedar-health/cedar_risks.txt",
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ),
    CorpusDocument(
        "cedar-health",
        "cedar_supplemental.txt",
        "Cedar Health Analytics - Supplemental Notes",
        "supplemental",
        "Supplemental note: 2023 revenue was 31.0 million EUR, while the financial summary states 30.0 million EUR. "
        "This discrepancy is unresolved and both figures remain documented. Synthetic data only.",
        "synthetic-corpus/cedar-health/cedar_supplemental.txt",
        (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ),
]

CORPUS_DOCUMENTS = CORPUS_DOCUMENTS


def corpus_documents() -> list[CorpusDocument]:
    return list(CORPUS_DOCUMENTS)


def write_corpus(storage_root, documents=None) -> list[CorpusDocument]:
    root = Path(storage_root)
    root.mkdir(parents=True, exist_ok=True)
    docs = list(documents) if documents is not None else corpus_documents()
    for document in docs:
        (root / document.filename).write_text(document.text, encoding="utf-8")
    return docs


def ingest_corpus(conn, storage_root, documents=None):
    docs = list(documents) if documents is not None else corpus_documents()
    write_corpus(storage_root, docs)
    results = []
    for document in docs:
        result = ingest_document_bytes(
            conn,
            document.company_id,
            document.filename,
            document.text.encode("utf-8"),
            storage_root=storage_root,
            title=document.title,
            doc_type=document.doc_type,
            source_reference=document.source_reference,
        )
        results.append((document, result))
    return results


def corpus_vector(document: CorpusDocument) -> "np.ndarray":
    import numpy as np

    vector = np.asarray(document.vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("corpus vector must not be zero")
    return (vector / np.float32(norm)).astype(np.float32)


def build_corpus(
    conn,
    storage_root,
    index_path,
    documents=None,
    embedding_model="text-embedding-3-small",
    embedding_dim=8,
):

    docs = list(documents) if documents is not None else corpus_documents()
    ingested = ingest_corpus(conn, storage_root, docs)
    for document, result in ingested:
        extraction = extract_txt(document.text.encode("utf-8"))
        chunks = chunk_extraction(
            extraction,
            result.document_id,
            document.company_id,
            chunk_size=3200,
            chunk_overlap=400,
        )
        vector = corpus_vector(document)
        vectors = [vector for _ in chunks]
        persist_document_chunks(
            conn,
            result.document_id,
            document.company_id,
            chunks,
            vectors,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
    store = VectorStore(
        db_path=str(conn.execute("PRAGMA database_list").fetchone()["file"]),
        index_path=str(index_path),
        embedding_dim=embedding_dim,
        embedding_model=embedding_model,
    )
    store.rebuild_index()
    return ingested, store
