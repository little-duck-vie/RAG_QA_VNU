from pathlib import Path
import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize


CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model_embedding import get_embedding_model
from src.retrieval.hybird_retrieval import HybridRetriever
from src.prepocessing.document_loader import normalize_metadata


class HybridVectorDB:
    def __init__(
        self,
        documents=None,
        embedding_model=None,
        persist_dir=None,
        collection_name="hybrid_retrieval_demo",
        load_existing=False,
    ):
        self.documents = documents or []
        self.embedding_model = embedding_model or get_embedding_model(
            "AITeamVN/Vietnamese_Embedding"
        )
        self.persist_dir = str(
            Path(persist_dir or (SRC_DIR / "data" / "vector_db")).resolve()
        )
        self.collection_name = collection_name
        self.load_existing = load_existing
        self.vector_db = self._build_vector_db(self.documents)
        self.bm25 = self._build_bm25_index(self.documents)

    def _sanitize_documents_for_vectorstore(self, documents):
        sanitized_documents = []
        for doc in documents or []:
            sanitized_documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata=normalize_metadata(doc.metadata),
                )
            )
        return sanitized_documents

    def _build_vector_db(self, documents=None):
        documents = documents or []
        if self.load_existing or not documents:
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_dir,
            )

        documents = self._sanitize_documents_for_vectorstore(documents)
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )

    def _build_bm25_index(self, documents=None):
        if not documents:
            return None

        tokenized_corpus = [word_tokenize(doc.page_content.lower()) for doc in documents]
        return BM25Okapi(tokenized_corpus)

    def get_retriever(self, search_kwargs=None):
        search_kwargs = search_kwargs or {"k": 5}
        return HybridRetriever(
            vector_db=self.vector_db,
            bm25=self.bm25,
            documents=self.documents,
            k=search_kwargs.get("k", 5),
            rrf_k=search_kwargs.get("rrf_k", 60),
        )


def main():
    documents = [
        Document(
            page_content="Hà Nội là thủ đô của Việt Nam và là trung tâm chính trị quan trọng.",
            metadata={"id": "doc_1", "topic": "dia_ly"},
        ),
        Document(
            page_content="Thành phố Hồ Chí Minh là đô thị lớn nhất Việt Nam về dân số.",
            metadata={"id": "doc_2", "topic": "dia_ly"},
        ),
        Document(
            page_content="Vịnh Hạ Long nổi tiếng với cảnh quan thiên nhiên và du lịch biển.",
            metadata={"id": "doc_3", "topic": "du_lich"},
        ),
    ]

    vector_db = HybridVectorDB(documents=documents)
    retriever = vector_db.get_retriever(search_kwargs={"k": 2, "rrf_k": 60})

    query = "Thủ đô của Việt Nam là gì?"
    results = retriever.get_relevant_documents(query)

    print(f"Query: {query}")
    print("Top results:")
    for index, doc in enumerate(results, start=1):
        print(f"{index}. {doc.page_content}")
        print(f"   metadata: {doc.metadata}")


if __name__ == "__main__":
    main()
