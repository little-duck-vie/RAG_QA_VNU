from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from langchain_core import Document
from langchain_core.retrievers import BaseRetrieval
from pydantic import Field
from typing import List, Any
import numpy as np

class BM25Retriever(BaseRetrieval):
    @classmethod
    def from_documents(cls, documents: List[Document], k: int = 5):
        tokenized_corpus = [word_tokenize(doc.page_content.lower()) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(bm25=bm25, documents=documents, k=k)
    def _get_relevant_documents(self, query: str) -> List[Document]:
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(bm25_scores)[::-1][:self.k]
        return [self.documents[i] for i in top_k_indices]