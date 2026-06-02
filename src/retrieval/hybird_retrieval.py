from collections import defaultdict

import numpy as np
from underthesea import word_tokenize


class HybridRetriever:
    def __init__(self, vector_db, bm25=None, documents=None, k=5, rrf_k=60):
        self.vector_db = vector_db
        self.bm25 = bm25
        self.documents = documents or []
        self.k = k
        self.rrf_k = rrf_k

    def get_relevant_documents(self, query):
        ranked_lists = []

        vector_results = self.vector_db.similarity_search(query, k=self.k)
        ranked_lists.append(vector_results)

        if self.bm25 is not None and self.documents:
            tokenized_query = word_tokenize(query.lower())
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_k_indices = np.argsort(bm25_scores)[::-1][: self.k]
            bm25_results = [self.documents[i] for i in top_k_indices]
            ranked_lists.append(bm25_results)

        fused_results = self._reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        return fused_results[: self.k]

    def invoke(self, query):
        return self.get_relevant_documents(query)

    def _reciprocal_rank_fusion(self, ranked_lists, k):
        rrf_scores = defaultdict(float)
        doc_content_map = {}

        for results in ranked_lists:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.page_content
                rrf_scores[doc_id] += 1 / (rank + k)
                doc_content_map[doc_id] = doc

        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_content_map[doc_id] for doc_id, _ in sorted_docs]
