from pathlib import Path
import re

from tqdm import tqdm

from underthesea import sent_tokenize
from langchain_core.documents import Document
import yaml
import numpy as np

WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


# Load configuration from YAML file
CONFIG_PATH = Path(__file__).resolve().with_name("prepocessing.yaml")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

threshold = config["threshold"]
min_chunk_size = config["min_chunk_size"]
max_chunk_size = config["max_chunk_size"]
overlap_size = config["overlap_size"]

class SemanticChunker:
    def __init__(
        self,
        model_name=None,
        embedding_model=None,
        threshold=threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        overlap_size=overlap_size,
        embedding_batch_size=512,
        document_batch_size=256,
    ):
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.embedding_batch_size = embedding_batch_size
        self.document_batch_size = document_batch_size
        if embedding_model is None:
            raise ValueError("SemanticChunker requires a preloaded embedding_model.")
        self.embedding_model = embedding_model

    def _split_into_sentences(self, text):
        sentences = sent_tokenize(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _compute_similarity(self, embedding1, embedding2):
        vec1 = np.asarray(embedding1, dtype=np.float32)
        vec2 = np.asarray(embedding2, dtype=np.float32)

        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / norm_product)

    def _embed_sentence_batches(self, sentences):
        if not sentences:
            return []

        all_embeddings = []
        for start in tqdm(
            range(0, len(sentences), self.embedding_batch_size),
            desc="Sentence Embedding",
        ):
            batch_sentences = sentences[start : start + self.embedding_batch_size]
            all_embeddings.extend(self.embedding_model.embed_documents(batch_sentences))
        return all_embeddings

    def _normalize_chunk_text(self, text):
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _is_valid_chunk(self, text):
        text = self._normalize_chunk_text(text)
        if len(text) < max(80, self.min_chunk_size // 2):
            return False

        words = WORD_PATTERN.findall(text)
        if len(words) < 8:
            return False

        non_space_chars = [char for char in text if not char.isspace()]
        if not non_space_chars:
            return False

        alpha_num_ratio = sum(char.isalnum() for char in non_space_chars) / len(non_space_chars)
        if alpha_num_ratio < 0.6:
            return False

        return True

    def _merge_small_chunks(self, chunks):
        if not chunks:
            return []

        merged_chunks = []
        for chunk in chunks:
            chunk = self._normalize_chunk_text(chunk)
            if not chunk:
                continue

            if not merged_chunks:
                merged_chunks.append(chunk)
                continue

            if len(chunk) < self.min_chunk_size:
                merged_chunks[-1] = self._normalize_chunk_text(f"{merged_chunks[-1]} {chunk}")
                continue

            if len(merged_chunks[-1]) < self.min_chunk_size:
                merged_chunks[-1] = self._normalize_chunk_text(f"{merged_chunks[-1]} {chunk}")
                continue

            merged_chunks.append(chunk)

        return merged_chunks

    def _chunk_by_semantic_similarity(self, sentences, sentence_embeddings):
        if not sentences:
            return []

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            pre_embedding = sentence_embeddings[i - 1]
            current_embedding = sentence_embeddings[i]
            similarity = self._compute_similarity(pre_embedding, current_embedding)
            chunk_text = self._normalize_chunk_text(' '.join(current_chunk))
            chunk_len = len(chunk_text)

            if chunk_len < self.min_chunk_size and chunk_len < self.max_chunk_size:
                current_chunk.append(sentences[i])
                continue

            if similarity < self.threshold or chunk_len >= self.max_chunk_size:
                chunks.append(chunk_text)
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(self._normalize_chunk_text(' '.join(current_chunk)))

        return self._merge_small_chunks(chunks)

    def _build_chunk_documents(self, doc, chunks):
        chunk_documents = []
        for idx, chunk_text in enumerate(chunks):
            chunk_text = self._normalize_chunk_text(chunk_text)
            if not chunk_text:
                continue
            if idx > 0 and self.overlap_size > 0:
                prev_chunk = chunks[idx - 1]
                if len(prev_chunk) > self.overlap_size:
                    overlap_text = prev_chunk[-self.overlap_size:]
                    chunk_text = self._normalize_chunk_text(overlap_text + " " + chunk_text)

            if not self._is_valid_chunk(chunk_text):
                continue

            chunk_documents.append(Document(page_content=chunk_text, metadata=doc.metadata))
        return chunk_documents

    def _iter_document_batches(self, documents):
        for start in range(0, len(documents), self.document_batch_size):
            yield documents[start : start + self.document_batch_size]

    def _process_document_batch(self, documents):
        prepared_docs = []
        flattened_sentences = []

        for doc in tqdm(documents, desc="Sentence Splitting", leave=False):
            sentences = self._split_into_sentences(doc.page_content)
            if not sentences:
                continue
            prepared_docs.append((doc, sentences))
            flattened_sentences.extend(sentences)

        all_embeddings = self._embed_sentence_batches(flattened_sentences)
        batch_chunks = []
        sentence_offset = 0

        for doc, sentences in tqdm(prepared_docs, desc="Semmantic Chunking", leave=False):
            sentence_count = len(sentences)
            sentence_embeddings = all_embeddings[
                sentence_offset : sentence_offset + sentence_count
            ]
            sentence_offset += sentence_count
            chunks = self._chunk_by_semantic_similarity(sentences, sentence_embeddings)
            batch_chunks.extend(self._build_chunk_documents(doc, chunks))

        return batch_chunks

    def split(self,document):
        all_chunks = []
        documents = list(document)
        total_batches = (len(documents) + self.document_batch_size - 1) // self.document_batch_size

        for document_batch in tqdm(
            self._iter_document_batches(documents),
            desc="Document Batches",
            total=total_batches,
        ):
            all_chunks.extend(self._process_document_batch(document_batch))

        return all_chunks
def main():
    from src.model.model_embedding import get_embedding_model

    sample_texts = [
        (
            "NVIDIA H100 hiện là GPU mạnh mẽ nhất cho việc huấn luyện AI với kiến trúc Hopper và bộ nhớ HBM3 tốc độ cao. Dòng chip này giúp giảm thời gian training các mô hình lớn xuống nhiều lần."
            "Llama-3 là một mô hình ngôn ngữ mã nguồn mở mới được Meta phát hành với khả năng suy luận"
            "vượt trội. Nó được tối ưu hóa để chạy hiệu quả ngay cả trên các thiết bị có tài nguyên hạn chế."
        )
    ]

    documents = [
        Document(page_content=text, metadata={"source": f"sample_{idx}.txt"})
        for idx, text in enumerate(sample_texts, start=1)
    ]

    embedding_model = get_embedding_model("AITeamVN/Vietnamese_Embedding")
    chunker = SemanticChunker(embedding_model=embedding_model)
    chunks = chunker.split(document=documents)

    print(f"Input documents: {len(documents)}")
    print(f"Output chunks: {len(chunks)}\n")

    for idx, chunk in enumerate(chunks, start=1):
        print(f"Chunk {idx}")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Length: {len(chunk.page_content)}")
        print(chunk.page_content)
        print("-" * 80)
if __name__ == "__main__":
    main()
