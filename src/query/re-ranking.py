import re
from pathlib import Path
import sys

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sentence_transformers import CrossEncoder
from tqdm import tqdm


CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model_llm import get_llm_model


class FocusedAnswerParser(StrOutputParser):
    def parse(self, text: str) -> str:
        text = (text or "").strip()
        if "[TRẢ LỜI]:" in text:
            answer = text.split("[TRẢ LỜI]:", 1)[1].strip()
        else:
            answer = text

        stop_markers = [
            "Bạn là trợ lý AI",
            "[TÀI LIỆU]:",
            "[CÂU HỎI]:",
            "Ngu canh:",
            "Câu hỏi:",
            "Tra loi:",
        ]
        for marker in stop_markers:
            if marker in answer:
                answer = answer.split(marker, 1)[0].strip()

        answer = re.sub(r"^\s*[\u2022\-\*]\s*", "", answer, flags=re.MULTILINE)
        answer = re.sub(r"\s+", " ", answer).strip()
        answer = answer.split("\n", 1)[0].strip()

        sentence_match = re.match(r"^(.{1,200}?[.!?])(?:\s|$)", answer)
        if sentence_match:
            answer = sentence_match.group(1).strip()

        answer = re.sub(r"(?:\s*[-:]\s*)?$", "", answer).strip()
        return answer or "Không có thông tin"


class CrossEncoderReranker:
    def __init__(self, model_name, top_k=3, max_length=512):
        self.model_name = model_name
        self.top_k = top_k
        self.max_length = max_length
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            trust_remote_code=True,
        )

    def rerank(self, query, documents):
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=min(16, len(pairs)),
            show_progress_bar=False,
        )
        scores = [float(score) for score in scores]

        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in doc_score_pairs[: self.top_k]]


class BatchRAG:
    def __init__(self, llm, reranker=None, batch_size=8, max_context_chars=4000):
        self.llm = llm
        self.reranker = reranker
        self.batch_size = batch_size
        self.max_context_chars = max_context_chars
        self.answer_parser = FocusedAnswerParser()
        self.prompt_template = """
Bạn là trợ lý AI phân tích tài liệu tiếng Việt.

[TÀI LIỆU]:
{context}

[CÂU HỎI]:
{question}

Hãy trả lời dựa trên tài liệu.
Chỉ trả lời đúng phần đáp án ngắn gọn trong một dòng.
Không giải thích.
Không lặp lại câu hỏi.
Không chép lại tài liệu.
Nếu tài liệu không đủ thông tin, trả lời đúng: Không có thông tin.

[TRẢ LỜI]:"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)

    def _format_docs(self, docs):
        formatted = []
        seen = set()
        total_chars = 0
        for doc in docs:
            content = (doc.page_content or "").strip()
            if content and content not in seen:
                projected_chars = total_chars + len(content)
                if formatted and projected_chars > self.max_context_chars:
                    break
                formatted.append(content)
                seen.add(content)
                total_chars += len(content)
        return "\n\n".join(formatted)

    def get_chain(self, retriever):
        format_docs = RunnableLambda(self._format_docs)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | self.answer_parser
        )
        return rag_chain

    def batch_retrieve(self, questions, retriever):
        all_contexts = []
        for question in tqdm(questions, desc="Batch Retrieval"):
            docs = retriever.invoke(question)
            if self.reranker is not None:
                docs = self.reranker.rerank(question, docs)

            contexts = [(doc.page_content or "").strip() for doc in docs]
            formatted_context = self._format_docs(docs)
            all_contexts.append(
                {
                    "question": question,
                    "context": contexts,
                    "format_context": formatted_context,
                }
            )
        return all_contexts

    def _normalize_llm_output(self, output):
        if hasattr(output, "content"):
            return output.content
        return str(output)

    def batch_generate(self, prompts):
        all_answers = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="Batch Generation"):
            batch_prompts = prompts[start : start + self.batch_size]
            for prompt in batch_prompts:
                raw_output = self.llm.invoke(prompt)
                parsed_answer = self.answer_parser.parse(
                    self._normalize_llm_output(raw_output)
                )
                all_answers.append(parsed_answer)
        return all_answers

    def answer_with_contexts_batch(self, questions, retriever):
        retrieved_data = self.batch_retrieve(questions, retriever)
        prompts = [
            self.prompt.format(
                context=data["format_context"],
                question=data["question"],
            )
            for data in retrieved_data
        ]
        answers = self.batch_generate(prompts)

        results = []
        for data, answer in zip(retrieved_data, answers):
            results.append(
                {
                    "question": data["question"],
                    "answer": answer,
                    "context": data["context"],
                }
            )
        return results


class PipelineLLMAdapter:
    def __init__(self, text_generation_pipeline):
        self.pipeline = text_generation_pipeline
        self.tokenizer = text_generation_pipeline.tokenizer

    def invoke(self, prompt):
        outputs = self.pipeline(
            prompt,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()


class SimpleRetriever:
    def __init__(self, documents, k=2):
        self.documents = documents
        self.k = k

    def invoke(self, query):
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored_docs = []

        for doc in self.documents:
            doc_terms = set(re.findall(r"\w+", doc.page_content.lower()))
            overlap = len(query_terms & doc_terms)
            scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[: self.k]]

    def __or__(self, other):
        return RunnableLambda(lambda query: other.invoke(self.invoke(query)))


def main():
    documents = [
        Document(
            page_content="Hà Nội là thủ đô của Việt Nam và là trung tâm chính trị, văn hóa quan trọng.",
            metadata={"id": "doc_1"},
        ),
        Document(
            page_content="Thành phố Hồ Chí Minh là thành phố lớn nhất Việt Nam về dân số và kinh tế.",
            metadata={"id": "doc_2"},
        ),
        Document(
            page_content="Đà Nẵng là thành phố biển nổi tiếng ở miền Trung Việt Nam.",
            metadata={"id": "doc_3"},
        ),
    ]

    retriever = SimpleRetriever(documents, k=2)
    llm_pipeline, _ = get_llm_model()
    rag = BatchRAG(llm=PipelineLLMAdapter(llm_pipeline), reranker=None, batch_size=2)

    questions = [
        "Thủ đô của Việt Nam là gì?",
        "Thành phố lớn nhất Việt Nam là thành phố nào?",
        "Đỉnh núi cao nhất Việt Nam là gì?",
    ]

    results = rag.answer_with_contexts_batch(questions, retriever)
    for index, result in enumerate(results, start=1):
        print(f"Cau hoi {index}: {result['question']}")
        print(f"Tra loi: {result['answer']}")
        print("Ngu canh:")
        for context in result["context"]:
            print(f"- {context}")
        print("-" * 60)


if __name__ == "__main__":
    main()
