import argparse
from collections import Counter
import gc
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import sys

import torch

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_embedding import get_embedding_model
from model.model_llm import get_llm_model
from prepocessing.chunking import SemanticChunker
from prepocessing.document_loader import (
    load_documents_from_directory,
    load_documents_from_jsonl,
    save_documents_to_jsonl,
)
from retrieval.vectorDB import HybridVectorDB


LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
THRESHOLD = 0.65
MIN_CHUNK_SIZE = 250
MAX_CHUNK_SIZE = 1000
OVERLAP_SIZE = 100
VECTOR_DB_COLLECTION_NAME = "hybrid_retrieval_demo"
K = 5
RRF_K = 60
MAX_LLM_INPUT_TOKENS = 3072
MAX_CONTEXT_CHARS = 4000
DEFAULT_DOCUMENTS_DIR = (PROJECT_ROOT / "data" / "documents").resolve()
DEFAULT_QUESTION_FILE = (PROJECT_ROOT / "data" / "QA" / "questions.txt").resolve()
DEFAULT_REFERENCE_FILE = (PROJECT_ROOT / "data" / "QA" / "reference_answers.txt").resolve()
DEFAULT_OUTPUT_DIR = (PROJECT_ROOT / "system_outputs").resolve()
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "system_output.txt"
DEFAULT_DATA_DIR = (PROJECT_ROOT / "data").resolve()
VECTOR_DB_DIR = str((DEFAULT_DATA_DIR / "vector_db").resolve())
DEFAULT_PROCESSED_DIR = (DEFAULT_DATA_DIR / "processed").resolve()


def _load_query_module():
    module_path = SRC_DIR / "query" / "re-ranking.py"
    spec = spec_from_file_location("query_reranking", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUERY_MODULE = _load_query_module()
BatchRAG = QUERY_MODULE.BatchRAG
WHITESPACE_PATTERN = re.compile(r"\s+")


def reset_vector_store(vector_db_dir):
    vector_db_path = Path(vector_db_dir).resolve()
    allowed_root = DEFAULT_DATA_DIR
    if allowed_root not in vector_db_path.parents:
        raise ValueError(f"Refusing to delete vector db outside data directory: {vector_db_path}")
    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)


def get_processed_cache_path(use_chunking):
    filename = "chunked_documents.jsonl" if use_chunking else "documents.jsonl"
    return DEFAULT_PROCESSED_DIR / filename


def ensure_processed_documents(use_chunking, embeddings, rebuild_index, documents_dir):
    cache_path = get_processed_cache_path(use_chunking)
    if cache_path.exists() and not rebuild_index:
        print(f"Loading processed documents from cache: {cache_path}")
        return load_documents_from_jsonl(cache_path)

    print(f"Loading raw documents from: {documents_dir}")
    documents = load_documents_from_directory(documents_dir)
    processed_documents = documents

    if use_chunking:
        chunker = SemanticChunker(
            embedding_model=embeddings,
            threshold=THRESHOLD,
            min_chunk_size=MIN_CHUNK_SIZE,
            max_chunk_size=MAX_CHUNK_SIZE,
            overlap_size=OVERLAP_SIZE,
        )
        processed_documents = chunker.split(document=documents)

    save_documents_to_jsonl(processed_documents, cache_path)
    print(f"Saved processed documents cache to: {cache_path}")
    return processed_documents


def build_system(
    documents=None,
    documents_dir=DEFAULT_DOCUMENTS_DIR,
    use_chunking=True,
    use_reranker=False,
    reset_index=False,
    load_generation_stack=True,
):
    embeddings = get_embedding_model(model_name=EMBEDDING_MODEL)
    chunker = None

    if documents is None:
        processed_documents = ensure_processed_documents(
            use_chunking=use_chunking,
            embeddings=embeddings,
            rebuild_index=reset_index,
            documents_dir=documents_dir,
        )
    else:
        processed_documents = documents

    if reset_index:
        reset_vector_store(VECTOR_DB_DIR)

    vector_store_exists = Path(VECTOR_DB_DIR).exists()
    vdb = HybridVectorDB(
        documents=processed_documents,
        embedding_model=embeddings,
        persist_dir=VECTOR_DB_DIR,
        collection_name=VECTOR_DB_COLLECTION_NAME,
        load_existing=vector_store_exists and not reset_index,
    )
    retriever = vdb.get_retriever(search_kwargs={"k": K, "rrf_k": RRF_K})

    if not load_generation_stack:
        return {
            "documents": processed_documents,
            "chunker": chunker,
            "vector_db": vdb,
            "retriever": retriever,
            "rag": None,
        }

    del embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    llm, _ = get_llm_model(
        model_name=LLM_MODEL,
        max_input_tokens=MAX_LLM_INPUT_TOKENS,
    )

    reranker = None
    if use_reranker:
        reranker = QUERY_MODULE.CrossEncoderReranker(model_name=RERANKER_MODEL, top_k=K)

    rag = BatchRAG(
        llm=llm,
        reranker=reranker,
        batch_size=2,
        max_context_chars=MAX_CONTEXT_CHARS,
    )

    return {
        "documents": processed_documents,
        "chunker": chunker,
        "vector_db": vdb,
        "retriever": retriever,
        "rag": rag,
    }


def answer_questions(system, questions):
    return system["rag"].answer_with_contexts_batch(
        questions=questions,
        retriever=system["retriever"],
    )


def print_results(results):
    for index, result in enumerate(results, start=1):
        print(f"Cau hoi {index}: {result['question']}")
        print(f"Tra loi: {result['answer']}")
        print("-" * 60)


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(item["answer"]).strip() for item in results]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_questions_from_file(file_path):
    question_file = Path(file_path)
    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    questions = [
        line.strip()
        for line in question_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise ValueError(f"Question file is empty: {question_file}")
    return questions


def load_reference_answers(file_path):
    reference_file = Path(file_path)
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference answer file not found: {reference_file}")

    references = []
    for line in reference_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            references.append([])
            continue
        references.append([item.strip() for item in line.split(";") if item.strip()])
    return references


def normalize_answer(text):
    text = (text or "").strip().lower()
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def exact_match_score(prediction, ground_truth):
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_recall_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    return overlap / len(gold_tokens)


def evaluate_predictions(results, reference_answers):
    predictions = [item["answer"] for item in results]
    if len(predictions) != len(reference_answers):
        raise ValueError(
            f"Prediction/reference size mismatch: {len(predictions)} vs {len(reference_answers)}"
        )

    exact_match_total = 0.0
    f1_total = 0.0
    recall_total = 0.0

    for prediction, references in zip(predictions, reference_answers):
        references = references or [""]
        exact_match_total += max(exact_match_score(prediction, ref) for ref in references)
        f1_total += max(token_f1_score(prediction, ref) for ref in references)
        recall_total += max(token_recall_score(prediction, ref) for ref in references)

    sample_count = len(predictions) or 1
    return {
        "exact_match": exact_match_total / sample_count,
        "f1": f1_total / sample_count,
        "answer_recall": recall_total / sample_count,
    }


def run_on_documents(
    documents_dir=DEFAULT_DOCUMENTS_DIR,
    questions=None,
    question_file=None,
    reference_file=None,
    use_chunking=True,
    use_reranker=False,
    reset_index=False,
    output_path=None,
    build_index_only=False,
):
    questions = questions or []
    if question_file:
        questions = load_questions_from_file(question_file)

    system = build_system(
        documents=None,
        documents_dir=documents_dir,
        use_chunking=use_chunking,
        use_reranker=use_reranker,
        reset_index=reset_index,
        load_generation_stack=not build_index_only and bool(questions),
    )

    if build_index_only:
        print("Index rebuild completed.")
        return []

    if not questions:
        print("Index/cache is ready. No questions were provided, so QA generation was skipped.")
        return []

    results = answer_questions(system, questions)
    print_results(results)

    if output_path:
        save_results(results, output_path)

    if reference_file:
        metrics = evaluate_predictions(results, load_reference_answers(reference_file))
        print(
            "Metrics: "
            f"EM={metrics['exact_match']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"Recall={metrics['answer_recall']:.4f}"
        )

    return results


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the RAG QA system on JSON documents.")
    parser.add_argument(
        "--documents-dir",
        default=str(DEFAULT_DOCUMENTS_DIR),
        help="Directory containing crawled JSON documents.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Question to ask the RAG system. Repeat this flag for multiple questions.",
    )
    parser.add_argument(
        "--question-file",
        default=str(DEFAULT_QUESTION_FILE),
        help="UTF-8 text file containing one question per line.",
    )
    parser.add_argument(
        "--reference-file",
        default=str(DEFAULT_REFERENCE_FILE),
        help="UTF-8 text file containing one or more reference answers per line, separated by semicolons.",
    )
    parser.add_argument(
        "--build-index-only",
        action="store_true",
        help="Only rebuild/load processed documents and vector index, without loading the LLM or answering questions.",
    )
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Disable semantic chunking and index full documents directly.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable reranking after retrieval.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Re-run preprocessing/chunking and rebuild the persisted vector index.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to save text answers.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run_on_documents(
        documents_dir=args.documents_dir,
        questions=args.question,
        question_file=args.question_file,
        reference_file=args.reference_file,
        use_chunking=not args.no_chunking,
        use_reranker=args.use_reranker,
        reset_index=args.rebuild_index,
        output_path=args.output,
        build_index_only=args.build_index_only,
    )


if __name__ == "__main__":
    main()
