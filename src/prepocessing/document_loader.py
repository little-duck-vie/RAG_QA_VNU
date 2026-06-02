import json
import re
from pathlib import Path

from langchain_core.documents import Document


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text):
    if text is None:
        return ""
    return WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def normalize_metadata(metadata):
    normalized = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, list):
            normalized[key] = " | ".join(
                normalize_text(item) for item in value if normalize_text(item)
            )
            continue
        normalized[key] = normalize_text(value) if isinstance(value, str) else value
    return normalized


def _extract_doc_id(record, fallback):
    raw_id = record.get("_id")
    if isinstance(raw_id, dict) and "$oid" in raw_id:
        return str(raw_id["$oid"])
    if raw_id is not None:
        return str(raw_id)
    return fallback


def _extract_created_at(record):
    created_at = record.get("create_at")
    if isinstance(created_at, dict) and "$date" in created_at:
        return str(created_at["$date"])
    if created_at is None:
        return None
    return str(created_at)


def _iter_json_records(file_path):
    raw_text = file_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return

    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        index = 0
        text_length = len(raw_text)
        while index < text_length:
            while index < text_length and raw_text[index].isspace():
                index += 1

            if index >= text_length:
                break

            record, end_index = decoder.raw_decode(raw_text, index)
            yield record
            index = end_index
        return

    if isinstance(parsed, list):
        for item in parsed:
            yield item
        return

    if isinstance(parsed, dict):
        yield parsed


def load_documents_from_json_file(file_path):
    file_path = Path(file_path)
    documents = []

    for index, record in enumerate(_iter_json_records(file_path), start=1):
        if not isinstance(record, dict):
            continue

        title = normalize_text(record.get("title"))
        content = normalize_text(record.get("content"))
        if not content:
            continue

        full_text = content if not title else f"{title}\n\n{content}"
        metadata = {
            "doc_id": _extract_doc_id(record, f"{file_path.stem}_{index}"),
            "url": normalize_text(record.get("url")),
            "title": title,
            "domain": normalize_text(record.get("domain")),
            "category": record.get("category") or [],
            "created_at": _extract_created_at(record),
            "source_file": str(file_path),
        }
        documents.append(
            Document(
                page_content=full_text,
                metadata=normalize_metadata(metadata),
            )
        )

    return documents


def load_documents_from_directory(documents_dir):
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    documents = []
    for file_path in sorted(documents_dir.glob("*.json")):
        documents.extend(load_documents_from_json_file(file_path))

    if not documents:
        raise ValueError(f"No usable documents found in: {documents_dir}")

    return documents


def save_documents_to_jsonl(documents, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            record = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_documents_from_jsonl(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Processed document cache not found: {file_path}")

    documents = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            documents.append(
                Document(
                    page_content=record.get("page_content", ""),
                    metadata=normalize_metadata(record.get("metadata") or {}),
                )
            )

    if not documents:
        raise ValueError(f"No cached documents found in: {file_path}")

    return documents
