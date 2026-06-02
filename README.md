# RAG VNU Project

Hệ thống hỏi đáp tiếng Việt theo kiến trúc RAG cho dữ liệu Đại học Quốc gia Hà Nội và các trường thành viên.

Repo hiện tại hỗ trợ:

- Crawl dữ liệu HTML từ một số website chính thức của VNU
- Tiền xử lý và semantic chunking
- Hybrid retrieval bằng `Chroma + BM25 + RRF`
- Tùy chọn reranking trước khi sinh câu trả lời
- Sinh câu trả lời bằng `Qwen/Qwen2.5-7B-Instruct`
- Chấm điểm đầu ra bằng `Exact Match`, `F1`, `Answer Recall`
- Chạy baseline chỉ dùng LLM để so sánh với RAG

## 1. Cấu trúc chính

```text
RAG/
├── data/
│   ├── documents/              # Dữ liệu tri thức dạng JSON/JSONL
│   ├── QA/
│   │   ├── train/
│   │   └── test/
│   └── vector_db/              # Chroma persist directory
└── src/
    ├── crawl/crawl.py
    ├── prepocessing/
    ├── retrieval/
    ├── query/re-ranking.py
    ├── model/
    ├── run_system.py
```

## 2. Yêu cầu môi trường

- Python 3.10+ khuyến nghị
- GPU là tốt nhất nếu chạy `Qwen/Qwen2.5-7B-Instruct`
- Windows PowerShell hoặc terminal tương đương

Cài thư viện:

```bash
pip install -r src/requirements.txt
```

## 3. Dữ liệu hiện có

Repo đã có sẵn dữ liệu tri thức trong `data/documents/` và dữ liệu đánh giá trong:

- `data/QA/train/questions.txt`
- `data/QA/train/reference_answers.txt`
- `data/QA/test/questions.txt`
- `data/QA/test/reference_answers.txt`


## 4. Chạy hệ RAG

### 4.1. Chỉ build index

```bash
python src/run_system.py --build-index-only --rebuild-index
```

Lệnh này sẽ:

- Nạp dữ liệu từ `data/documents`
- Chunk tài liệu nếu bật chunking
- Tạo lại Chroma DB trong `data/vector_db`

### 4.2. Chạy RAG trên tập test

```bash
python src/run_system.py ^
  --documents-dir data/documents ^
  --question-file data/QA/test/questions.txt ^
  --output data/QA/test/system_output.txt
```

### 4.3. Chạy RAG có reranker

```bash
python src/run_system.py ^
  --documents-dir data/documents ^
  --question-file data/QA/test/questions.txt ^
  --use-reranker ^
  --output data/QA/test/system_output_rerank.txt
```



## 5. Các file quan trọng

- `src/run_system.py`: entry point chính của hệ RAG
- `src/crawl/crawl.py`: crawler HTML
- `src/prepocessing/document_loader.py`: nạp và chuẩn hóa tài liệu
- `src/prepocessing/chunking.py`: semantic chunking
- `src/retrieval/vectorDB.py`: Chroma + BM25
- `src/retrieval/hybird_retrieval.py`: hybrid retrieval bằng RRF
- `src/query/re-ranking.py`: reranking và BatchRAG
- `src/model/model_llm.py`: load LLM
- `src/model/model_embedding.py`: load embedding model

## 6. Mô hình đang dùng

- Embedding model: `AITeamVN/Vietnamese_Embedding`
- LLM: `Qwen/Qwen2.5-7B-Instruct`