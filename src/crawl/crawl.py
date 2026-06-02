"""Simple domain-restricted crawler for VNU and its member universities.

Saves extracted pages as JSONL (one JSON object per line) under `data/documents/` 
matching the target format framework exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

# Bỏ qua cảnh báo SSL và cảnh báo XML
import urllib3
import warnings
from bs4 import XMLParsedAsHTMLWarning

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "documents"

# Sử dụng User-Agent chuẩn
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

SEEDS = {
    "ussh": "https://ussh.vnu.edu.vn",
    "ulis": "https://ulis.vnu.edu.vn",
    "ump": "https://ump.vnu.edu.vn",
    "ul": "https://law.vnu.edu.vn",
    "vnu": "https://vnu.edu.vn",
}

def is_same_domain(seed: str, url: str) -> bool:
    try:
        s = urllib.parse.urlparse(seed).netloc
        u = urllib.parse.urlparse(url).netloc
        return u.endswith(s) or s.endswith(u)
    except Exception:
        return False

def normalize_url(base: str, link: str) -> Optional[str]:
    if not link:
        return None
    link = link.split("#")[0]
    parsed = urllib.parse.urljoin(base, link)
    if parsed.startswith("mailto:") or parsed.startswith("javascript:"):
        return None
    return parsed

def fetch_url(url: str, timeout: int = 10) -> Optional[requests.Response]:
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=timeout, verify=False, stream=True)
        if r.status_code == 200:
            # Lọc bỏ file PDF/nhị phân khổng lồ ngay từ lúc nhận Header
            content_type = r.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type:
                return r
    except Exception:
        return None
    return None

def extract_text_from_html(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Khử toàn bộ thành phần rác giao diện
    for noise in soup(["header", "footer", "nav", "script", "style", "noscript", "aside"]):
        noise.decompose()
    for noise in soup.find_all("div", id=re.compile(r"header|footer|menu|sidebar|navigation", re.I)):
        noise.decompose()
    for noise in soup.find_all("div", class_=re.compile(r"header|footer|menu|sidebar|navigation", re.I)):
        noise.decompose()

    # Tìm nội dung chính
    main = soup.find("article") or soup.find("main")
    if main:
        # SỬA ĐỔI: Thay "\n".join bằng " ".join để ép toàn bộ văn bản thành một chuỗi phẳng không chứa Enter
        text = " ".join(p.get_text(separator=" ", strip=True) for p in main.find_all("p") if p.get_text(strip=True))
        if text.strip():
            return title, _clean_text(text)

    # Lựa chọn khối div lớn nếu không có thẻ article/main
    candidates = soup.find_all(["div", "section"])
    best = None
    best_len = 0
    for c in candidates:
        txt = c.get_text(separator=" ", strip=True)
        ln = len(txt)
        if ln > best_len:
            best_len = ln
            best = txt
    if best_len > 50:
        return title, _clean_text(best)

    # Fallback cuối cùng: Thay "\n".join bằng " ".join
    allp = " ".join(p.get_text(separator=" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True))
    return title, _clean_text(allp)


def _clean_text(s: str) -> str:
    # SỬA ĐỔI TRỌNG TÂM: Biến tất cả các khoảng trắng, dấu tab, phím Enter thực tế (\r, \n) thành 1 khoảng trắng duy nhất
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_doc_entry(url: str, title: str, content: str, domain: str) -> Dict:
    # BẮT BUỘC THÊM: Chuẩn hóa xóa dấu gạch chéo rớt lại ở cuối URL để đồng bộ chuỗi hoàn toàn
    normalized_url = url.strip().rstrip("/")
    
    # Thực hiện băm SHA-1 trên chuỗi đã được chuẩn hóa phẳng
    uid = hashlib.sha1(normalized_url.encode("utf-8")).hexdigest()
    now = datetime.utcnow().isoformat() + "Z"
    
    return {
        "_id": {"$oid": uid},
        "url": normalized_url, # Lưu trữ URL đã được làm sạch đồng bộ
        "title": title,
        "content": content,
        "domain": domain,
        "category": [],
        "create_at": {"$date": now},
        "_class": "com.news.scanner.entity.News"
    }

def save_output(docs: List[Dict], outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    tmp = outpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        # CẬP NHẬT QUAN TRỌNG: Lưu dưới định dạng JSON Lines (JSONL)
        # Ép mỗi document thành 1 dòng duy nhất, không dùng indent
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    os.replace(tmp, outpath)

def crawl(seed: str, limit: int = 2500, delay: float = 0.5) -> List[Dict]:
    q = queue.Queue()
    q.put((seed, 0))
    visited: Set[str] = set()
    docs: List[Dict] = []

    while not q.empty() and len(docs) < limit:
        url, depth = q.get()
        if url in visited:
            continue
        visited.add(url)

        time.sleep(delay)
        resp = fetch_url(url)
        if resp is None:
            continue

        title, content = extract_text_from_html(resp.text)
        if content and len(content) > 50:
            # Truyền lại biến domain vào hàm
            docs.append(make_doc_entry(url, title, content, urllib.parse.urlparse(seed).netloc))

        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.find_all("a", href=True):
            n = normalize_url(url, a.get("href"))
            if not n:
                continue
            if not is_same_domain(seed, n):
                continue
            if n in visited:
                continue
            
            # Chặn các file rác làm phình dung lượng
            if re.search(r"\.(jpg|jpeg|png|gif|svg|css|js|pdf|doc|docx|xls|xlsx|zip|rar|mp4|ppt|pptx|xml|json)$", n, re.I):
                continue
            q.put((n, depth + 1))

    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--school", choices=["ussh", "ulis", "ump", "ul", "vnu", "all"], default="all")
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if args.school == "all":
        targets = ["ussh", "ulis", "ump", "ul", "vnu"]
    else:
        targets = [args.school]

    for t in targets:
        seed = SEEDS.get(t)
        if not seed:
            continue
        print(f"Crawling {t} from {seed} (limit={args.limit})...")
        docs = crawl(seed, limit=args.limit, delay=args.delay)
        outpath = os.path.join(args.output_dir, f"{t}.json")
        save_output(docs, outpath)
        print(f"Saved {len(docs)} documents to {outpath}")

if __name__ == "__main__":
    main()