from pypdf import PdfReader

def load_pdf(file_path: str) -> list[dict]:
    """讀 PDF，回傳 [{page: int, text: str}, ...]"""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            print(f"⚠️ 第 {i+1} 頁解析失敗: {e}")
            continue

        # 三重檢查：非 None、是 str、去空白後非空
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            continue

        pages.append({"page": i + 1, "text": text})
    return pages

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/app/data/test.pdf"
    pages = load_pdf(path)
    print(f"共 {len(pages)} 頁有有效內容")
    if pages:
        print(f"第一頁前 200 字: {pages[0]['text'][:200]}")