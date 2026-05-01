import re
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: int

def sanitize_unicode(text: str) -> str:
    """
    清理會讓 tokenizer 炸掉的非法 unicode：
    - 孤立 surrogate (U+D800-U+DFFF 沒配對的)
    - NULL byte
    - 其他控制字元（保留 \n \t \r）
    保留所有合法字元，包括數學符號、中文、emoji。
    """
    # 1. 用 'replace' 策略過濾掉孤立 surrogate
    #    encode 成 utf-8 時 surrogateescape 會把它們替換掉
    text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    
    # 2. 移除 NULL 和危險控制字元，但保留 \n \t \r
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    return text

def recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """
    遞迴切分：依序試 separators，找到能讓每塊 <= chunk_size 的切法。
    separators 從大到小，例如 ["\n\n", "\n", ". ", " ", ""]
    """
    if len(text) <= chunk_size:
        return [text]

    # 嘗試當前最大粒度的 separator
    separator = separators[0]
    remaining = separators[1:] if len(separators) > 1 else [""]

    if separator == "":
        # 最後手段：硬切
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    splits = text.split(separator)
    chunks = []
    current = ""

    for s in splits:
        candidate = current + (separator if current else "") + s
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # 如果單一 split 還是太大，遞迴用更細的 separator
            if len(s) > chunk_size:
                chunks.extend(recursive_split(s, chunk_size, remaining))
                current = ""
            else:
                current = s

    if current:
        chunks.append(current)

    return chunks


def add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """讓相鄰 chunks 重疊 overlap 個字元"""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + chunks[i])
    return overlapped


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    對每頁文字做 chunking，保留頁碼資訊（之後 retrieval 要顯示來源）。
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    all_chunks = []
    chunk_id = 0

    for page in pages:
        raw_chunks = recursive_split(page["text"], chunk_size, separators)
        overlapped = add_overlap(raw_chunks, overlap)

        for text in overlapped:
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            text = sanitize_unicode(text)
            if len(text) < 50:  # 過濾太短的雜訊（頁碼、頁首等）
                continue
            # 過濾掉控制字元、不可見字元堆積的雜訊
            printable_ratio = sum(1 for c in text if c.isprintable() or c.isspace()) / len(text)
            if printable_ratio < 0.8:
                continue
            
            all_chunks.append(Chunk(
                text=text,
                page=page["page"],
                chunk_id=chunk_id,
            ))
            chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    from src.pdf_loader import load_pdf

    pages = load_pdf("/app/data/test.pdf")
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)

    print(f"總 chunk 數: {len(chunks)}")
    print(f"平均長度: {sum(len(c.text) for c in chunks) / len(chunks):.0f} 字")
    print(f"\n--- Chunk 0 (第 {chunks[0].page} 頁) ---")
    print(chunks[0].text)
    print(f"\n--- Chunk 1 (第 {chunks[1].page} 頁) ---")
    print(chunks[1].text)
    print(f"\n--- Chunk 5 (第 {chunks[5].page} 頁) ---")
    print(chunks[5].text)