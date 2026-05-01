"""一鍵建索引：讀 PDF → chunk → embed → 存"""
import sys
from src.pdf_loader import load_pdf
from src.chunker import chunk_pages
from src.embedder import embed_and_store, reset_collection

def build(pdf_path: str, collection_name: str = "papers", reset: bool = True):
    print(f"📄 讀取 {pdf_path}")
    pages = load_pdf(pdf_path)
    print(f"   {len(pages)} 頁")

    print(f"✂️  切分 chunks...")
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    print(f"   {len(chunks)} 個 chunks")

    if reset:
        reset_collection(collection_name)

    print(f"🔢 生成 embedding 並存入 ChromaDB...")
    count = embed_and_store(chunks, collection_name)
    print(f"✅ 完成！共 {count} 個向量")

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/app/data/test.pdf"
    build(pdf_path)