from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from src.chunker import Chunk

# 全域只載入一次模型（cold start 約 5-10 秒）
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("載入 embedding 模型中...")
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("模型載入完成")
    return _model


def get_chroma_client():
    """ChromaDB 持久化 client"""
    return chromadb.PersistentClient(
        path="/app/chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(name: str = "papers"):
    """取得或建立 collection"""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # 用 cosine similarity
    )


def embed_and_store(chunks: list[Chunk], collection_name: str = "papers") -> int:
    """把 chunks embedding 後存進 ChromaDB"""
    model = get_model()
    collection = get_or_create_collection(collection_name)

    # 過濾掉非字串或空字串（最後一道防線）
    valid_chunks = [
        c for c in chunks
        if isinstance(c.text, str) and c.text.strip()
    ]
    skipped = len(chunks) - len(valid_chunks)
    if skipped > 0:
        print(f"⚠️ 跳過 {skipped} 個無效 chunks")

    if not valid_chunks:
        print("❌ 沒有有效 chunks，停止")
        return 0

    texts = [c.text for c in valid_chunks]
    chunks = valid_chunks  # 後面 ids/metadatas 也要用過濾後的
    
    print(f"對 {len(texts)} 個 chunks 做 embedding...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # ChromaDB 需要每筆有唯一 id
    ids = [f"chunk_{c.chunk_id}" for c in chunks]
    metadatas = [{"page": c.page, "chunk_id": c.chunk_id} for c in chunks]

    collection.upsert(  # upsert 而非 add，重跑不會炸
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"已存入 {collection.count()} 個向量到 collection '{collection_name}'")
    return collection.count()


def reset_collection(collection_name: str = "papers"):
    """清空 collection（之後要換論文時用）"""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        print(f"已刪除 collection '{collection_name}'")
    except Exception as e:
        print(f"Collection 不存在或已被刪除: {e}")


if __name__ == "__main__":
    from src.pdf_loader import load_pdf
    from src.chunker import chunk_pages

    pages = load_pdf("/app/data/test.pdf")
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    
    # 為了測試，先清空再寫入
    reset_collection("papers")
    count = embed_and_store(chunks)
    
    # 驗證：抓回第一筆看看
    collection = get_or_create_collection("papers")
    sample = collection.get(ids=["chunk_0"], include=["documents", "metadatas", "embeddings"])
    print(f"\n--- 驗證 chunk_0 ---")
    print(f"頁數: {sample['metadatas'][0]['page']}")
    print(f"文字: {sample['documents'][0][:150]}...")
    print(f"向量維度: {len(sample['embeddings'][0])}")
    print(f"向量前 5 維: {sample['embeddings'][0][:5]}")