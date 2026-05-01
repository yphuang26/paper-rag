from dataclasses import dataclass
from src.embedder import get_model, get_or_create_collection


@dataclass
class RetrievedChunk:
    text: str
    page: int
    chunk_id: int
    score: float  # 相似度分數，越高越相關


def retrieve(
    query: str,
    top_k: int = 5,
    collection_name: str = "papers",
) -> list[RetrievedChunk]:
    """
    對 query 做檢索，回傳 top_k 個最相關的 chunks。
    """
    model = get_model()
    collection = get_or_create_collection(collection_name)

    # 1. 把 query 編碼成向量
    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    # 2. 在 ChromaDB 查最近的 top_k 個
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # 3. ChromaDB 回傳的 distance 是 cosine distance (1 - similarity)
    #    換算成 similarity 比較直覺
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist
        chunks.append(RetrievedChunk(
            text=doc,
            page=meta["page"],
            chunk_id=meta["chunk_id"],
            score=similarity,
        ))

    return chunks


if __name__ == "__main__":
    # 用幾個有代表性的問題測試
    queries = [
        "What is multi-head attention?",
        "How does positional encoding work?",
        "What is the computational complexity of self-attention?",
        "Who are the authors of this paper?",
        "What's the weather today?",  # 故意問無關的，看分數會不會掉
    ]

    for query in queries:
        print(f"\n{'='*70}")
        print(f"❓ Query: {query}")
        print(f"{'='*70}")

        results = retrieve(query, top_k=3)

        for i, r in enumerate(results, 1):
            print(f"\n[Top {i}] 第 {r.page} 頁 | similarity={r.score:.3f}")
            print(f"   {r.text[:200]}...")