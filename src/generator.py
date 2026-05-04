import os
from typing import Generator
from google import genai
from google.genai import types
from dataclasses import dataclass
from src.retriever import retrieve, RetrievedChunk

# 全域 client
_client = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 環境變數沒設定")
        _client = genai.Client(api_key=api_key)
    return _client


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk]
    query: str


SYSTEM_PROMPT = """You are a helpful research assistant answering questions about an academic paper.

Rules:
1. Answer ONLY based on the provided context passages below.
2. If the context doesn't contain enough information, say "I cannot find this information in the provided context." Do NOT make up information.
3. When referencing information, cite the page number in square brackets like [p.5].
4. Be concise but precise. Quote technical terms exactly as they appear in the paper.
5. If the question is in Chinese, answer in Chinese. If in English, answer in English.
6. Do not start with phrases like "Based on the context" or "According to the passages". Answer directly.
"""


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Passage {i} | Page {chunk.page} | Relevance: {chunk.score:.2f}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""Context passages from the paper:

{context}

---

Question: {query}

Answer:"""


def generate_answer(
    query: str,
    top_k: int = 5,
    model: str = "gemini-2.5-flash",
    collection_name: str = "papers",
) -> RAGAnswer:
    """完整 RAG 流程：retrieve → build prompt → generate"""
    
    chunks = retrieve(query, top_k=top_k, collection_name=collection_name)
    
    if not chunks:
        return RAGAnswer(
            answer="No documents found. Please build the index first.",
            sources=[],
            query=query,
        )

    user_prompt = build_user_prompt(query, chunks)

    client = get_client()
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.3,
        ),
    )

    answer_text = response.text

    return RAGAnswer(
        answer=answer_text,
        sources=chunks,
        query=query,
    )


def generate_answer_stream(
    query: str,
    top_k: int = 5,
    model: str = "gemma-4-26b-a4b-it",
    collection_name: str = "papers",
) -> tuple[Generator[str, None, None], list[RetrievedChunk]]:
    """串流版 RAG：回傳 (text_generator, chunks)，讓呼叫端邊收邊顯示。"""
    chunks = retrieve(query, top_k=top_k, collection_name=collection_name)

    if not chunks:
        def _empty():
            yield "No documents found. Please build the index first."
        return _empty(), []

    user_prompt = build_user_prompt(query, chunks)
    client = get_client()

    response_stream = client.models.generate_content_stream(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.3,
        ),
    )

    def _stream():
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    return _stream(), chunks


def print_answer(result: RAGAnswer):
    print(f"\n{'='*70}")
    print(f"❓ {result.query}")
    print(f"{'='*70}")
    print(f"\n💡 Answer:\n{result.answer}")
    print(f"\n📚 Sources:")
    for i, src in enumerate(result.sources, 1):
        print(f"   [{i}] Page {src.page} (similarity={src.score:.3f})")
        print(f"       {src.text[:120]}...")


if __name__ == "__main__":
    test_queries = [
        "What is multi-head attention and how does it work?",
        "What is the dimensionality of the model in the base Transformer?",
        "How does positional encoding work in this paper?",
        "What dataset was used for training?",
        "Who won the World Cup in 2022?",
    ]

    for q in test_queries:
        result = generate_answer(q, top_k=5)
        print_answer(result)