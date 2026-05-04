import os
import streamlit as st
from src.pdf_loader import load_pdf
from src.chunker import chunk_pages
from src.embedder import embed_and_store, reset_collection, get_or_create_collection
from src.generator import generate_answer_stream

# ─────────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Paper RAG Assistant")
st.caption("Upload an academic paper and ask questions about its content.")

# ─────────────────────────────────────────────────
# Session state 初始化
# ─────────────────────────────────────────────────
if "indexed" not in st.session_state:
    # 啟動時檢查 ChromaDB 有沒有資料（從上次 session 留下的）
    try:
        collection = get_or_create_collection("papers")
        st.session_state.indexed = collection.count() > 0
        st.session_state.chunk_count = collection.count()
    except Exception:
        st.session_state.indexed = False
        st.session_state.chunk_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# ─────────────────────────────────────────────────
# Sidebar: PDF 上傳 + 設定
# ─────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload an academic paper to build the knowledge base.",
    )

    if uploaded_file is not None:
        # 檔名變了就重建索引
        if st.session_state.current_pdf != uploaded_file.name:
            # 把上傳的檔案存到 /app/data
            pdf_path = f"/app/data/{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.status("Building index...", expanded=True) as status:
                st.write("📄 Parsing PDF...")
                pages = load_pdf(pdf_path)
                st.write(f"   Found {len(pages)} pages")

                st.write("✂️ Chunking text...")
                chunks = chunk_pages(pages, chunk_size=500, overlap=50)
                st.write(f"   Created {len(chunks)} chunks")

                st.write("🔢 Generating embeddings...")
                reset_collection("papers")
                count = embed_and_store(chunks)
                st.write(f"   Stored {count} vectors")

                status.update(label="Index ready!", state="complete")

            st.session_state.indexed = True
            st.session_state.chunk_count = count
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.messages = []  # 換論文清空對話
            st.rerun()

    st.divider()

    if st.session_state.indexed:
        st.success(f"✅ Index ready ({st.session_state.chunk_count} chunks)")
        if st.session_state.current_pdf:
            st.info(f"📄 {st.session_state.current_pdf}")
    else:
        st.warning("No document indexed yet. Upload a PDF to start.")

    st.divider()

    st.subheader("⚙️ Retrieval Settings")

    GEMINI_MODELS = {
        "gemma-4-26b (預設)": "gemma-4-26b-a4b-it",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-pro (高品質)": "gemini-2.5-pro",
        "gemini-3-flash-preview (最新)": "gemini-3-flash-preview",
        "gemini-3.1-flash-lite (最省配額)": "gemini-3.1-flash-lite-preview",
    }

    selected_model_label = st.selectbox(
        "模型",
        options=list(GEMINI_MODELS.keys()),
        help="gemma-4-26b 為開源模型；gemini-2.5-flash 免費方案每天限 20 次，遇到配額超限可切換其他模型。",
    )
    selected_model = GEMINI_MODELS[selected_model_label]

    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────────
# 主畫面: 對話介面
# ─────────────────────────────────────────────────
if not st.session_state.indexed:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

# 顯示歷史訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] Page {src['page']}** — similarity: `{src['score']:.3f}`"
                    )
                    st.markdown(f"> {src['text']}")
                    st.divider()

# 使用者輸入
if prompt := st.chat_input("Ask a question about the paper..."):
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        try:
            stream, source_chunks = generate_answer_stream(prompt, top_k=top_k, model=selected_model)
            answer_text = st.write_stream(stream)

            sources = [
                {"page": s.page, "score": s.score, "text": s.text}
                for s in source_chunks
            ]

            with st.expander(f"📚 Sources ({len(sources)} chunks)"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"**[{i}] Page {src['page']}** — similarity: `{src['score']:.3f}`"
                    )
                    st.markdown(f"> {src['text']}")
                    st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
            })

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })