# Paper RAG Assistant

一個基於 RAG（Retrieval-Augmented Generation）架構的學術論文問答系統。透過語意向量檢索定位論文中的相關段落，並結合大型語言模型生成有頁碼引用的精確回答。

## 架構

```
PDF 上傳
  → 解析頁面文字（pypdf）
  → 遞迴切分 Chunks（chunk_size=800, overlap=100）
  → 向量化（multilingual-e5-small，passage: 前綴）
  → 存入 ChromaDB（cosine similarity）
  → 使用者提問 → 向量檢索 Top-K Chunks（query: 前綴）
  → 組裝 Prompt → LLM 生成回答
  → 顯示回答 + 來源頁碼
```

### 技術棧

| 元件      | 技術                                                       |
| --------- | ---------------------------------------------------------- |
| UI        | Streamlit                                                  |
| Embedding | `sentence-transformers` (`intfloat/multilingual-e5-small`) |
| Vector DB | ChromaDB（本地持久化）                                     |
| LLM       | Google Gemini / Gemma（可切換，預設 gemma-4-26b）          |
| PDF 解析  | pypdf                                                      |
| 容器化    | Docker + Docker Compose                                    |

## 快速開始

### 1. 設定 API Key

建立 `.env` 檔案：

```
GEMINI_API_KEY=your_api_key_here
```

> 至 [Google AI Studio](https://aistudio.google.com/) 申請免費 API Key。

### 2. 用 Docker 啟動（推薦）

```bash
docker compose up --build
```

開啟瀏覽器 → `http://localhost:8501`

### 3. 本地啟動（不用 Docker）

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **注意**：本地執行時，`embedder.py` 與 `build_index.py` 中的路徑（`/app/data`、`/app/chroma_db`）需改為本地路徑。

## 使用方式

1. 在左側 Sidebar 上傳 PDF 論文
2. 等待系統完成索引建立（解析 → 切分 → 向量化）
3. 在對話框輸入問題
4. 點開 **Sources** 折疊區塊可查看引用的原文段落與頁碼

**Retrieval Settings**：

- **模型**：可切換 gemma-4-26b（預設，開源）、gemini-2.5-flash、gemini-3-flash-preview、gemini-3.1-flash-lite-preview。
- **Top K**：可調整 Top K（預設 5），數值越高參考段落越多，但 Prompt 也越長。

## 延伸閱讀

各模組設計細節、工具選型理由與常見問題，請見 [introduction.md](introduction.md)。
