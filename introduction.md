# 技術說明

> 整體架構與快速開始請見 [README.md](README.md)。本文件說明各模組的設計細節與技術決策。

---

## 專案結構

```
paper-rag/
├── app.py                  # Streamlit UI 主程式
├── src/
│   ├── pdf_loader.py       # PDF 解析
│   ├── chunker.py          # 遞迴文字切分 + overlap
│   ├── embedder.py         # Embedding 生成 + ChromaDB 讀寫
│   ├── retriever.py        # 向量相似度檢索
│   ├── generator.py        # RAG 流程 + Gemini 回答生成
│   └── build_index.py      # CLI 建索引工具
├── data/                   # 放置 PDF 檔案
├── chroma_db/              # ChromaDB 持久化資料（自動產生）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 各模組說明

### `pdf_loader.py`

使用 pypdf 逐頁擷取文字，保留頁碼資訊供後續回答引用。

### `chunker.py`

Embedding 模型有 token 上限（MiniLM 有效範圍約 256 tokens）。Chunk 過大時，單一向量必須壓縮多個主題的語義，導致向量在空間中趨向「均值」，與任何單一主題的距離都被拉遠，retrieval 精準度下降。Chunk 過小則切斷語義單位，缺乏上下文的片段讓 LLM 資訊不足，難以生成有效回答。

| 情況       | 問題                     |
| ---------- | ------------------------ |
| Chunk 太大 | 語義模糊、retrieval 不準 |
| Chunk 太小 | 缺上下文、LLM 答不出來   |
| 切錯位置   | 句子被腰斬               |

**Recursive 切分策略**

依優先序嘗試段落符（`\n\n`）、換行符（`\n`）、句點（`. `）、空格（` `），最後才執行硬切。每一級分隔符對應一個語義層級，優先在語義完整的邊界分割，可最大化每個 chunk 的語義內聚性（semantic cohesion），讓切出來的片段盡量是一個完整的論述單位，而非人為截斷的文字碎片。

**Overlap**

Chunk A 的結尾與 Chunk B 的開頭重疊一段文字，避免重要資訊剛好卡在切割點被截斷。本專案設定 `chunk_size=500`、`overlap=50`，即每個 chunk 500 字元、與前一個 chunk 重疊 50 字元（10%）。

### `embedder.py`

將切好的 chunks 逐一餵入 `all-MiniLM-L6-v2`，每個 chunk 轉成 384 維向量後存入 ChromaDB。「相似」的定義是 cosine similarity——兩個向量夾角越小，語義越接近。

### `retriever.py`

把使用者的問題變成 384 維向量，去 ChromaDB 找最像的 top-k 個 chunks 回傳。ChromaDB 回傳 cosine distance，換算為 similarity score（`1 - distance`）。

**對稱檢索（Symmetric Retrieval）**

問題和文件用同一個 embedding 模型編碼。這聽起來理所當然，但其實是個設計選擇——進階做法會用「非對稱」，即 query encoder 和 passage encoder 分開訓練（例如 DPR），問答效果更好但複雜度翻倍。對稱檢索是 production 最常見的起點。

### `generator.py`

把使用者問題與 top-k chunks 組成 prompt，送給 LLM 生成有引用來源的答案。

**Prompt 工程三原則**

RAG 的生成品質高度依賴 prompt 設計，三個核心原則：

1. **限制知識來源（Grounding）**：明確要求 LLM 僅能根據提供的 context 作答，防止模型混入訓練資料中的先驗知識，確保回答的可追溯性。
2. **強制引用（Citation）**：要求標註來源頁碼，讓回答可被驗證，同時促使模型對每句陳述形成自我約束。
3. **優雅失敗（Graceful Fallback）**：當 context 不足時，明確指示回答「資料中未提及」而非推斷或編造——這是在 prompt 層面防止幻覺（hallucination）最直接的手段。

---

## 工具選型

### Embedding 模型：`all-MiniLM-L6-v2`

- 384 維，模型小（約 90MB）、推論快
- 本地執行，不需要 API 費用，文件不外傳
- 在 MTEB benchmark 短文本檢索任務上表現足夠
- 缺點：以英文為主，中文效果有限

### 向量資料庫：ChromaDB

- 內嵌（embedded）模式，不需要額外架 server
- 自動持久化到本地檔案（`chroma_db/`）
- API 簡潔，適合快速原型開發

---

## Q&A

**Q：chunk size 怎麼決定的？**

取決於 embedding 模型的有效 context 範圍。MiniLM-L6 在 256 tokens 內語義最準，超過後效果遞減。500 字元（中英混合約對應 100–150 tokens）在上限內留有 buffer。Overlap 設 chunk 的 10%（50 字元）是業界常見起點，實際可透過 retrieval 評測再調整。

**Q：為什麼不用語義切分？**

語義切分需要先跑 embedding 找邊界，建索引時間翻倍。對結構化的學術論文，邊際效益不大。若是會議記錄、客服對話等主題跳躍明顯的內容，才值得引入。

**Q：為什麼不用 OpenAI embedding？**

成本、隱私、可離線。MiniLM 對短文本檢索已經足夠，且不需付費或將文件傳給第三方。中文場景則考慮 `BAAI/bge-base-zh` 或 `text-embedding-3-small`。

**Q：為什麼不用 Pinecone / Weaviate / pgvector？**

資料量小（< 1000 chunks），ChromaDB 內嵌模式最省事。Production 規模才需要 Pinecone（managed）或 pgvector（已有 Postgres 時）。

**Q：top_k = 5 怎麼決定的？**

trade-off：太少會漏掉相關上下文；太多會讓雜訊增加、context 變長、成本上升，答案品質反而可能下降（dilution effect）。5 是常見起點，正式部署會用評測集（手動標註的 query-answer 配對）跑 recall@k 找甜蜜點。

**Q：Cosine similarity vs Euclidean distance vs Dot product 怎麼選？**

sentence-transformers 的模型已做過 L2 normalization，三種數學上等價。Cosine 範圍 [-1, 1] 最直覺所以採用。若模型沒有 normalize，dot product 會被向量長度影響，cosine 比較穩。
