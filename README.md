# Quote Retrieval – Semantic Search with LLM Embeddings  

This project implements a **semantic search system** that retrieves quotes based on meaning rather than exact keywords. By embedding quotes and user queries into a high-dimensional vector space, the system finds the closest matches using **FAISS similarity search**.  

---

## 🎯 Motivation  
Traditional keyword search fails when wording changes. For example, “Follow your dreams” and “Chase your goals” may not match directly. This project uses **large language model (LLM) embeddings** to capture semantic meaning and return relevant quotes, even if phrased differently.  

---

## 🚀 Features  

- **Semantic Embedding**  
  - Embedded ~500,000 quotes into 1024-dimensional vectors using the **GIST-large-Embedding-v0** model.  

- **FAISS Similarity Search**  
  - Indexed all embeddings with **FAISS (Facebook AI Similarity Search)** for efficient nearest-neighbor lookups.  

- **Key–Value Store for Metadata**  
  - Stored quotes and authors in a **dbm + pickle key–value store**, linked to FAISS indices for fast retrieval.  

- **Interactive Query Interface**  
  - Users input paraphrased quotes; system embeds query and returns the **top-3 most semantically similar quotes** in real time.  

---

## 🛠 Tech Stack  

- **Languages:** Python 3  
- **Libraries & Tools:** FAISS, SentenceTransformers, dbm, pickle, NumPy  
- **Model:** [GIST-large-Embedding-v0](https://huggingface.co/avsolatorio/GIST-large-Embedding-v0) for text embeddings  
- **Data:** ~500k quotes (CSV)  

---

## 📊 Example  

```bash
$ python3 find_quote.py  

Loading model...  
Loading index...  
Paraphrase your quote: Let your dreams guide you.  

194805: "Live your dreams."  
    Cynthia Vespia  

313200: "Your dreams will take you where you belong."  
    Debasish Mridha  

429832: "Believe in and follow your dreams."  
    Kristanna Loken  

Time taken: 0.143 seconds
