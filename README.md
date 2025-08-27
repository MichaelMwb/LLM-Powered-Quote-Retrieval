# Quote Retrieval – Semantic Search with LLM Embeddings  

This project implements a **semantic search system** that retrieves quotes based on meaning rather than exact keywords. By embedding quotes and user queries into a high-dimensional vector space, the system finds the closest matches using **FAISS similarity search**.  

---

## 🎯 Motivation  

Traditional keyword search fails when wording changes. For example, “Follow your dreams” and “Chase your goals” may not match directly. This project uses **large language model (LLM) embeddings** to capture semantic meaning and return relevant quotes, even if phrased differently.  

---

## 🚀 How It Works  

- **Embeddings** → Quotes are encoded into 1024-dimensional vectors using [GIST-large-Embedding-v0](https://huggingface.co/avsolatorio/GIST-large-Embedding-v0).  
- **FAISS Indexing** → Vectors are stored in a FAISS index for fast nearest-neighbor search.  
- **Metadata Store** → Quotes and authors are linked via a 'dbm + pickle' key–value store.  
- **Querying** → User input is embedded and compared against the index; the top-3 most similar quotes are returned in real time.  

---

## 🛠 Usage  

You can build a new index and try it with the **sample quotes file** or download the **prebuilt FAISS index and the Quote Metadata Database** with the link below **(~500k quotes)**:  

👉 [Download Prebuilt Project](https://drive.google.com/drive/folders/13RcP0Xfi9E1Wkp1QZ0D7zUPSaQkuId3d?usp=sharing)  

### How to Build a New Index

First Run:
```
python3 make_index.py   # May take a long time depending on GPU
```
Then Run:
```
python3 find_quote.py  # For running tests
```
## 📊 Example Run  
Loading model...

Loading FAISS index...

Using key-value store of type dbm.dumb

Paraphrase your quote: Coding is the future  

148380: 'Computer Coding is a life skill for this generation.'  
&nbsp;&nbsp;&nbsp;&nbsp;Tamara Zentic MS  

291906: 'If you control the code, you control the world. This is the future that awaits us.'  
&nbsp;&nbsp;&nbsp;&nbsp;Marc Goodman, *Future Crimes: Everything Is Connected, Everyone Is Vulnerable, and What We Can Do About It*  

199487: 'Coding is other type of magic!'  
&nbsp;&nbsp;&nbsp;&nbsp;Deyth Banger  

Time taken: 0.283 seconds  

## ⚠️ GPU / CUDA Requirement
This project requires a CUDA-enabled GPU. The code is set to run on `device="cuda"`.
- Tested with **CUDA 12.1** and `torch==2.5.1+cu121`.
- Make sure your system has a compatible GPU driver and CUDA runtime installed.  
- If you want to run on CPU instead, change `device="cuda"` to `device="cpu"` in the code.
