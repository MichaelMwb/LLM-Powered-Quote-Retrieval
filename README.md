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

You can build a new index and try it with the **sample quotes file** or download the **prebuilt FAISS index and the Quote Metadata Database (~500k quotes)**:  
👉 [Download Prebuilt Project](https://your-google-drive-link.com)  

### Build a New Index - Windows Usage
Run: python make_index.py (May take a long time depending on GPU)

Then Run: python find_quote.py (For testing)

## 📊 Example Run  

    PS C:\Users\micha\OneDrive\CS - 3600\Homework10\LLM> python find_quote.py
    Loading model...
    Loading FAISS index...
    Using key-value store of type dbm.dumb
    Paraphrase your quote: People love to grind

    22359: 'You can either complain about it or grind, I choose to grind.'
            Kyle Vidrine, Wake Up The Winner In You: Your Time Is Now

    360281: 'I got that money on my mind but I ain't blind. I see that if I want it, I have to grind.'
            Jonathan Anthony Burkett

    325464: 'You could never understand why I grind like I doMakiyah & Jalani why I grind like I do'
            Nicki Minaj
    Time taken: 1.466 seconds


    Paraphrase your quote: Coding is the future

    148380: 'Computer Coding is a life skill for this generation.'
            Tamara Zentic MS

    291906: 'If you control the code, you control the world. This is the future that awaits us.'
            Marc Goodman, Future Crimes: Everything Is Connected, Everyone Is Vulnerable, and What We Can Do About It

    199487: 'Coding is other type of magic!'
            Deyth Banger
    Time taken: 0.283 seconds
