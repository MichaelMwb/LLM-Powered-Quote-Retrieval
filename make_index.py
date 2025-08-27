from sentence_transformers import SentenceTransformer
from csv import reader
import dbm
import faiss
import pickle
import os
import numpy as np
from time import perf_counter

DB_PATH = "quotes.db"
INDEX_PATH = "quotes.index"

# Remove index file if necessary
if os.path.isfile(INDEX_PATH):
    print(f"Removing {INDEX_PATH}")
    os.remove(INDEX_PATH)

start_time = perf_counter()

print("Loading model...")
# Load the model

model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0", device="cuda")

after_loading_model_time = perf_counter()

# Read the CSV file of quotes
print("Reading CSV...")
sentences = []
attributes = []
with open("quotes.csv", "r", encoding="utf-8") as file:
    quote_reader = reader(file)
    for row in quote_reader:
        if len(row) < 2:
            continue
        sentences.append(row[0])
        attributes.append(row[1])

after_csv_time = perf_counter()

print(f"Read {len(sentences)} quotes. Embedding...")

# Get a numpy array of vectors
embeddings = model.encode(sentences).astype(np.float32)

after_embedding_time = perf_counter()

# How long are the vectors?
dimension = embeddings.shape[1]

print("Storing embeddings in FAISS index...")

# Make an index for vectors using inner products
index = faiss.IndexFlatIP(dimension)

# Put the vectors into the index
## 
index.add(embeddings)
# Save the index (with the vectors) to INDEX_PATH
## 
faiss.write_index(index, INDEX_PATH)

after_index_time = perf_counter()

print("Storing text/attributes to key-value store...")

# Open a key-value store
## 
db= dbm.open(DB_PATH, "c")
# For each quote
for i in range(len(sentences)):
    # Convert the index to bytes
    key_bytes = str(i).encode("utf-8")## 

    # Convert the quote and its info to bytes using pickle
    value_bytes = pickle.dumps((sentences[i], attributes[i]))  

    # Store them in the key-value store
    db[key_bytes] = value_bytes

# Close the key-value store
## 
db.close()
after_kv_time = perf_counter()

print(f"Wrote {len(sentences)} quotes to {DB_PATH }and {INDEX_PATH}")

print(f"Total time: {after_kv_time - start_time:.2f} seconds")
print(f"\tLoading model: {after_loading_model_time - start_time:.2f} seconds")
print(f"\tReading CSV: {after_csv_time - after_loading_model_time:.2f} seconds")
print(f"\tLLM Embedding: {after_embedding_time - after_csv_time:.2f} seconds")
print(f"\tStoring embeddings in FAISS index: {after_index_time - after_embedding_time:.2f} seconds")
print(f"\tStoring text/attributes to key-value store: {after_kv_time - after_index_time:.2f} seconds")
