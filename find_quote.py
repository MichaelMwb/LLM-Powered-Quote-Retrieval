from sentence_transformers import SentenceTransformer
import dbm
import faiss
import pickle
import numpy as np
from time import perf_counter

DB_PATH = "quotes.db"
INDEX_PATH = "quotes.index"

# How many quotes should we display?
BEST_K = 3

print("Loading model...")
model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

# Open the key-value store
print(f"Using key-value store of type {dbm.whichdb(DB_PATH)}")
db = dbm.open(DB_PATH, "r")

results_file = open("recent_found_indices.txt", "a")

# Loop so you can do multiple queries
while True:

    # Get input
    sample_quote = input("Paraphrase your quote: ")

    # No input?
    if sample_quote.strip() == "":
        # Exit
        break

    start_time = perf_counter()

    # Convert the sentence to a vector
    embedding = model.encode(sample_quote, convert_to_tensor=True).cpu().numpy().astype(np.float32).reshape(1, -1)



    # find the BEST_K closest vectors
    _, I = index.search(embedding, BEST_K)

    end_time = perf_counter()

    # Step throught the results
    for i in range(BEST_K):
        # Get the index out of the numpy array and into an int
        idx =int(I[0][i])## 
        print(f"{idx}", file=results_file)

        ## 
        sentence = pickle.loads(db[str(idx).encode("utf-8")])[0]
        attributes = pickle.loads(db[str(idx).encode("utf-8")])[1]


        # Display for the user
        print(f"\n{idx}: '{sentence}'\n\t{attributes}")
    print(f"Time taken: {end_time - start_time:.3f} seconds")
    print("\n")

results_file.close()

# Close the key-value store
## 
db.close()
