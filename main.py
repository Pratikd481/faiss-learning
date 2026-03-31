import numpy as np
import faiss
import os
import time

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "data/vectors.npy"
INDEX_PATH = "index/faiss.index"


def generateVectors(vector_size=1000, num_vectors=128):
    """
    Generates random vectors and saves them to a file.
    
    Parameters:
    vector_size (int): The size of each vector.
    num_vectors (int): The number of vectors to generate.
    """
    vectors = np.random.rand(num_vectors, vector_size).astype("float32")
    os.makedirs("data", exist_ok=True)
    np.save(DATA_PATH, vectors)
    print(f"Generated {num_vectors} vectors of size {vector_size} and saved to {DATA_PATH}")
    
    
# -----------------------------
# STEP 2: LOAD VECTORS
# -----------------------------
def load_vectors():
    vectors = np.load(DATA_PATH)
    print(f"[✔] Loaded vectors: {vectors.shape}")
    return vectors


# -----------------------------
# STEP 3: BUILD INDEX
# -----------------------------

def build_index(vectors):
    dim = vectors.shape[1]
    base_index  = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap(base_index)
    
    ids = np.arange(len(vectors))
    index.add_with_ids(vectors, ids)
    
    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"[✔] Index built with {index.ntotal} vectors")
    

# -----------------------------
# STEP 4: LOAD INDEX
# -----------------------------
def load_index():
    index = faiss.read_index(INDEX_PATH)
    print(f"[✔] Index loaded with {index.ntotal} vectors")
    return index

# -----------------------------
# STEP 5: SEARCH
# -----------------------------
def search(index, query_vector, k=5):
    distances, indices = index.search(query_vector, k)
    return distances, indices


# -----------------------------
# STEP 6: Timed Search
# -----------------------------
def timed_search(index, query, k=5):
    start = time.time()
    D, I = index.search(query, k)
    end = time.time()

    print(f"Search time: {(end - start)*1000:.2f} ms")
    return D, I

# -----------------------------
# MAIN FLOW
# -----------------------------
if __name__ == "__main__":

    # ⚠️ RUN ONLY FIRST TIME
    
    generateVectors(num_vectors=10000)  # Generate 10,000 vectors of size 1,000

    # ⚠️ RUN ONLY FIRST TIME
    vectors = load_vectors()
    build_index(vectors)

    # NORMAL FLOW (AFTER SETUP)
    vectors = load_vectors()
    index = load_index()

    # Take first vector as query
    query = vectors[0].reshape(1, -1)

    D, I = timed_search(index, query)

    print("\n🔍 Query Results")
    print("Indices:", I)
    print("Distances:", D)
    print("Total vectors:", index.ntotal)
    print("Vector dimension:", vectors.shape[1])
    
    
    
#----------------------- tested with 10k vectors -----------------------
# python main.py 
# Generated 10000 vectors of size 1000 and saved to data/vectors.npy
# [✔] Loaded vectors: (10000, 1000)
# [✔] Index built with 10000 vectors
# [✔] Loaded vectors: (10000, 1000)
# [✔] Index loaded with 10000 vectors
# [✔] Loaded vectors: (10000, 1000)
# [✔] Index built with 10000 vectors
# [✔] Loaded vectors: (10000, 1000)
# [✔] Index loaded with 10000 vectors
# [✔] Loaded vectors: (10000, 1000)
# [✔] Index loaded with 10000 vectors
# [✔] Index loaded with 10000 vectors
# Search time: 2.90 ms

# 🔍 Query Results
# Indices: [[   0 3051 5245  231 4565]]
# Distances: [[  0.      148.44266 149.53099 149.65929 150.50362]]
# Total vectors: 10000
# Vector dimension: 1000