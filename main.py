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
# STEP 7: BUILD IVF INDEX
# -----------------------------
def build_ivf_index(vectors):
    dim = vectors.shape[1]
    nlist = 100  # number of clusters

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    # IMPORTANT: training required
    index.train(vectors)

    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)

    print("IVF index built.")
    

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
    index.nprobe = 10  # For IVF index, set number of clusters to search
    D, I = index.search(query, k)
    end = time.time()

    print(f"Search time: {(end - start)*1000:.2f} ms")
    return D, I

# -----------------------------
# MAIN FLOW
# -----------------------------
if __name__ == "__main__":

    # ⚠️ RUN ONLY FIRST TIME
    
    #generateVectors(num_vectors=500000)  # Generate 500,000 vectors of size 1,000

    # ⚠️ RUN ONLY FIRST TIME
    # vectors = load_vectors()
    # build_ivf_index(vectors)

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


#----------------------- tested with 100k vectors -----------------------
# Generated 100000 vectors of size 1000 and saved to data/vectors.npy
# [✔] Loaded vectors: (100000, 1000)
# [✔] Index built with 100000 vectors
# [✔] Loaded vectors: (100000, 1000)
# [✔] Index loaded with 100000 vectors
# Search time: 20.25 ms

# 🔍 Query Results
# Indices: [[    0 92901 13804  2327 91453]]
# Distances: [[  0.      138.39545 139.37825 139.8269  140.25293]]
# Total vectors: 100000
# Vector dimension: 1000


#----------------------- tested with 500k vectors -----------------------
# Generated 500000 vectors of size 1000 and saved to data/vectors.npy
# [✔] Loaded vectors: (500000, 1000)
# [✔] Index built with 500000 vectors
# [✔] Loaded vectors: (500000, 1000)
# [✔] Index loaded with 500000 vectors
# Search time: 88.84 ms

# 🔍 Query Results
# Indices: [[     0 411798  16594  18188  28242]]
# Distances: [[  0.      144.12036 145.71042 145.82642 145.87706]]
# Total vectors: 500000
# Vector dimension: 1000

#----------------------- tested with 500k vectors IVF index -----------------------
# Generated 500000 vectors of size 1000 and saved to data/vectors.npy
# [✔] Loaded vectors: (500000, 1000)
# IVF index built.
# [✔] Loaded vectors: (500000, 1000)
# [✔] Index loaded with 500000 vectors
# Search time: 4.62 ms

# 🔍 Query Results
# Indices: [[     0  72050  71871 268431 332933]]
# Distances: [[  0.      147.5133  147.95512 148.03984 148.10765]]
# Total vectors: 500000
# Vector dimension: 1000


#------ tested with 500k vectors IVF index : nprobe = 1 ----------------
# [✔] Loaded vectors: (500000, 1000)
# [✔] Index loaded with 500000 vectors
# Search time: 1.85 ms

# 🔍 Query Results
# Indices: [[     0  72050  71871 268431 332933]]
# Distances: [[  0.      147.5133  147.95512 148.03984 148.10765]]
# Total vectors: 500000
# Vector dimension: 1000


#------ tested with 500k vectors IVF index : nprobe = 5 ----------------
# [✔] Loaded vectors: (500000, 1000)
# [✔] Index loaded with 500000 vectors
# Search time: 8.72 ms

# 🔍 Query Results
# Indices: [[     0 253257 296062 136835  71864]]
# Distances: [[  0.      144.14429 145.167   146.70537 146.84361]]
# Total vectors: 500000
# Vector dimension: 1000

#------ tested with 500k vectors IVF index : nprobe = 10 ----------------
# [✔] Loaded vectors: (500000, 1000)
# [✔] Index loaded with 500000 vectors
# Search time: 18.29 ms

# 🔍 Query Results
# Indices: [[     0 496092 253257  74918 287411]]
# Distances: [[  0.      141.62726 144.14429 144.35953 144.85239]]
# Total vectors: 500000
# Vector dimension: 1000



#in case of 1 probe, accutacy(147) is lower but search time(1.85 ms) is faster. With 5 probes, we get better accuracy (144)but search time increases(8.72 ms). With 10 probes, we get even better accuracy(141) but search time(18.29 ms) increases further. This illustrates the trade-off between speed and accuracy when using IVF indexes in Faiss.