from typing import List, Dict, Tuple
import faiss
import pickle
import numpy as np

def build_vectorstore(embeddings: np.ndarray, chunks: List[Dict],
                      faiss_path: str = "data/index.faiss",
                      pickle_path: str = "data/index.pkl") -> Tuple[str, str]:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product (cosine because normalized)
    index.add(embeddings)
    faiss.write_index(index, faiss_path)

    meta = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
    with open(pickle_path, "wb") as f:
        pickle.dump({"metadata": meta}, f)

    return faiss_path, pickle_path

if __name__ == "__main__":
    print("[VectorStore] ✅ Ready")
