from typing import List, Dict, Tuple
import numpy as np

def embed_chunks(chunks: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 48, normalize: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(model_name)
    embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i:i+batch_size]
        vec = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(vec)

    X = np.vstack(embs).astype("float32")
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    return X, chunks

if __name__ == "__main__":
    print("[Embedder] ✅ Ready")
