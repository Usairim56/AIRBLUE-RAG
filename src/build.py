import os
import numpy as np
import faiss

from ingest import load_blocks
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.docstore import InMemoryDocstore  # works with LC 0.2.x


def build_pipeline():
    print("===== Building Vector Store (Cosine) =====")

    # 1) Load chunks
    blocks = load_blocks()
    print(f"[Build] Loaded {len(blocks)} chunks")

    documents = [Document(page_content=b["text"], metadata=b["metadata"]) for b in blocks]
    print(f"[Build] Created {len(documents)} documents")

    # 2) Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [d.page_content for d in documents]
    vecs = embeddings.embed_documents(texts)
    vecs = np.asarray(vecs, dtype="float32")

    # 3) Normalize → cosine similarity
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"[Build] FAISS index: {index.ntotal} vectors (dim={dim})")

    # 4) Proper docstore + id map
    ids = [str(i) for i in range(len(documents))]
    docstore = InMemoryDocstore({i: doc for i, doc in zip(ids, documents)})
    id_map = {i: ids[i] for i in range(len(ids))}

    # 5) Wrap as VectorStore and save (this writes data/index.faiss + data/index.pkl)
    vs = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=id_map,
        embedding_function=embeddings,  # needed for reload
    )

    os.makedirs("data", exist_ok=True)
    vs.save_local("data", index_name="index")

    print("✅ Wrote: data/index.faiss and data/index.pkl (LangChain format only)")


if __name__ == "__main__":
    build_pipeline()
