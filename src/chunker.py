from typing import List, Dict

def chunk_blocks(blocks: List[Dict]) -> List[Dict]:
    """
    No chunking — each block becomes one full chunk.
    """
    out: List[Dict] = []
    for i, b in enumerate(blocks):
        text = b.get("text", "")
        meta = dict(b.get("metadata", {}))
        meta["source_block"] = i
        out.append({"text": text, "metadata": meta})
    return out

if __name__ == "__main__":
    print("[Chunker] ✅ Ready — No splitting applied")
