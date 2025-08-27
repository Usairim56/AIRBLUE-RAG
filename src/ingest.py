import json
import os

def tier1_chunks(data):
    raw = data.get("tier1", [])
    print(f"[DEBUG] Tier1 raw entries: {len(raw)}")

    chunks = []
    for qa in raw:
        q, a = qa["question"], qa["answer"]
        chunks.append({
            "text": f"Q: {q}\nA: {a}",
            "metadata": {
                "tier": "tier1",
                "category": "faqs"
            },
        })
    return chunks


def tier2_chunks(data):
    categories = data.get("tier2", {})
    print(f"[DEBUG] Tier2 categories: {len(categories)}")

    chunks = []
    for cat, val in categories.items():
        texts = []

        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, list):
                    texts.extend([str(item) for item in v])
                else:
                    texts.append(str(v))
        elif isinstance(val, list):
            texts.extend([str(item) for item in val])
        else:
            texts.append(str(val))

        combined_text = "\n".join(texts).strip()
        chunks.append({
            "text": combined_text,
            "metadata": {"tier": "tier2", "category": cat}
        })
    return chunks


def tier3_chunks(data):
    categories = data.get("tier3", {})
    print(f"[DEBUG] Tier3 categories: {len(categories)}")

    chunks = []
    for cat, val in categories.items():
        texts = []

        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, list):
                    texts.extend([str(item) for item in v])
                else:
                    texts.append(str(v))
        elif isinstance(val, list):
            texts.extend([str(item) for item in val])
        else:
            texts.append(str(val))

        combined_text = "\n".join(texts).strip()
        chunks.append({
            "text": combined_text,
            "metadata": {"tier": "tier3", "category": cat}
        })
    return chunks


def load_blocks():
    path = os.path.join("data", "airblue_chatbot.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    t1 = tier1_chunks(data)
    print(f"[Ingest] Tier1 chunks: {len(t1)}")

    t2 = tier2_chunks(data)
    print(f"[Ingest] Tier2 chunks: {len(t2)}")

    t3 = tier3_chunks(data)
    print(f"[Ingest] Tier3 chunks: {len(t3)}")

    all_chunks = t1 + t2 + t3
    print(f"[Ingest] Total chunks: {len(all_chunks)}")
    return all_chunks
