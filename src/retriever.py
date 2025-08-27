# retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class HybridRetriever:
    def __init__(
        self,
        faiss_path="data",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        k_search=20,             # get more candidates initially
        top_n=8,                 # final chunks to return
        tier_weights=None,
        keyword_boost=0.10,
        verbose=True,
    ):
        self.verbose = verbose
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        if self.verbose:
            print(f"[retriever] Loading FAISS index from {faiss_path}")

        self.vectorstore = FAISS.load_local(
            faiss_path,
            self.embeddings,
            index_name="index",
            allow_dangerous_deserialization=True,
        )

        if self.verbose:
            try:
                count = len(self.vectorstore.index_to_docstore_id)
            except Exception:
                count = "?"
            print(f"[retriever] Loaded {count} documents")

        self.k_search = k_search
        self.top_n = top_n
        self.keyword_boost = keyword_boost
        self.tier_weights = tier_weights or {"tier1": 0.05, "tier2": 0.02, "tier3": 0.00}

    def _apply_tier_boost(self, metadata):
        return self.tier_weights.get(metadata.get("tier", ""), 0.0)

    def _apply_keyword_boost(self, text, query):
        q_tokens = query.lower().split()
        t = text.lower()
        return self.keyword_boost if any(tok in t for tok in q_tokens) else 0.0

    def retrieve(self, query):
        if self.verbose:
            print(f"\n[retriever] Query: {query}")

        # Step 1: fetch candidates
        results = self.vectorstore.similarity_search_with_score(query, k=self.k_search)
        if not results:
            if self.verbose:
                print("[retriever] No results.")
            return []

        # Step 2: apply boosts
        docs = []
        for doc, raw_score in results:
            tier_boost = self._apply_tier_boost(doc.metadata)
            kw_boost = self._apply_keyword_boost(doc.page_content, query)
            boosted = float(raw_score) + tier_boost + kw_boost
            docs.append(
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "raw_score": float(raw_score),
                    "tier_boost": tier_boost,
                    "keyword_boost": kw_boost,
                    "score_boosted": boosted,
                }
            )

        # Step 3: normalize & sort globally
        max_s = max(d["score_boosted"] for d in docs)
        for d in docs:
            d["score_norm"] = d["score_boosted"] / max_s if max_s > 0 else 0.0
        docs.sort(key=lambda x: x["score_norm"], reverse=True)

        # Step 4: final selection (no tier quotas)
        top_docs = docs[: self.top_n]

        if self.verbose:
            print(f"[retriever] Selected top {len(top_docs)} docs (global ranking)\n")
            for i, d in enumerate(top_docs, 1):
                print("=" * 80)
                print(
                    f"[{i}] tier={d['metadata'].get('tier')} "
                    f"| category={d['metadata'].get('category')} "
                    f"| raw={d['raw_score']:.4f} "
                    f"| tier+={d['tier_boost']:.2f} "
                    f"| kw+={d['keyword_boost']:.2f} "
                    f"| norm={d['score_norm']:.4f}"
                )
                print("- full text -")
                print(d["text"])

        return top_docs
