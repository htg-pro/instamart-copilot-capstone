"""part_d/rag.py - Retrieval-Augmented Generation (RAG) over policy documents."""
import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

def chunk_policy_documents(policy_dir=os.path.join("data", "policies"), chunk_size=40, overlap=10):
    """
    Splits policy documents into chunks of chunk_size words with given overlap.
    For overlap=10 and chunk_size=40, step is 30 words.
    Stops after the first chunk whose end reaches or passes the document's end.
    """
    policy_files = sorted(glob.glob(os.path.join(policy_dir, "*.txt")))
    chunks = []
    step = chunk_size - overlap if overlap < chunk_size else chunk_size

    for pf in policy_files:
        fname = os.path.basename(pf)
        with open(pf, "r", encoding="utf-8") as f:
            text = f.read().strip()
        words = text.split()
        idx = 0
        while True:
            chunk_words = words[idx : idx + chunk_size]
            chunks.append({
                "source": fname,
                "text": " ".join(chunk_words)
            })
            if idx + chunk_size >= len(words):
                break
            idx += step

    return chunks

class PolicyRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def retrieve(self, query, k=3):
        """
        Transforms query and returns top-k chunks ranked by cosine similarity.
        """
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_indices = sims.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(sims[i]),
                "text": self.chunks[i]["text"]
            }
            for i in top_indices
        ]

    def answer(self, question, k=3):
        """
        Retrieves top-k chunks, constructs prompt, calls mock_llm.chat with POLICY ASSISTANT.
        """
        top_chunks = self.retrieve(question, k=k)
        context_parts = [f"[{c['source']}]\n{c['text']}" for c in top_chunks]
        context_str = "\n".join(context_parts)

        user_content = f"CONTEXT:\n{context_str}\nQUESTION:\n{question}"
        messages = [
            {
                "role": "system",
                "content": "POLICY ASSISTANT. Answer the question using strictly the supplied context. "
                           "If the context does not contain the answer, state that you could not find the information in the policy documents."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        resp = mock_llm.chat(messages)
        ans_text = resp.choices[0].message.content
        sources = [c["source"] for c in top_chunks]
        return ans_text, sources, resp.usage.total_tokens

def evaluate_policy_rag(retriever, eval_csv_path=os.path.join("data", "raw", "policy_eval.csv")):
    eval_df = pd.read_csv(eval_csv_path)

    doc_hits = 0
    fact_hits = 0
    fallback_hits = 0
    total_answerable = 0
    total_unanswerable = 0

    detailed_results = []

    for _, row in eval_df.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        ans, sources, tokens = retriever.answer(question)

        is_answerable = expected_doc != "NONE"
        doc_hit = False
        fact_hit = False
        fallback_hit = False

        if is_answerable:
            total_answerable += 1
            if expected_doc in sources:
                doc_hits += 1
                doc_hit = True
            if expected_fact in ans:
                fact_hits += 1
                fact_hit = True
        else:
            total_unanswerable += 1
            if ans.startswith("I could not find"):
                fallback_hits += 1
                fallback_hit = True

        detailed_results.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "sources": sources,
            "answer": ans,
            "doc_hit": doc_hit,
            "fact_hit": fact_hit,
            "fallback_hit": fallback_hit
        })

    return {
        "doc_hits": doc_hits,
        "total_answerable": total_answerable,
        "fact_hits": fact_hits,
        "fallback_hits": fallback_hits,
        "total_unanswerable": total_unanswerable,
        "details": detailed_results
    }
