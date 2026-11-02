"""Offline policy retrieval, exact assignment prompts, and grounding evaluation."""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

ROOT = Path(__file__).resolve().parents[1]


def chunk_policy_documents(policy_dir=None, chunk_size=40, overlap=10):
    """Split whitespace words, stopping at the first chunk reaching the document end."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")
    policy_dir = Path(policy_dir) if policy_dir is not None else ROOT / "data/policies"
    chunks = []
    for path in sorted(policy_dir.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        for start in range(0, len(words), chunk_size - overlap):
            end = min(start + chunk_size, len(words))
            chunks.append({"source": path.name, "text": " ".join(words[start:end]),
                           "start_word": start, "end_word": end})
            if end == len(words):
                break
    return chunks


class PolicyRetriever:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("At least one non-empty policy chunk is required.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def retrieve(self, query, k=3):
        """Return source, cosine score, and text for the k most similar chunks."""
        if k < 1:
            raise ValueError("k must be positive.")
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        indices = similarities.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(similarities[i]),
                 "text": self.chunks[i]["text"]} for i in indices]

    def build_messages(self, question, k=3):
        """Construct the two-message, retrieval-grounded prompt without altering chunks."""
        retrieved = self.retrieve(question, k=k)
        context = "\n".join(f"[{chunk['source']}]\n{chunk['text']}" for chunk in retrieved)
        messages = [
            {"role": "system", "content": "POLICY ASSISTANT. Answer only from the supplied context. "
             "If the information is missing, say I could not find that information in the policy documents."},
            {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
        ]
        return messages, [chunk["source"] for chunk in retrieved]

    def answer(self, question, k=3):
        """Return the mock's text, retrieved sources, and measured token usage."""
        messages, sources = self.build_messages(question, k=k)
        response = mock_llm.chat(messages)
        return response.choices[0].message.content, sources, int(response.usage.total_tokens)


def evaluate_policy_rag(retriever, eval_csv_path=None):
    """Evaluate document hits, case-sensitive expected facts, and fallback prefixes."""
    path = Path(eval_csv_path) if eval_csv_path is not None else ROOT / "data/raw/policy_eval.csv"
    evaluation = pd.read_csv(path).fillna("")
    details = []
    for number, row in enumerate(evaluation.to_dict("records"), start=1):
        answer, sources, tokens = retriever.answer(row["question"])
        answerable = row["expected_doc"] != "NONE"
        details.append({"question_id": f"Q{number}", **row, "sources": sources,
                        "answer": answer, "tokens_used": tokens,
                        "doc_hit": answerable and row["expected_doc"] in sources,
                        "fact_hit": answerable and row["expected_fact"] in answer,
                        "fallback_hit": not answerable and answer.startswith("I could not find")})
    return {
        "doc_hits": sum(d["doc_hit"] for d in details),
        "fact_hits": sum(d["fact_hit"] for d in details),
        "fallback_hits": sum(d["fallback_hit"] for d in details),
        "total_answerable": sum(d["expected_doc"] != "NONE" for d in details),
        "total_unanswerable": sum(d["expected_doc"] == "NONE" for d in details),
        "details": details,
    }
