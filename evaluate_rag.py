import os
import sys
sys.path.append(os.path.abspath("."))

from rag.retriever import load_vector_store
from transformers import pipeline
import json

print("Loading vector DB...")
db = load_vector_store()

print("Loading model...")
generator = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    device=-1
)

def ask_question(query):
    docs = db.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""You are an intelligent assistant.
Answer the question using ONLY the context.
Context:
{context}
Question:
{query}
Answer:"""
    result = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )[0]["generated_text"]
    answer = result.split("Answer:")[-1].strip()
    return answer, [doc.page_content for doc in docs]

def score_faithfulness(answer, contexts):
    context_combined = " ".join(contexts).lower()
    answer_words = answer.lower().split()
    if len(answer_words) == 0:
        return 0.0
    matched = sum(1 for w in answer_words if w in context_combined)
    return round(matched / len(answer_words), 3)

def score_relevancy(question, answer):
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    if len(q_words) == 0:
        return 0.0
    overlap = q_words.intersection(a_words)
    return round(len(overlap) / len(q_words), 3)

test_questions = [
    "What is the main topic of this document?",
    "What are the key points mentioned?",
    "What conclusions does the document make?",
]

print("\nRunning evaluation...\n")

all_faithfulness = []
all_relevancy = []

for q in test_questions:
    print(f"Q: {q}")
    answer, contexts = ask_question(q)
    print(f"A: {answer[:100]}...")

    f_score = score_faithfulness(answer, contexts)
    r_score = score_relevancy(q, answer)

    all_faithfulness.append(f_score)
    all_relevancy.append(r_score)

    print(f"Faithfulness: {f_score} | Relevancy: {r_score}\n")

avg_faithfulness = round(sum(all_faithfulness) / len(all_faithfulness), 3)
avg_relevancy = round(sum(all_relevancy) / len(all_relevancy), 3)

print("===== EVALUATION RESULTS =====")
print(f"Avg Faithfulness:     {avg_faithfulness}")
print(f"Avg Answer Relevancy: {avg_relevancy}")
print("================================")

results = {
    "model": "microsoft/phi-2",
    "embeddings": "sentence-transformers/all-mpnet-base-v2",
    "chunk_size": 500,
    "chunk_overlap": 100,
    "top_k_retrieval": 5,
    "avg_faithfulness": avg_faithfulness,
    "avg_answer_relevancy": avg_relevancy,
    "per_question": [
        {
            "question": test_questions[i],
            "faithfulness": all_faithfulness[i],
            "answer_relevancy": all_relevancy[i]
        }
        for i in range(len(test_questions))
    ]
}

with open("ragas_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to ragas_results.json")