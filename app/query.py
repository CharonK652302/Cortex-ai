import os
from dotenv import load_dotenv

from rag.retriever import load_vector_store
from transformers import pipeline

print("🚀 Starting AI system...")

# ===============================
# LOAD ENV
# ===============================
load_dotenv()

# ===============================
# LOAD VECTOR DB (ONLY ONCE)
# ===============================
print("📂 Loading vector database...")
db = load_vector_store()
print("✅ Vector DB loaded!\n")

# ===============================
# LOAD Phi-2 MODEL
# ===============================
print("🔄 Loading Phi-2 model (first time takes ~1-2 min)...")

generator = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    device=-1
)

print("✅ Phi-2 model loaded!\n")


# ===============================
# MAIN FUNCTION
# ===============================
def ask_question(query):

    # Step 1: Retrieve relevant docs
    docs = db.similarity_search(query, k=5)

    print("\n🔍 Retrieved Context:\n")
    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}:\n{doc.page_content[:300]}\n")

    # Step 2: Create context
    context = "\n".join([doc.page_content for doc in docs])

    # Step 3: Create improved prompt
    prompt = f"""
You are an intelligent assistant.

Carefully read the context and answer accurately.

If the answer is not clearly present, say "I don't know".

Give a clear and concise answer.

Context:
{context}

Question:
{query}

Answer:
"""

    # Step 4: Generate response
    result = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.2
    )[0]["generated_text"]

    # Step 5: Extract only answer part
    answer = result.split("Answer:")[-1].strip()

    return answer


# ===============================
# CLI LOOP
# ===============================
if __name__ == "__main__":
    while True:
        user_input = input("💬 Ask a question (or type 'exit'): ")

        if user_input.lower() == "exit":
            print("👋 Exiting...")
            break

        answer = ask_question(user_input)

        print("\n🤖 AI Answer:\n")
        print(answer)
        print("\n" + "="*50 + "\n")