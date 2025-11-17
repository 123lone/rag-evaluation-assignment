import os
import sys
from dotenv import load_dotenv
from groq import Groq
from assignment_1.retriever import Retriever

# Load environment variables
import os
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def generate_rag_answer(query):
    retriever = Retriever()
    results = retriever.retrieve(query, k=3)
    context = "\n".join(results["documents"][0])

    prompt = f"""
You are a helpful RAG assistant.
Use ONLY the context provided below to answer the question.

Context:
{context}

Question:
{query}

Keep your answer factual and grounded in the context.
"""

    chat = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # FIXED: proper Groq message access
    answer = chat.choices[0].message.content
    return answer, context
