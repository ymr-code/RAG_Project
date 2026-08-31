"""
answer.py

Full RAG pipeline: retrieves the most relevant leaflet chunks for a
user question and passes them to Phi-3.5 Mini as grounding context.
The model is instructed to answer only from the given context.
"""

import sqlite3
import json
import math
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "arveles.db"
TOP_K = 4
CHAT_MODEL_ALIAS = "phi-3.5-mini" 
RELEVANCE_THRESHOLD = 0.50

SYSTEM_PROMPT = """You are a medical information assistant that answers questions ONLY based on the ARVELES 25 mg package leaflet excerpts provided to you as context.

Rules you MUST follow:
1. Only use the information given in the context below. Do not use any outside knowledge.
2. If the context does not contain enough information to answer the question, say clearly: "This information is not available in the provided leaflet excerpts." Do not guess or make up an answer.
3. Before answering, check whether the provided context excerpts actually relate to the question. If they discuss unrelated topics, say so honestly instead of forcing an answer.
4. Do not give medical advice beyond what is written in the leaflet. Always remind the user to consult a doctor or pharmacist for personal medical decisions.
5. Keep answers concise and clear.
6. When citing a section, use ONLY the exact section names given in the context. Never invent a section number.
"""


def load_clients():
    """Load and return (embedding_client, chat_client)."""
    config = Configuration(app_name="arveles_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    chat_model.download(lambda p: print(f"\rDownloading chat model: {p:.2f}%", end="", flush=True))
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    return embed_client, chat_client


def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def load_all_chunks(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, section, subsection, frequency, text, embedding FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "section": r[1],
            "subsection": r[2],
            "frequency": r[3],
            "text": r[4],
            "embedding": json.loads(r[5]),
        }
        for r in rows
    ]


def expand_query_for_search(query):
    """
    This knowledge base covers a single drug (ARVELES). Users may ask
    generic questions like "what is this drug" without naming it. Since
    the embedding model relies partly on lexical overlap, such queries
    score poorly against chunks that mention "ARVELES" by name. If the
    brand name is missing, prepend it before embedding (search only -
    the original query is still sent to the LLM unchanged).
    """
    if "arveles" not in query.lower():
        return f"ARVELES {query}"
    return query


def get_top_chunks(query, embed_client, all_chunks, top_k=TOP_K):
    search_query = expand_query_for_search(query)
    response = embed_client.generate_embedding(search_query)
    query_vector = response.data[0].embedding

    scored_chunks = [(cosine_similarity(query_vector, c["embedding"]), c) for c in all_chunks]
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]


def build_context_text(top_chunks):
    parts = []
    for score, chunk in top_chunks:
        label = chunk["subsection"] or chunk["section"]
        parts.append(f"[Section: {label}]\n{chunk['text']}")
    return "\n\n".join(parts)


def answer_query(query, embed_client, chat_client, all_chunks):
    top_chunks = get_top_chunks(query, embed_client, all_chunks)

    best_score = top_chunks[0][0] if top_chunks else 0.0
    if best_score < RELEVANCE_THRESHOLD:
        fallback = (
            "Your question seems too broad or may not be directly covered "
            "by this leaflet's content. Could you ask about a specific topic — "
            "for example: usage/dosage, precautions, side effects, pregnancy, "
            "or storage?"
        )
        return fallback, top_chunks

    context_text = build_context_text(top_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context from the leaflet:\n\n{context_text}\n\nQuestion: {query}"},
    ]

    response = chat_client.complete_chat(messages)
    return response.choices[0].message.content, top_chunks


def main():
    print("=== ARVELES RAG - Q&A ===\n")

    embed_client, chat_client = load_clients()
    all_chunks = load_all_chunks(DB_PATH)
    print(f"{len(all_chunks)} chunks loaded.\n")

    while True:
        query = input("Enter a question (or 'q' to quit): ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        print("\nThinking...\n")
        answer, top_chunks = answer_query(query, embed_client, chat_client, all_chunks)

        print("--- ANSWER ---")
        print(answer)

        print("\n--- Sources used ---")
        for score, chunk in top_chunks:
            label = chunk["subsection"] or chunk["section"]
            print(f"  - {label} (score: {score:.3f})")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
