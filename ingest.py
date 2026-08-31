"""
ingest.py

Parses the ARVELES leaflet HTML, chunks it by section (h3 blocks),
embeds each chunk with Foundry Local's embedding model, and stores
chunk text + embedding + metadata in SQLite.

Requirements:
    pip install beautifulsoup4 foundry-local-sdk openai
"""

import sqlite3
import json
from bs4 import BeautifulSoup
from foundry_local_sdk import Configuration, FoundryLocalManager

HTML_PATH = "arveles_leaflet_en.html"
DB_PATH = "arveles.db"


def load_embedding_client():
    """Load and return the Foundry Local embedding client."""
    config = Configuration(app_name="arveles_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.download(lambda p: print(f"\rDownloading model: {p:.2f}%", end="", flush=True))
    print()
    model.load()
    print("Embedding model ready.\n")

    return model.get_embedding_client()


def parse_html_into_chunks(html_path):
    """
    Parse the HTML leaflet into a list of chunks. Each <h3> block
    (heading + following <p>/<li> until the next <h3>) becomes one
    chunk. Sections without <h3> tags are stored as a single chunk.

    Returns: list of {section, subsection, frequency, text}
    """
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    chunks = []

    # Content that sits directly under <article> but outside any <section>
    # (e.g. the top-line "For oral use.") was previously dropped entirely
    # because we only ever iterated soup.find_all("section"). Capture it
    # as its own small chunk instead of silently losing it.
    article_tag = soup.find("article")
    if article_tag:
        loose_parts = [
            el.get_text(strip=True)
            for el in article_tag.find_all("p", recursive=False)
            if el.get_text(strip=True)
        ]
        if loose_parts:
            chunks.append({
                "section": "administrative",
                "subsection": "general-info",
                "frequency": None,
                "text": "General information. " + " ".join(loose_parts),
            })

    for section_tag in soup.find_all("section"):
        section_name = section_tag.get("data-section", "unknown")
        section_title_tag = section_tag.find("h2")
        section_title = section_title_tag.get_text(strip=True) if section_title_tag else section_name

        h3_tags = section_tag.find_all("h3")

        if not h3_tags:
            text_parts = [el.get_text(strip=True) for el in section_tag.find_all(["p", "li"])]
            full_text = f"{section_title}. " + " ".join(text_parts)
            chunks.append({
                "section": section_name,
                "subsection": None,
                "frequency": None,
                "text": full_text.strip(),
            })
        else:
            for h3 in h3_tags:
                subsection_title = h3.get_text(strip=True)
                top_frequency = h3.get("data-frequency")

                # Group content into "buckets": bucket 0 holds anything
                # directly under the h3 (the common case). If an h4
                # sub-heading appears (e.g. "Combinations not
                # recommended", "Uncommon side effects"), a new tagged
                # bucket opens for it, so each risk/frequency tier stays
                # its own chunk instead of merging into one blob.
                
                buckets = [{"h4_title": None, "tier": top_frequency, "parts": []}]

                for sib in h3.find_next_siblings():
                    if sib.name == "h3":
                        break
                    if sib.name == "h4":
                        tier = sib.get("data-frequency") or sib.get("data-interaction-tier")
                        buckets.append({
                            "h4_title": sib.get_text(strip=True),
                            "tier": tier,
                            "parts": [],
                        })
                    elif sib.name == "ul":
                        buckets[-1]["parts"].extend(
                            li.get_text(strip=True) for li in sib.find_all("li")
                        )
                    elif sib.name == "p":
                        buckets[-1]["parts"].append(sib.get_text(strip=True))

                for bucket in buckets:
                    if not bucket["parts"]:
                        continue
                    label = (
                        f"{subsection_title} - {bucket['h4_title']}"
                        if bucket["h4_title"]
                        else subsection_title
                    )
                    full_text = f"{section_title} - {label}. " + " ".join(bucket["parts"])
                    chunks.append({
                        "section": section_name,
                        "subsection": label,
                        "frequency": bucket["tier"],
                        "text": full_text.strip(),
                    })

    return chunks


def setup_database(db_path):
    """Create the SQLite database and chunks table if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section TEXT,
            subsection TEXT,
            frequency TEXT,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def main():
    print("=== ARVELES RAG - Ingestion ===\n")

    embed_client = load_embedding_client()

    print(f"Parsing '{HTML_PATH}' into chunks...")
    chunks = parse_html_into_chunks(HTML_PATH)
    print(f"Found {len(chunks)} chunks.\n")

    conn = setup_database(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks")  # avoid duplicates on re-run
    conn.commit()

    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}/{len(chunks)}] Embedding: {chunk['subsection'] or chunk['section']}")

        response = embed_client.generate_embedding(chunk["text"])
        embedding = response.data[0].embedding

        cursor.execute(
            """
            INSERT INTO chunks (section, subsection, frequency, text, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                chunk["section"],
                chunk["subsection"],
                chunk["frequency"],
                chunk["text"],
                json.dumps(embedding),  # stored as a JSON string
            ),
        )

    conn.commit()
    conn.close()

    print(f"\nDone. {len(chunks)} chunks saved to '{DB_PATH}'.")


if __name__ == "__main__":
    main()
