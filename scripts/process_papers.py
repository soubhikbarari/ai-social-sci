import os
import re
import json
import requests
import openai
import feedparser
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load .env variables if running locally
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

PAPERS_FILE = "feed.md"
OUTPUT_FILE = "citations.md"
TRACKING_DIR = ".cache"
PROCESSED_JSON = os.path.join(TRACKING_DIR, "feed-processed.json")
TOPICS_FILE = "topics.txt"


def load_processed():
    if not os.path.exists(PROCESSED_JSON):
        return set(), {}
    with open(PROCESSED_JSON, "r") as f:
        data = json.load(f)
        return set(data.keys()), data


def save_processed(block, meta, summary, processed_dict):
    os.makedirs(TRACKING_DIR, exist_ok=True)
    processed_dict[block] = {
        "title": meta["title"],
        "url": meta["url"],
        "year": meta["year"],
        "authors": meta["authors"],
        "journal": meta["journal"],
        "topics": summary.get("topics", []),
        "takeaway": summary.get("takeaway", "")
    }
    with open(PROCESSED_JSON, "w") as f:
        json.dump(processed_dict, f, indent=2)


def load_topics():
    if not os.path.exists(TOPICS_FILE):
        return []
    with open(TOPICS_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


def fetch_metadata_from_doi(doi):
    url = f"https://api.crossref.org/works/{doi}"
    r = requests.get(url)
    if not r.ok:
        raise ValueError(f"DOI {doi} not found")
    data = r.json()["message"]
    title = re.sub(r"\s+", " ", data["title"][0]).strip()
    return {
        "title": title,
        "authors": ", ".join([f"{a['given']} {a['family']}" for a in data.get("author", [])]),
        "year": data.get("published-print", data.get("created", {})).get("date-parts", [[None]])[0][0],
        "journal": data.get("container-title", [""])[0],
        "abstract": data.get("abstract", "").replace("<jats:p>", "").replace("</jats:p>", "").replace("\n", " ").strip(),
        "url": data.get("URL", f"https://doi.org/{doi}"),
        "doi": doi
    }


def search_crossref_by_title(title):
    url = "https://api.crossref.org/works"
    params = {"query": title, "rows": 1}
    r = requests.get(url, params=params)
    r.raise_for_status()
    items = r.json()["message"]["items"]
    if not items:
        raise ValueError("No match found in CrossRef")
    return items[0]


def fetch_arxiv_metadata(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    r = requests.get(url)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    if not feed.entries:
        raise ValueError("No results from arXiv")
    entry = feed.entries[0]
    title = re.sub(r"\s+", " ", entry.title).strip()
    return {
        "title": title,
        "authors": ", ".join(a.name for a in entry.authors),
        "year": entry.published[:4],
        "journal": "arXiv",
        "abstract": entry.summary.replace("\n", " ").strip(),
        "url": entry.link,
        "doi": entry.get("arxiv_doi", "")
    }


def get_json_summary_and_tags(text, allowed_topics):
    topic_list = "\n".join(allowed_topics)
    prompt = f"""
Given the following abstract, output a JSON object with the following fields:
- takeaway: a single-sentence summary of the main finding
- topics: 3–5 most relevant topics from the list below

Topics:
{topic_list}

Abstract:
{text}

Respond only with a JSON object.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        timeout=60
    )
    return json.loads(response.choices[0].message.content)


def append_to_markdown(meta, summary):
    with open(OUTPUT_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n### {meta['title']}\n")
        f.write(f"- Url: {meta['url']}\n")
        f.write(f"- Year: {meta['year']}\n")
        f.write(f"- Authors: {meta['authors']}\n")
        f.write(f"- Venue: {meta['journal']}\n")
        topics = ', '.join(summary.get('topics', [])).replace("\n", " ").strip()
        takeaway = summary.get('takeaway', '').replace("\n", " ").strip()
        f.write(f"- Topics: {topics}\n")
        f.write(f"- Takeaway: {takeaway}\n")


def extract_arxiv_id(url):
    match = re.search(r"arxiv.org/abs/([\w.]+)", url)
    return match.group(1) if match else None


def extract_doi(line):
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", line, re.I)
    return match.group(0) if match else None


def extract_title_with_gpt(line):
    prompt = f"Extract the most likely academic paper title from the following line:\n\n{line}\n\nRespond only with the title."
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=30
    )
    return response.choices[0].message.content.strip()


def extract_urls_from_block(text_block):
    return re.findall(r"https?://\S+", text_block)


def split_blocks_by_blank_lines(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    return [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]


def main():
    processed_blocks, processed_dict = load_processed()
    topics = load_topics()
    blocks = split_blocks_by_blank_lines(PAPERS_FILE)

    b = 0
    for block in blocks:
        b = b + 1
        print(f"[{b}/{len(blocks)}]")
        if block in processed_blocks:
            continue

        urls = extract_urls_from_block(block)
        if not urls:
            print(f"No URL found in block, skipping:\n{block[:100]}...")
            continue

        try:
            for url in urls:
                if url.endswith(".pdf") or any(x in url for x in ["drive.google.com", "github.com", "metr.org"]):
                    print(f"Skipping unsupported URL: {url}")
                    continue

                print(f"Processing URL: {url}")
                doi = extract_doi(url)
                if doi:
                    meta = fetch_metadata_from_doi(doi)
                elif "arxiv.org" in url:
                    arxiv_id = extract_arxiv_id(url)
                    meta = fetch_arxiv_metadata(arxiv_id)
                else:
                    print("Attempting GPT-based title extraction...")
                    title = extract_title_with_gpt(block)
                    result = search_crossref_by_title(title)
                    doi = result["DOI"]
                    meta = fetch_metadata_from_doi(doi)

                summary = get_json_summary_and_tags(meta["abstract"] or meta["title"], topics)
                append_to_markdown(meta, summary)
                save_processed(block, meta, summary, processed_dict)

        except Exception as e:
            print(f"Failed to process block:\n{block[:100]}...\nError: {e}")


if __name__ == "__main__":
    main()
