import json

from pathlib import Path
import fitz
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


from dotenv import load_dotenv
load_dotenv()


TREE_INDEX_PATH     = "pageindex_tree.json"
DEFAULT_MODEL       = "gpt-4o-mini"
MAX_PAGES_PER_NODE  = 10
TOP_K_NODES         = 3

llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2)


def load_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page_number": i + 1, "text": text})
    doc.close()
    print(f"[Indexer] Loaded {len(pages)} pages from '{pdf_path}'")
    return pages


def _chunk_pages(pages: list[dict], max_pages: int) -> list[list[dict]]:
    return [pages[i:i + max_pages] for i in range(0, len(pages), max_pages)]


def _summarise_chunk(chunk: list[dict], node_id: str) -> dict:
    combined = "\n\n".join(
        f"[Page {p['page_number']}]\n{p['text'][:3000]}" for p in chunk
    )
    prompt = (
        "Read the following document pages and return a JSON object with:\n"
        '  "title": concise section title (< 10 words)\n'
        '  "summary": 2-4 sentence summary of key information\n\n'
        f"Pages:\n{combined}\n\n"
        "Respond ONLY with valid JSON, no markdown fences."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    try:
        info = json.loads(resp.content)
    except json.JSONDecodeError:
        info = {"title": f"Section {node_id}", "summary": ""}

    return {
        "node_id":    node_id,
        "title":      info.get("title", f"Section {node_id}"),
        "summary":    info.get("summary", ""),
        "start_page": chunk[0]["page_number"],
        "end_page":   chunk[-1]["page_number"],
        "pages":      chunk,
    }


def build_tree_index(pdf_path: str, save_path: str = TREE_INDEX_PATH) -> dict:
    pages  = load_pdf(pdf_path)
    chunks = _chunk_pages(pages, MAX_PAGES_PER_NODE)
    nodes  = []

    print(f"[Indexer] Building {len(chunks)} nodes…")
    for i, chunk in enumerate(chunks):
        nid = f"{i + 1:04d}"
        print(f"  → Node {nid} (pages {chunk[0]['page_number']}–{chunk[-1]['page_number']})")
        nodes.append(_summarise_chunk(chunk, nid))

    summaries = "\n".join(f"- [{n['node_id']}] {n['title']}: {n['summary']}" for n in nodes)
    doc_resp  = llm.invoke([HumanMessage(
        content=f"Write a 3-sentence overview of this document based on its sections:\n{summaries}\nPlain text only."
    )])

    tree = {
        "doc_description": doc_resp.content.strip(),
        "total_pages":     pages[-1]["page_number"] if pages else 0,
        "nodes":           nodes,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    print(f"[Indexer] ✅ Tree index saved → {save_path}")
    return tree



def load_tree_index(path: str = TREE_INDEX_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Tree index not found at '{path}'. Run --index first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
    


