import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract


def clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def is_toc_page(text: str) -> bool:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return False
    matches = sum(1 for l in lines if re.search(r"\.{10,}\s*\d+\s*$", l))
    return matches >= 5


def parse_toc(doc: fitz.Document, max_scan_pages: int = 40):
    toc_pages = []
    for i in range(1, min(max_scan_pages, len(doc))):
        t = doc[i].get_text("text")
        if is_toc_page(t):
            toc_pages.append(i)
        elif toc_pages and not is_toc_page(t):
            break

    entries = []
    level1 = None
    for pi in toc_pages:
        text = doc[pi].get_text("text")
        for raw in text.splitlines():
            line = raw.strip()
            m = re.search(r"^(.*?)\s*\.{10,}\s*(\d+)\s*$", line)
            if not m:
                continue
            title = re.sub(r"\s+", " ", m.group(1).strip())
            page = int(m.group(2))
            caps_ratio = sum(1 for c in title if c.isupper()) / max(1, sum(1 for c in title if c.isalpha()))
            if caps_ratio > 0.8 and len(title) > 3:
                level1 = title
                entries.append({"level": 1, "title": title, "page": page, "parent": None})
            else:
                entries.append({"level": 2, "title": title, "page": page, "parent": level1})

    # dedupe
    seen = set()
    uniq = []
    for e in entries:
        key = (e["level"], e["title"], e["page"], e["parent"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)

    return toc_pages, uniq


def build_toc_intervals(entries, total_pages: int):
    entries = sorted(entries, key=lambda e: (e["page"], e["level"]))
    level1s = [e for e in entries if e["level"] == 1]
    level1s.sort(key=lambda e: e["page"])

    for idx, e in enumerate(level1s):
        e["end_page"] = (level1s[idx + 1]["page"] - 1) if idx + 1 < len(level1s) else total_pages

    level2s = [e for e in entries if e["level"] == 2]
    by_parent = {}
    for e in level2s:
        by_parent.setdefault(e["parent"], []).append(e)

    for parent, lst in by_parent.items():
        lst.sort(key=lambda x: x["page"])
        pe = next((l1 for l1 in level1s if l1["title"] == parent), None)
        parent_end = pe["end_page"] if pe else total_pages
        for i, e in enumerate(lst):
            e["end_page"] = (lst[i + 1]["page"] - 1) if i + 1 < len(lst) else parent_end

    def section_for_page(p: int):
        l1 = None
        for e in level1s:
            if e["page"] <= p <= e["end_page"]:
                l1 = e
                break
        if not l1:
            return None, None

        sub = None
        for e in by_parent.get(l1["title"], []):
            if e["page"] <= p <= e["end_page"]:
                sub = e
                break
        return l1, sub

    return section_for_page


def extract_lines_with_style(page: fitz.Page):
    d = page.get_text("dict")
    lines = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            max_size = max(s.get("size", 0) for s in spans) if spans else 0
            lines.append({"text": clean_line(text), "size": max_size})
    return lines


def page_is_image_heavy(doc: fitz.Document, page_index: int) -> bool:
    page = doc[page_index]
    text = page.get_text("text") or ""
    core = re.sub(r"Clare White", "", text, flags=re.I)
    core = re.sub(r"\b\d+\b", "", core)
    core = core.strip()
    images = page.get_images(full=True)
    return (len(core) < 80 and len(images) > 0)


def extract_page_images(doc: fitz.Document, page_index: int, out_assets: Path):
    page = doc[page_index]
    images = page.get_images(full=True)
    paths = []
    for img in images:
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        fname = f"p{page_index+1:03d}_img{xref}.png"
        fpath = out_assets / fname
        pix.save(str(fpath))
        paths.append(fpath)
    return paths


def infer_tags(title: str | None, sec1: str | None, sec2: str | None):
    text = " ".join([t for t in [title, sec1, sec2] if t]).lower()
    tags = []
    if sec1:
        tags.append(sec1)

    crisis_cues = [
        "emergency", "bleed", "haemorrhage", "hemorrhage", "arrest", "anaphylaxis",
        "malignant hyperthermia", "can't", "cannot", "failed", "desat", "desats", "pph",
        "rupture", "airway emergency", "shock",
    ]
    if any(c in text for c in crisis_cues):
        tags.append("crisis")
    if "airway" in text or "trache" in text or "laryng" in text:
        tags.append("airway")
    # de-dupe preserve order
    out = []
    for t in tags:
        if t not in out:
            out.append(t)
    return out


def chunk_page_text(doc: fitz.Document, page_index: int):
    page = doc[page_index]
    lines = extract_lines_with_style(page)

    filtered = []
    for l in lines:
        if l["text"].lower() == "clare white":
            continue
        if re.fullmatch(r"\d+", l["text"]):
            continue
        filtered.append(l)

    if not filtered:
        return []

    sizes = [l["size"] for l in filtered]
    med = statistics.median(sizes)
    heading_threshold = max(med + 3, 14)

    chunks = []
    current = None
    for l in filtered:
        is_heading = l["size"] >= heading_threshold and len(l["text"]) < 140
        if is_heading:
            if current and current["body"]:
                chunks.append(current)
            current = {"heading": l["text"], "body": []}
        else:
            if current is None:
                current = {"heading": None, "body": []}
            current["body"].append(l["text"])

    if current and current["body"]:
        chunks.append(current)

    return chunks


def extract_tables(pdf: pdfplumber.PDF, page_index: int):
    # Best-effort; many "tables" in study PDFs are actually images.
    page = pdf.pages[page_index]
    try:
        tables = page.extract_tables() or []
    except Exception:
        return []

    cleaned = []
    for t in tables:
        # remove empty tables
        if not t or len(t) < 2:
            continue
        cleaned.append(t)
    return cleaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--doc-title", default=None)
    ap.add_argument("--source-id", default=None)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    doc_title = args.doc_title or pdf_path.stem
    source_id = args.source_id or re.sub(r"[^a-zA-Z0-9_]+", "_", pdf_path.stem)

    doc = fitz.open(str(pdf_path))
    toc_pages, toc_entries = parse_toc(doc)
    section_for_page = build_toc_intervals(toc_entries, total_pages=len(doc))

    # open with pdfplumber for best-effort table extraction
    pdfp = pdfplumber.open(str(pdf_path))

    chunks = []
    for page_index in range(len(doc)):
        page_num = page_index + 1

        # skip cover + TOC pages if detected
        if toc_pages and page_index in [0, *toc_pages]:
            continue

        l1, l2 = section_for_page(page_num)
        sec1 = l1["title"] if l1 else None
        sec2 = l2["title"] if l2 else None

        if page_is_image_heavy(doc, page_index):
            img_paths = extract_page_images(doc, page_index, assets_dir)

            # OCR first image (fast + usually enough)
            ocr_text = ""
            if img_paths:
                try:
                    ocr_text = pytesseract.image_to_string(Image.open(img_paths[0]))
                except Exception:
                    ocr_text = ""

            # Title from page text (excluding author/page)
            raw_lines = (doc[page_index].get_text("text") or "").splitlines()
            title = None
            for line in raw_lines:
                line = line.strip()
                if not line or line.lower() == "clare white" or re.fullmatch(r"\d+", line):
                    continue
                title = line
                break

            alt = clean_line((ocr_text or "")[:1200]) if ocr_text else (title or "Algorithm image")
            chunk_id_seed = f"{doc_title}|{args.version}|p{page_num}|img|{title or ''}"
            chunk = {
                "chunk_id": f"{source_id}_{args.version}_p{page_num:03d}_img_{hashlib.sha1(chunk_id_seed.encode()).hexdigest()[:8]}",
                "source_id": source_id,
                "doc_title": doc_title,
                "version": args.version,
                "page_start": page_num,
                "page_end": page_num,
                "section_path": [t for t in [sec1, sec2, title] if t],
                "specialty_tags": infer_tags(title, sec1, sec2),
                "periop_phase_tags": ["all_phases"],
                "content_type": "algorithm_image",
                "modality": "image",
                "title": title or sec2 or sec1 or f"Page {page_num}",
                "keywords": [],
                "text": alt,
                "image": {
                    "paths": [p.name for p in img_paths],
                    "ocr_text": (ocr_text or "")[:5000],
                },
                "citations": [{"type": "pdf_page", "page": page_num, "label": f"p{page_num}"}],
            }
            chunks.append(chunk)
            continue

        # text chunks
        segs = chunk_page_text(doc, page_index)
        for si, seg in enumerate(segs):
            heading = seg["heading"]
            body = "\n".join(seg["body"]).strip()
            text = f"{heading}\n{body}".strip() if heading else body
            if len(text) < 40:
                continue
            seed = f"{doc_title}|{args.version}|p{page_num}|txt|{si}|{text[:200]}"
            chunk = {
                "chunk_id": f"{source_id}_{args.version}_p{page_num:03d}_txt_{si:02d}_{hashlib.sha1(seed.encode()).hexdigest()[:6]}",
                "source_id": source_id,
                "doc_title": doc_title,
                "version": args.version,
                "page_start": page_num,
                "page_end": page_num,
                "section_path": [t for t in [sec1, sec2, heading] if t],
                "specialty_tags": infer_tags(heading, sec1, sec2),
                "periop_phase_tags": ["all_phases"],
                "content_type": "text",
                "modality": "text",
                "title": heading or sec2 or sec1 or f"Page {page_num}",
                "keywords": [],
                "text": text,
                "citations": [{"type": "pdf_page", "page": page_num, "label": f"p{page_num}"}],
            }
            chunks.append(chunk)

        # tables (best-effort)
        tables = extract_tables(pdfp, page_index)
        for ti, t in enumerate(tables):
            rendered = "\n".join([" | ".join([clean_line(c or "") for c in row]) for row in t])
            if len(rendered.strip()) < 40:
                continue
            seed = f"{doc_title}|{args.version}|p{page_num}|table|{ti}|{rendered[:200]}"
            chunk = {
                "chunk_id": f"{source_id}_{args.version}_p{page_num:03d}_tbl_{ti:02d}_{hashlib.sha1(seed.encode()).hexdigest()[:6]}",
                "source_id": source_id,
                "doc_title": doc_title,
                "version": args.version,
                "page_start": page_num,
                "page_end": page_num,
                "section_path": [t for t in [sec1, sec2, f'Table {ti+1}'] if t],
                "specialty_tags": infer_tags(f"table {ti+1}", sec1, sec2),
                "periop_phase_tags": ["all_phases"],
                "content_type": "table",
                "modality": "table",
                "title": f"Table {ti+1}",
                "keywords": [],
                "text": "TABLE\n" + rendered,
                "table": {"rows": t},
                "citations": [{"type": "pdf_page", "page": page_num, "label": f"p{page_num}"}],
            }
            chunks.append(chunk)

    out_chunks = out_dir / "chunks.jsonl"
    with out_chunks.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    (out_dir / "toc.json").write_text(json.dumps(toc_entries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(chunks)} chunks to: {out_chunks}")
    print(f"Assets folder: {assets_dir}")


if __name__ == "__main__":
    main()
