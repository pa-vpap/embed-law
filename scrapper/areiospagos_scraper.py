import os
import time
import hashlib
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup, UnicodeDammit
import argparse
from typing import Optional, List, Dict, Tuple
import tempfile
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) court-scraper/1.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,el;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

def _greek_score(text: str) -> float:
    if not text:
        return 0.0
    greek_ranges = [
        (0x0370, 0x03FF),  # Greek and Coptic
        (0x1F00, 0x1FFF),  # Greek Extended
    ]
    total_letters = 0
    greek_letters = 0
    for ch in text:
        if ch.isalpha():
            total_letters += 1
            cp = ord(ch)
            for a, b in greek_ranges:
                if a <= cp <= b:
                    greek_letters += 1
                    break
    return (greek_letters / total_letters) if total_letters else 0.0


def _decode_html_best(content: bytes) -> str:
    # First pass with UnicodeDammit
    dammit = UnicodeDammit(content, is_html=True)
    html = dammit.unicode_markup
    score = _greek_score(html or "")
    # If low Greek ratio and content is non-ascii, try Greek encodings
    if score < 0.1 and any(b & 0x80 for b in content):
        for enc in ("cp1253", "iso-8859-7"):
            try:
                alt = content.decode(enc, errors="replace")
                if _greek_score(alt) > score:
                    html = alt
                    score = _greek_score(alt)
            except Exception:
                continue
    return html or content.decode("utf-8", errors="replace")


def _decode_and_soup(resp: requests.Response) -> BeautifulSoup:
    # Robustly detect encoding and prefer outputs with more Greek characters
    html = _decode_html_best(resp.content)
    return BeautifulSoup(html, "lxml")


def build_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_soup(session: requests.Session, url: str, timeout: float = 30.0):
    resp = session.get(url, headers=HEADERS, timeout=timeout)
    return _decode_and_soup(resp), resp.url


def post_soup(
    session: requests.Session,
    url: str,
    data: dict,
    referer: str = "https://www.areiospagos.gr/nomologia/apofaseis.asp",
    timeout: float = 30.0,
):
    """Submit the search form via POST (application/x-www-form-urlencoded)."""
    # Prime cookies with referer visit (helps set ASPSESSION and GA cookies)
    try:
        session.get(referer, headers=HEADERS, timeout=timeout)
    except Exception:
        pass
    headers = {
        **HEADERS,
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://www.areiospagos.gr",
        "Referer": referer,
    }
    resp = session.post(url, headers=headers, data=data, timeout=timeout)
    return _decode_and_soup(resp), resp.url


def is_site_error_page(soup: BeautifulSoup) -> bool:
    text = soup.get_text("\n", strip=True)
    markers = [
        "Microsoft JET Database Engine",
        "apofaseis.mdb",
        "/LIBRARY.TXT",
        "error '80004005'",
    ]
    return any(m in text for m in markers)


def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_decision_links(soup: BeautifulSoup, current_page_url: str) -> List[str]:
    """Find decision detail links, resolve relative URLs, and dedupe by 'apof' or 'cd'."""
    candidates: List[str] = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if "apofaseis_display.asp" in href.lower():
            resolved = urljoin(current_page_url, href)
            p = urlparse(resolved)
            # Keep links under /nomologia/ path to avoid wrong root resolution
            if "nomologia" not in p.path or not p.path.lower().endswith("apofaseis_display.asp"):
                continue
            candidates.append(resolved)
    # Deduplicate by 'apof' or 'cd' parameter to reduce duplicates
    seen_keys = set()
    out: List[str] = []
    for url in candidates:
        q = parse_qs(urlparse(url).query)
        key = q.get("apof", q.get("cd", [url]))[0]
        if key not in seen_keys:
            seen_keys.add(key)
            out.append(url)
    return out


def find_next_page(soup: BeautifulSoup, current_page_url: str):
    # Try to identify a next-page link heuristically
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip()
        href = a.get("href") or ""
        if not href:
            continue
        if text in ("Επόμενη", "Επόμενο", "Next", ">") or "start=" in href or "page=" in href:
            return urljoin(current_page_url, href)
    return None


def derive_filename_from_url(url: str) -> str:
    q = parse_qs(urlparse(url).query)
    # Prefer the apof number if present (e.g., 1482_2018)
    if "apof" in q and q["apof"]:
        stem = q["apof"][0]
    elif "cd" in q and q["cd"]:
        stem = q["cd"][0]
    else:
        stem = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    # Sanitize filename
    safe = "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_"))
    return safe or "decision"

def ensure_utf8_meta(soup: BeautifulSoup) -> None:
    """Ensure <meta charset="utf-8"> exists and remove conflicting http-equiv meta tags."""
    head = soup.find("head")
    if head is None:
        head = soup.new_tag("head")
        if soup.html:
            soup.html.insert(0, head)
        else:
            soup.insert(0, head)
    # Remove existing http-equiv content-type metas
    for m in head.find_all("meta"):
        http_equiv = (m.get("http-equiv") or m.get("http_equiv") or "").lower()
        if http_equiv == "content-type":
            m.decompose()
    # Ensure charset meta present and set to utf-8
    has_charset = False
    for m in head.find_all("meta"):
        if m.get("charset"):
            m["charset"] = "utf-8"
            has_charset = True
            break
    if not has_charset:
        new_meta = soup.new_tag("meta")
        new_meta.attrs["charset"] = "utf-8"
        head.insert(0, new_meta)


def _docling_convert(local_html_path: str, fmt: str) -> Tuple[str, str]:
    """Return (content, extension) for the chosen Docling export format."""
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(local_html_path)
    doc = result.document
    fmt_lower = (fmt or "markdown").lower()
    if fmt_lower == "markdown":
        return doc.export_to_markdown(), "md"
    if fmt_lower == "html":
        return doc.export_to_html(), "html"
    if fmt_lower == "json":
        # Try common JSON export methods
        if hasattr(doc, "export_to_dict"):
            return json.dumps(doc.export_to_dict(), ensure_ascii=False, indent=2), "json"
        if hasattr(doc, "export_to_json"):
            return doc.export_to_json(), "json"
        # Fallback to markdown if JSON not available
        return doc.export_to_markdown(), "md"
    # Default fallback
    return doc.export_to_markdown(), "md"


def save_decision(
    session: requests.Session,
    url: str,
    out_dir: str,
    save_txt: bool = True,
    timeout: float = 30.0,
    overwrite: bool = False,
    docling: bool = False,
    docling_format: str = "markdown",
    docling_formats: Optional[List[str]] = None,
    docling_only: bool = False,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    soup, final_url = get_soup(session, url, timeout=timeout)
    if is_site_error_page(soup):
        print(f"Skipping error page for decision URL: {final_url}")
        return None, None
    fname = derive_filename_from_url(final_url)
    html_path = os.path.join(out_dir, f"{fname}.html")
    txt_path = os.path.join(out_dir, f"{fname}.txt")

    os.makedirs(out_dir, exist_ok=True)

    # If Docling is requested and we're saving only its result, skip HTML reuse check
    if not docling or not docling_only:
        # Skip existing if not overwriting
        if not overwrite and os.path.exists(html_path):
            # Primary path is the HTML when not docling-only
            return html_path, (txt_path if (save_txt and os.path.exists(txt_path)) else None), [os.path.basename(html_path)]

    # Prepare source HTML for downstream steps
    ensure_utf8_meta(soup)
    html_str = str(soup)
    wrote_final_html = False
    if not docling or not docling_only:
        # Save raw HTML if not docling-only
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        wrote_final_html = True

    # Optional Docling conversion
    saved_files: List[str] = []
    docling_out_path = None
    if docling:
        # Use a temp file to feed Docling
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_html = os.path.join(tmpdir, f"{fname}.html")
            with open(tmp_html, "w", encoding="utf-8") as tf:
                tf.write(html_str)
            fmts = (docling_formats if docling_formats else [docling_format])
            for fmt in fmts:
                content, ext = _docling_convert(tmp_html, fmt)
                out_ext = "md" if ext == "md" else ("json" if ext == "json" else "docling.html")
                out_path = os.path.join(out_dir, f"{fname}.{out_ext}")
                with open(out_path, "w", encoding="utf-8") as outf:
                    outf.write(content)
                saved_files.append(os.path.basename(out_path))
                # Prefer markdown as the primary docling path; else first produced
                if docling_out_path is None or out_ext == "md":
                    docling_out_path = out_path

    if not docling_only and save_txt:
        # Basic text extraction
        text = soup.get_text("\n", strip=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    if docling and docling_only:
        return docling_out_path, None, saved_files if saved_files else ([os.path.basename(docling_out_path)] if docling_out_path else [])
    # Include HTML in saved file list when we saved it
    if wrote_final_html:
        saved_files.insert(0, os.path.basename(html_path))
    return (html_path if wrote_final_html else docling_out_path), (txt_path if (save_txt and not docling_only) else None), saved_files


def crawl_results(
    seed_results_url: str,
    out_dir: str = "areiospagos_decisions",
    delay: float = 1.0,
    max_pages: Optional[int] = None,
    form_data: Optional[dict] = None,
    referer: Optional[str] = None,
    save_results_pages: bool = False,
    limit_per_page: Optional[int] = None,
    save_txt: bool = True,
    retries: int = 3,
    backoff: float = 0.5,
    timeout: float = 30.0,
    overwrite: bool = False,
    docling: bool = False,
    docling_format: str = "markdown",
    docling_formats: Optional[List[str]] = None,
    docling_only: bool = False,
):
    session = build_session(retries=retries, backoff=backoff)
    visited_pages = set()
    page_url = seed_results_url
    total = 0
    pages = 0
    manifest: List[Dict[str, str]] = []

    while page_url and (max_pages is None or pages < max_pages):
        if page_url in visited_pages:
            break
        visited_pages.add(page_url)

        # Use POST for the first page if form data is provided; subsequent pages via GET
        if pages == 0 and form_data is not None:
            soup, final_url = post_soup(session, page_url, form_data, referer=referer or "https://www.areiospagos.gr/nomologia/apofaseis.asp", timeout=timeout)
        else:
            soup, final_url = get_soup(session, page_url, timeout=timeout)
        # Optionally save the results page for inspection
        if save_results_pages:
            os.makedirs(out_dir, exist_ok=True)
            page_idx = pages + 1
            with open(os.path.join(out_dir, f"results_page_{page_idx}.html"), "w", encoding="utf-8") as f:
                f.write(str(soup))

        # Abort early if the results page itself is an error
        if is_site_error_page(soup):
            print("Encountered site error page on results; stopping. Try different criteria or later.")
            break

        # Collect decisions on this page
        decision_links = extract_decision_links(soup, final_url)
        print(f"Found {len(decision_links)} decision link(s) on page {pages+1}.")
        saved_this_page = 0
        for idx, durl in enumerate(decision_links, start=1):
            if limit_per_page is not None and idx > limit_per_page:
                break
            try:
                primary_path, _, saved_list = save_decision(
                    session,
                    durl,
                    out_dir,
                    save_txt=save_txt,
                    timeout=timeout,
                    overwrite=overwrite,
                    docling=docling,
                    docling_format=docling_format,
                    docling_formats=docling_formats,
                    docling_only=docling_only,
                )
                if primary_path:
                    total += 1
                    saved_this_page += 1
                    # Track in manifest
                    manifest.append({"url": durl, "files": saved_list if saved_list else ([os.path.basename(primary_path)] if primary_path else [])})
                time.sleep(delay)
            except Exception as e:
                print(f"Failed to save {durl}: {e}")
        print(f"Saved {saved_this_page}/{len(decision_links)} from page {pages+1}.")

        # Find next page
        next_url = find_next_page(soup, final_url)
        if next_url and next_url not in visited_pages and urlparse(next_url).path.endswith("apofaseis_result.asp"):
            page_url = next_url
            pages += 1
            time.sleep(delay)
        else:
            break

    # Write a manifest for bookkeeping
    try:
        if total:
            with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"count": total, "items": manifest}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Couldn't write manifest: {e}")

    print(f"Saved {total} decisions from {len(visited_pages)} results page(s).")


def parse_args():
    p = argparse.ArgumentParser(description="Download Areios Pagos court decision pages from a search results URL.")
    p.add_argument("seed_results_url", help="Full apofaseis_result.asp URL obtained after performing a search with criteria (S=1)")
    p.add_argument("--out-dir", default="areiospagos_decisions", help="Output directory for saved decisions")
    p.add_argument("--delay", type=float, default=1.0, help="Delay (seconds) between requests")
    p.add_argument("--max-pages", type=int, default=None, help="Maximum number of results pages to crawl (default: all)")
    p.add_argument("--form-urlencoded", dest="form_urlencoded", default=None, help="application/x-www-form-urlencoded for initial POST (e.g., 'S=1&etos=2018&arithmos=...')")
    p.add_argument("--form-json", dest="form_json", default=None, help="Path to JSON file with form key/value pairs for initial POST")
    p.add_argument("--referer", default="https://www.areiospagos.gr/nomologia/apofaseis.asp", help="Referer to use for POST (default: search form page)")
    p.add_argument("--save-results-pages", action="store_true", help="Save each results page HTML into the output directory for debugging")
    p.add_argument("--limit-per-page", type=int, default=None, help="Save at most N decisions per results page (for quick testing)")
    p.add_argument("--html-only", action="store_true", help="Only save HTML files (skip generating .txt extracts)")
    p.add_argument("--retries", type=int, default=3, help="HTTP retries for GET/POST requests")
    p.add_argument("--backoff", type=float, default=0.5, help="Retry backoff factor (seconds)")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout (seconds)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files instead of skipping them")
    p.add_argument("--docling", action="store_true", help="Process each decision with Docling and save the converted output")
    p.add_argument("--docling-format", default="markdown", choices=["markdown", "html", "json"], help="Docling export format (default: markdown)")
    p.add_argument("--docling-save", nargs="+", choices=["markdown", "html", "json"], help="Save multiple Docling formats at once (space-separated)")
    p.add_argument("--docling-only", action="store_true", help="Save only Docling output (do not save raw HTML or .txt)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    form_data = None
    if args.form_json:
        import json
        with open(args.form_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if not isinstance(obj, dict):
                raise SystemExit("--form-json must contain a JSON object of key/value pairs")
            form_data = {str(k): str(v) for k, v in obj.items()}
    elif args.form_urlencoded:
        from urllib.parse import parse_qsl
        form_data = {k: v for k, v in parse_qsl(args.form_urlencoded, keep_blank_values=True)}
    crawl_results(
        seed_results_url=args.seed_results_url,
        out_dir=args.out_dir,
        delay=args.delay,
        max_pages=args.max_pages,
        form_data=form_data,
        referer=args.referer,
        save_results_pages=args.save_results_pages,
        limit_per_page=args.limit_per_page,
        save_txt=(not args.html_only),
        retries=args.retries,
        backoff=args.backoff,
        timeout=args.timeout,
        overwrite=args.overwrite,
    docling=args.docling,
    docling_format=args.docling_format,
    docling_formats=args.docling_save,
    docling_only=args.docling_only,
    )
