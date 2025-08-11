# Areios Pagos Decisions Scraper

This mini-project downloads decision pages (HTML + optional text) from the Supreme Civil and Criminal Court of Greece (Άρειος Πάγος) website, starting from a search results page under S=1 (criteria search).

## Usage

1) Perform a search at:
   https://www.areiospagos.gr/nomologia/apofaseis.asp (section: ΑΝΑΖΗΤΗΣΗ ΑΠΟΦΑΣΕΩΝ με κριτήρια)

2) After you click Search, copy the full results URL (it will look like `apofaseis_result.asp?S=1&...`).

3) Run the scraper:

```bash
# Always quote the URL in zsh to avoid globbing of ? and &
python scrapper/areiospagos_scraper.py 'https://www.areiospagos.gr/nomologia/apofaseis_result.asp?S=1&...'
```

Options:
- `--out-dir` output directory (default: `areiospagos_decisions`)
- `--delay` delay in seconds between requests (default: 1.0)
- `--max-pages` limit results pages to crawl (default: all)

Output: one `.html` (raw page) and one `.txt` (plaintext extraction) per decision.

### If your request is a POST (form submit)
Sometimes the site expects a POST with `application/x-www-form-urlencoded` (like the browser form).

Two options:

1) Provide URL-encoded form data inline (for the first page only):

```bash
python scrapper/areiospagos_scraper.py 'https://www.areiospagos.gr/nomologia/apofaseis_result.asp?S=1' \
   --form-urlencoded 'S=1&etos=2018&arithmos=1482&tmima=%CE%96'
```

2) Or use a JSON file with key/values:

```json
{
   "S": "1",
   "etos": "2018",
   "arithmos": "1482",
   "tmima": "Ζ"
}
```

```bash
python scrapper/areiospagos_scraper.py 'https://www.areiospagos.gr/nomologia/apofaseis_result.asp?S=1' \
   --form-json scrapper/form.json
```

The scraper will POST for the first page (to set cookies/criteria) and then follow pagination via GET.

## Install (local)

The root project already has a `requirements.txt` for the embedding service. For this scraper, install additional deps:

```bash
pip install requests beautifulsoup4 lxml
```

Alternatively, create a separate virtualenv and install those three packages only.

## Notes
- Please respect robots.txt and the site's Terms of Use.
- Keep polite delays (1–2 seconds).
- The site may change structure; if pagination labels differ, adjust `find_next_page()` in `areiospagos_scraper.py`.
 - If you get “please fill at least one criterion,” add filters and either pass the full results URL or use the POST options above.
