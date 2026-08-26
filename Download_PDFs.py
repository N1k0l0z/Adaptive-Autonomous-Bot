import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

DEST_FOLDER = "/Users/nikoloz/Desktop/Adaptive-Autonomous-Bot/Database_Service/Raw_Content"
INFOBOOKS_URL = "https://infobooks.org/free-pdf-books/business/economics/"

os.makedirs(DEST_FOLDER, exist_ok=True)

print("Opening InfoBooks Economics page via Playwright...")

pdf_entries = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    page = context.new_page()
    page.goto(INFOBOOKS_URL, wait_until="networkidle")
    
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(2)
    
    html_content = page.content()
    browser.close()

soup = BeautifulSoup(html_content, "html.parser")

# Target all book containers on InfoBooks
# Fallback to finding all 'a' tags with 'Download' if structural containers vary
book_cards = soup.find_all(["article", "div"], class_=re.compile(r"book|card|item", re.I))

if not book_cards:
    book_cards = soup.find_all("div")

seen_urls = set()

for card in book_cards:
    download_btn = card.find("a", href=True, string=re.compile(r"Download", re.I))
    if not download_btn:
        # Check href attribute directly
        download_btn = card.find("a", href=re.compile(r"\.pdf|/pdf/", re.I))
        
    if download_btn:
        pdf_url = urljoin(INFOBOOKS_URL, download_btn["href"])
        if pdf_url in seen_urls:
            continue
            
        title_tag = card.find(["h2", "h3", "h4", "strong", "a"])
        title_text = title_tag.text.strip() if title_tag else ""
        
        if not title_text or title_text.lower() in ["download", "read download", "read"]:
            text_snippets = [t.strip() for t in card.stripped_strings if len(t.strip()) > 3 and t.strip().lower() not in ["download", "read", "verified pdf", "secure download"]]
            title_text = text_snippets[0] if text_snippets else f"Economics_Book_{len(pdf_entries)+1}"

        seen_urls.add(pdf_url)
        pdf_entries.append((title_text, pdf_url))

print(f"Found {len(pdf_entries)} unique economics books ready for download.\n")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": INFOBOOKS_URL
})

downloaded_count = 0

for idx, (raw_title, pdf_url) in enumerate(pdf_entries, 1):
    clean_title = re.sub(r'[^\w\s-]', '', raw_title).strip().replace(' ', '_')
    if not clean_title or clean_title == "Download":
        clean_title = f"Economics_Book_{idx}"
        
    filename = f"{clean_title}.pdf"
    file_path = os.path.join(DEST_FOLDER, filename)
    
    if os.path.exists(file_path):
        print(f"[{idx}/{len(pdf_entries)}] Skipping (Already Exists): {filename}")
        continue

    print(f"[{idx}/{len(pdf_entries)}] Downloading: {filename} ...")
    try:
        res = session.get(pdf_url, stream=True, allow_redirects=True, timeout=30)
        if res.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        
            with open(file_path, "rb") as f:
                header = f.read(5)
                if header.startswith(b"%PDF-"):
                    print(f"Saved: {filename}")
                    downloaded_count += 1
                else:
                    print(f"File failed PDF header check, deleting: {filename}")
                    os.remove(file_path)
        else:
            print(f"HTTP {res.status_code} on {pdf_url}")
    except Exception as e:
        print(f"Error: {e}")

print(f"Download complete! Total new PDFs saved: {downloaded_count}")