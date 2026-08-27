import os
import re
import time
import requests
import pandas as pd
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

DEST_FOLDER = "Document_Upload_Service/Raw_Content"
CSV_PATH = os.path.join(DEST_FOLDER, "metadata_law.csv")
BASE_URL = "https://infobooks.org"
MAIN_BUSINESS_URL = "https://infobooks.org/free-pdf-books/law/"

os.makedirs(DEST_FOLDER, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": MAIN_BUSINESS_URL
})

metadata_records = []
book_counter = 1
seen_download_urls = set()

cat_match = re.search(r"/free-pdf-books/([^/]+)", MAIN_BUSINESS_URL)
category_name = cat_match.group(1) if cat_match else "book"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    page = context.new_page()

    subcategory_urls = []
    try:
        page.goto(MAIN_BUSINESS_URL, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)
        
        main_soup = BeautifulSoup(page.content(), "html.parser")
        
        for a_tag in main_soup.select("li a[href*='/free-pdf-books/law/']"):
            href = a_tag.get("href")
            if href and href.strip("/") != "free-pdf-books/law":
                full_sub_url = urljoin(BASE_URL, href)
                if full_sub_url not in subcategory_urls:
                    subcategory_urls.append(full_sub_url)
                    
        print(f"Found {len(subcategory_urls)} subcategories to process.")
        
    except Exception as e:
        print(f"Error loading main index page: {e}")
        browser.close()
        exit(1)

    for sub_url in subcategory_urls:
        print(f"\nProcessing Subcategory: {sub_url}")
        
        try:
            page.goto(sub_url, wait_until="domcontentloaded", timeout=60000)
            for _ in range(5):
                page.evaluate("window.scrollBy(0, 1000)")
                time.sleep(1)
        except Exception as e:
            print(f"Skipping {sub_url} due to error: {e}")
            continue

        cat_soup = BeautifulSoup(page.content(), "html.parser")
        cards = cat_soup.find_all("li", class_="pdf-card")

        for card in cards:
            title_node = card.find("h3", class_="pdf-card__title")
            if not title_node:
                continue
            raw_title = title_node.text.strip()

            desc_node = card.find("p", class_="pdf-card__desc")
            description = desc_node.text.strip() if desc_node else ""

            author_node = card.find("p", class_="pdf-card__author")
            author = author_node.text.strip() if author_node else ""

            chips = [chip.text.strip() for chip in card.find_all("span", class_="pdf-card__chip")]
            doc_format, pages, file_size = "", "", ""

            for chip in chips:
                if "Format:" in chip:
                    doc_format = chip.replace("Format:", "").strip()
                elif "pages" in chip.lower():
                    pages = chip.lower().replace("pages", "").strip()
                elif any(unit in chip.upper() for unit in ["MB", "KB", "GB"]):
                    file_size = chip.strip()

            dl_btn = card.find("a", class_=re.compile(r"pdf-card__btn--download|pdf-download"))
            direct_pdf_url = ""
            if dl_btn and dl_btn.get("href"):
                direct_pdf_url = urljoin(sub_url, dl_btn["href"])

            if not direct_pdf_url or direct_pdf_url in seen_download_urls:
                continue

            try:
                res = session.get(direct_pdf_url, timeout=30)
                if res.status_code == 200 and res.content.startswith(b"%PDF-"):
                    
                    item_id = f"{category_name}_{book_counter}"
                    filename = f"{item_id}.pdf"
                    file_path = os.path.join(DEST_FOLDER, filename)

                    with open(file_path, "wb") as f:
                        f.write(res.content)

                    metadata_records.append({
                        "id": item_id,
                        "filename": filename,
                        "title": raw_title,
                        "author": author,
                        "description": description,
                        "pages": pages,
                        "file_size": file_size,
                        "format": doc_format,
                        "category": category_name,
                        "subcategory_url": sub_url,
                        "download_url": direct_pdf_url
                    })

                    seen_download_urls.add(direct_pdf_url)
                    print(f"Downloaded [{book_counter}]: {filename} - {raw_title}")
                    book_counter += 1

            except Exception:
                continue

    browser.close()

if metadata_records:
    new_df = pd.DataFrame(metadata_records)
    
    if os.path.exists(CSV_PATH):
        # Append to existing law metadata without replacing headers
        existing_df = pd.read_csv(CSV_PATH)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_csv(CSV_PATH, index=False)
        print(f"\nAppended {len(new_df)} new law records. Total records in {CSV_PATH}: {len(final_df)}")
    else:
        new_df.to_csv(CSV_PATH, index=False)
        print(f"\nSuccessfully downloaded {len(new_df)} files. Saved to {CSV_PATH}")
else:
    print("No valid PDFs were downloaded.")