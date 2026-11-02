"""
Scraper for Para Fendri public skincare product pages.

Scope:
- Public category pages only.
- Public product detail pages only.
- No personal data, account data, cart data, or private data is collected.

Output:
- data/raw/products/parafendri_products_raw.csv

Run:
    python scripts/scraper_parafendri.py
"""

import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://parafendri.tn"

CATEGORY_URLS = [
    # Face care by skin type / concern
    "https://parafendri.tn/18-peaux-grasses",
    "https://parafendri.tn/75-peaux-seches",
    "https://parafendri.tn/19-peaux-mixtes",
    "https://parafendri.tn/20-peaux-acneiques",
    "https://parafendri.tn/23-peaux-sensibles-",
    "https://parafendri.tn/24-anti-taches-et-depigmentants",
    "https://parafendri.tn/25-apaisants",
    "https://parafendri.tn/88-peeling-du-visage",
    "https://parafendri.tn/27-nettoyants-et-demaquillants",

    # Sun care useful for skincare routines
    "https://parafendri.tn/31-creme-solaire-",
    "https://parafendri.tn/34-apres-solaire",
    "https://parafendri.tn/96-caudalie-protection-solaire-",
]

HEADERS = {
    "User-Agent": (
        "RenYouAIInternshipBot/1.0 "
        "(educational internship project; contact: student@example.com)"
    )
}

RAW_OUTPUT_PATH = Path("data/raw/products/parafendri_products_raw.csv")

REQUEST_TIMEOUT = 20
MIN_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 2.5


def polite_sleep() -> None:
    """
    Sleep between requests to avoid overloading the website.
    """
    time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))


def fetch_html(url: str) -> BeautifulSoup:
    """
    Fetch a webpage and return a BeautifulSoup parser.
    """
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    return BeautifulSoup(response.text, "lxml")


def clean_text(value) -> str:
    """
    Normalize text extracted from HTML.
    """
    if value is None:
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip()

    return value


def parse_price_tnd(text: str) -> float | None:
    """
    Extract a TND price from text.

    Examples:
        '30,000 TND' -> 30.0
        '30.500 TND' -> 30.5
    """
    if not text:
        return None

    match = re.search(r"(\d+[,.]?\d*)\s*TND", text, flags=re.IGNORECASE)

    if not match:
        return None

    price = match.group(1).replace(",", ".")

    try:
        return float(price)
    except ValueError:
        return None


def get_pagination_urls(category_url: str, soup: BeautifulSoup) -> list[str]:
    """
    Extract pagination URLs from a category page.

    The first category URL is always included.
    """
    urls = {category_url}
    base_category_url = category_url.split("?")[0]

    for link in soup.select("a[href]"):
        href = link.get("href")

        if not href:
            continue

        full_url = urljoin(BASE_URL, href)

        is_same_category = base_category_url in full_url
        is_pagination = "page=" in full_url or "?page=" in full_url

        if is_same_category and is_pagination:
            urls.add(full_url)

    return sorted(urls)


def is_likely_product_name(product_name: str) -> bool:
    """
    Avoid collecting menu/category/navigation links as products.
    """
    if not product_name:
        return False

    normalized_name = product_name.lower().strip()

    ignored_exact_words = {
        "accueil",
        "soins",
        "visage",
        "solaire",
        "peaux",
        "femmes",
        "hommes",
        "catégories",
        "categories",
        "sous-catégories",
        "sous-categories",
        "marques",
        "promotions",
        "nouveautés",
        "nouveautes",
    }

    if normalized_name in ignored_exact_words:
        return False

    if len(normalized_name) < 5:
        return False

    return True


def extract_brand_from_block(block: BeautifulSoup, product_name: str) -> str:
    """
    Try to infer brand from the product block.
    """
    if block is None:
        return ""

    brand_selectors = [
        ".product-manufacturer",
        ".manufacturer",
        ".brand",
        ".product-brand",
    ]

    for selector in brand_selectors:
        brand_tag = block.select_one(selector)

        if brand_tag:
            brand_name = clean_text(brand_tag.get_text(" "))

            if brand_name and brand_name.lower() != product_name.lower():
                return brand_name

    links = block.select("a[href]")
    link_texts = [
        clean_text(link.get_text(" "))
        for link in links
        if clean_text(link.get_text(" "))
    ]

    if product_name in link_texts:
        product_index = link_texts.index(product_name)

        if product_index > 0:
            possible_brand = link_texts[product_index - 1]

            ignored_brand_values = {
                "add to compare",
                "ajouter au comparateur",
                "ajouter au panier",
            }

            if possible_brand.lower() not in ignored_brand_values:
                return possible_brand

    return ""


def clean_short_description(
    block_text: str,
    product_name: str,
    brand_name: str,
) -> str:
    """
    Remove repeated noise from the product block text.
    """
    short_description = clean_text(block_text)

    noise_patterns = [
        product_name,
        brand_name,
        "Ajouter au panier",
        "Add to Compare",
        "Disponibilité:",
        "Disponibilite:",
        "En stock",
        "Rupture de stock",
    ]

    for noise in noise_patterns:
        if noise:
            short_description = short_description.replace(noise, " ")

    short_description = re.sub(
        r"\d+[,.]?\d*\s*TND",
        " ",
        short_description,
        flags=re.IGNORECASE,
    )
    short_description = re.sub(r"\s+", " ", short_description).strip()

    return short_description


def parse_product_cards(soup: BeautifulSoup, category_url: str) -> list[dict]:
    """
    Parse product cards from a category listing page.

    The site HTML does not consistently expose standard product-card classes,
    so this parser detects product-like links from headings and nearby blocks.
    """
    products = []
    product_candidates = []

    for heading in soup.select("h2, h3"):
        link = heading.select_one("a[href]")

        if not link:
            continue

        product_name = clean_text(link.get_text(" "))
        product_url = urljoin(BASE_URL, link.get("href"))

        if not is_likely_product_name(product_name):
            continue

        product_candidates.append((heading, product_name, product_url))

    for heading, product_name, product_url in product_candidates:
        block = heading.find_parent()

        for _ in range(5):
            if block is None:
                break

            block_text = clean_text(block.get_text(" "))

            has_price = "TND" in block_text.upper()
            has_availability = "disponibilité" in block_text.lower() or "disponibilite" in block_text.lower()

            if has_price or has_availability:
                break

            block = block.find_parent()

        block_text = clean_text(block.get_text(" ")) if block else ""

        price_tnd = parse_price_tnd(block_text)

        lower_block_text = block_text.lower()

        if "rupture de stock" in lower_block_text:
            availability = "out_of_stock"
        elif "en stock" in lower_block_text:
            availability = "in_stock"
        else:
            availability = "unknown"

        brand_name = extract_brand_from_block(block, product_name)
        short_description = clean_short_description(
            block_text=block_text,
            product_name=product_name,
            brand_name=brand_name,
        )

        products.append(
            {
                "product_name": product_name,
                "brand_name": brand_name,
                "price_tnd": price_tnd,
                "availability": availability,
                "short_description": short_description,
                "product_url": product_url,
                "category_url": category_url,
            }
        )

    return products


def extract_product_image_url(soup: BeautifulSoup) -> str:
    """
    Extract a valid product image URL while ignoring loaders/placeholders.
    """
    image_candidates = soup.select(
        "img.js-qv-product-cover, "
        ".product-cover img, "
        "img[itemprop='image'], "
        "img"
    )

    blocked_patterns = [
        "loader.svg",
        "spinner",
        "placeholder",
        "logo",
        "icon",
        "blank",
    ]

    for img in image_candidates:
        src = (
            img.get("data-full-size-image-url")
            or img.get("data-src")
            or img.get("src")
        )

        if not src:
            continue

        image_url = urljoin(BASE_URL, src)

        if any(pattern in image_url.lower() for pattern in blocked_patterns):
            continue

        return image_url

    return ""


def extract_detail_brand_name(soup: BeautifulSoup) -> str:
    """
    Extract brand name from product detail page if available.
    """
    selectors = [
        ".product-manufacturer a",
        ".manufacturer a",
        ".brand a",
        ".product-brand a",
        "[itemprop='brand']",
    ]

    for selector in selectors:
        brand_tag = soup.select_one(selector)

        if brand_tag:
            brand_name = clean_text(brand_tag.get_text(" "))

            if brand_name:
                return brand_name

    return ""


def extract_detail_description(soup: BeautifulSoup) -> str:
    """
    Extract detailed product description from common product information blocks.
    """
    description_parts = []

    selectors = [
        ".product-description",
        "#description",
        ".tabs .tab-content",
        ".product-information",
        "[itemprop='description']",
        ".product-description-short",
        ".description",
    ]

    for selector in selectors:
        for tag in soup.select(selector):
            text = clean_text(tag.get_text(" "))

            if text and text not in description_parts:
                description_parts.append(text)

    return " ".join(description_parts)


def extract_ingredients_or_composition(page_text: str) -> str:
    """
    Extract ingredient/composition-related text from the full product page text.

    This is heuristic because product pages do not always expose structured
    ingredient fields.
    """
    if not page_text:
        return ""

    normalized_page_text = clean_text(page_text)
    lower_text = normalized_page_text.lower()

    ingredient_keywords = [
        "composition",
        "ingrédients",
        "ingredients",
        "inci",
    ]

    stop_keywords = [
        "produits similaires",
        "dans la même catégorie",
        "dans la meme categorie",
        "avis",
        "commentaires",
        "accessoires",
    ]

    for keyword in ingredient_keywords:
        start_index = lower_text.find(keyword)

        if start_index == -1:
            continue

        end_index = len(normalized_page_text)

        for stop_keyword in stop_keywords:
            stop_index = lower_text.find(stop_keyword, start_index + len(keyword))

            if stop_index != -1:
                end_index = min(end_index, stop_index)

        extracted = normalized_page_text[start_index:end_index]
        extracted = clean_text(extracted)

        # Safety limit to avoid storing a whole page by mistake.
        return extracted[:2000]

    return ""


def parse_product_detail(product_url: str) -> dict:
    """
    Parse one product detail page.
    """
    empty_result = {
        "detail_description": "",
        "ingredients": "",
        "image_url": "",
        "detail_brand_name": "",
    }

    try:
        soup = fetch_html(product_url)
    except requests.RequestException as exc:
        print(f"Failed product detail {product_url}: {exc}")
        return empty_result

    page_text = clean_text(soup.get_text(" "))

    image_url = extract_product_image_url(soup)
    detail_brand_name = extract_detail_brand_name(soup)
    detail_description = extract_detail_description(soup)
    ingredients = extract_ingredients_or_composition(page_text)

    return {
        "detail_description": detail_description,
        "ingredients": ingredients,
        "image_url": image_url,
        "detail_brand_name": detail_brand_name,
    }


def scrape_categories(category_urls: list[str]) -> pd.DataFrame:
    """
    Scrape listing pages for all target categories.
    """
    all_products = []

    for category_url in category_urls:
        print(f"Fetching category: {category_url}")

        try:
            soup = fetch_html(category_url)
        except requests.RequestException as exc:
            print(f"Failed to fetch category {category_url}: {exc}")
            continue

        page_urls = get_pagination_urls(category_url, soup)
        print(f"Found {len(page_urls)} pages")

        for page_url in tqdm(page_urls, desc=f"Scraping {category_url}"):
            try:
                page_soup = fetch_html(page_url)
                products = parse_product_cards(page_soup, category_url)
                all_products.extend(products)
                polite_sleep()

            except requests.RequestException as exc:
                print(f"Failed to fetch listing page {page_url}: {exc}")

    df = pd.DataFrame(all_products)

    if not df.empty:
        df = df.drop_duplicates(
            subset=["product_url", "product_name"],
        ).reset_index(drop=True)

    return df


def enrich_with_product_details(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Visit product detail pages and enrich listing data.
    """
    if products_df.empty:
        return products_df

    enriched_rows = []

    for _, row in tqdm(
        products_df.iterrows(),
        total=len(products_df),
        desc="Scraping product details",
    ):
        row_dict = row.to_dict()
        product_url = row_dict.get("product_url", "")

        if product_url:
            detail_data = parse_product_detail(product_url)
            row_dict.update(detail_data)
            polite_sleep()
        else:
            row_dict.update(
                {
                    "detail_description": "",
                    "ingredients": "",
                    "image_url": "",
                    "detail_brand_name": "",
                }
            )

        enriched_rows.append(row_dict)

    enriched_df = pd.DataFrame(enriched_rows)

    if not enriched_df.empty:
        enriched_df = enriched_df.drop_duplicates(
            subset=["product_url", "product_name"],
        ).reset_index(drop=True)

    return enriched_df


def main() -> None:
    """
    Main scraping pipeline.
    """
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    products_df = scrape_categories(CATEGORY_URLS)
    print("Scraped listing products:", products_df.shape)

    enriched_df = enrich_with_product_details(products_df)
    print("Scraped enriched products:", enriched_df.shape)

    enriched_df.to_csv(
        RAW_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved raw products to: {RAW_OUTPUT_PATH}")

    if not enriched_df.empty:
        print("\nQuick quality report:")
        print("Non-empty ingredients:", (enriched_df["ingredients"].fillna("") != "").sum())
        print("Non-empty detail_description:", (enriched_df["detail_description"].fillna("") != "").sum())
        print("Non-empty image_url:", (enriched_df["image_url"].fillna("") != "").sum())
        print("Loader images:", enriched_df["image_url"].fillna("").str.contains("loader.svg", case=False).sum())


if __name__ == "__main__":
    main()