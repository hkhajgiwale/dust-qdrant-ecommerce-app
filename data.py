# data.py
"""
Robust product page scraper for Shopify-like storefronts.
Functions:
 - fetch_collection_product_urls(collection_url, limit)
 - fetch_product_data(product_url)
"""

import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

HEADERS = {"User-Agent": "Mozilla/5.0 (DemoBot/1.0)"}


def fetch_collection_product_urls(collection_url: str, limit: int = 50) -> list:
    """
    Scrape a collection page to find product page URLs.
    Returns a list of full product URLs (deduplicated).
    This is a simple scraper; some sites require JS rendering (use Playwright in that case).
    """
    resp = requests.get(collection_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls = []
    for a in soup.select("a[href*='/products/']"):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(collection_url, href)
        urls.append(full)

    # dedupe while preserving order
    seen = set()
    filtered = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            filtered.append(u)
    return filtered[:limit]


# ---- helpers for product extraction ----
def parse_ld_json(soup):
    """Return first product-like LD+JSON object if present."""
    for tag in soup.select("script[type='application/ld+json']"):
        try:
            txt = tag.string or ""
            if not txt:
                continue
            data = json.loads(txt)
            # data can be list or dict
            if isinstance(data, list):
                for item in data:
                    t = item.get("@type", "")
                    if isinstance(t, str) and "product" in t.lower():
                        return item
            elif isinstance(data, dict):
                t = data.get("@type", "")
                if isinstance(t, str) and "product" in t.lower():
                    return data
                # sometimes the product is nested in graph
                if "@graph" in data and isinstance(data["@graph"], list):
                    for item in data["@graph"]:
                        if "product" in (item.get("@type", "") or "").lower():
                            return item
        except Exception:
            continue
    return None


def extract_price_from_ld(ld):
    if not ld:
        return None
    offers = ld.get("offers")
    if isinstance(offers, dict):
        p = offers.get("price")
        try:
            if p is not None:
                return float(str(p).replace(",", ""))
        except:
            pass
    return None


def extract_images_from_ld(ld):
    if not ld:
        return []
    imgs = ld.get("image") or ld.get("images") or []
    if isinstance(imgs, str):
        return [imgs]
    return imgs


def extract_og(soup):
    meta = {}
    og_title = soup.select_one('meta[property="og:title"], meta[name="og:title"]')
    if og_title and og_title.get("content"):
        meta["title"] = og_title.get("content")
    og_desc = soup.select_one('meta[property="og:description"], meta[name="og:description"], meta[name="description"]')
    if og_desc and og_desc.get("content"):
        meta["description"] = og_desc.get("content")
    og_image = soup.select_one('meta[property="og:image"], meta[name="og:image"]')
    if og_image and og_image.get("content"):
        meta["images"] = [og_image.get("content")]
    return meta


def extract_price_by_selectors(soup):
    selectors = [
        ".product-single__price", ".price", ".product-price", ".price__regular", ".price--large",
        ".money", ".product__price", ".variant__price", ".price-item--regular", ".price-item--sale"
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(" ", strip=True)
            # find number
            m = re.search(r"([\d\.,]+)", text.replace(",", ""))
            if m:
                try:
                    return float(m.group(1))
                except:
                    continue
    return None


def extract_price_from_js(html):
    m = re.search(r'"price"\s*:\s*"?(?P<p>[\d\.,]+)"?', html)
    if m:
        try:
            return float(m.group("p").replace(",", ""))
        except:
            pass
    m = re.search(r'price\W*:\W*(?P<p>[\d\.,]+)', html)
    if m:
        try:
            return float(m.group("p").replace(",", ""))
        except:
            pass
    return None


def fetch_product_data(product_url: str) -> dict:
    """
    Robustly extract product data (title, description, price, images).
    Returns dictionary with keys:
        product_url, title, description, price (float or None), images (list[str])
    """
    resp = requests.get(product_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    ld = parse_ld_json(soup)
    title = None
    description = None
    price = None
    images = []

    if ld:
        title = ld.get("name") or ld.get("headline")
        description = ld.get("description") or ld.get("review", {}).get("reviewBody")
        price = extract_price_from_ld(ld)
        images = extract_images_from_ld(ld)

    # OG fallback
    og = extract_og(soup)
    if og:
        title = title or og.get("title")
        description = description or og.get("description")
        images = images or og.get("images", [])

    # selectors fallback
    if not title:
        el = soup.select_one("h1, .product-single__title, .product-title, .product__title")
        if el:
            title = el.get_text(strip=True)
    if not description:
        desc_el = soup.select_one(".product-single__description, .product-description, .description, .rte")
        if desc_el:
            description = desc_el.get_text(" ", strip=True)
    if price is None:
        price = extract_price_by_selectors(soup)

    # last-resort JS parse
    if price is None:
        price = extract_price_from_js(html)

    # images fallback: gather img srcs
    if not images:
        imgs = []
        for img in soup.select("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not src:
                continue
            full = urljoin(product_url, src)
            # heuristics to skip icons/thumbnails: skip very small images in filename or containing 'icon'
            if "icon" in full or "sprite" in full:
                continue
            imgs.append(full)
        # dedupe
        seen = set()
        images = [x for x in imgs if not (x in seen or seen.add(x))]

    return {
        "product_url": product_url,
        "title": (title or "").strip(),
        "description": (description or "").strip(),
        "price": float(price) if price is not None else None,
        "images": images
    }
