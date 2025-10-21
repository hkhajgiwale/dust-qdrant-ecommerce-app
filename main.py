# main.py

import argparse
from ingest import ingest_from_store

def main():
    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    parser.add_argument("--collection_url", type=str, required=True, help="URL of collection page to scrape")
    parser.add_argument("--limit", type=int, default=20, help="Max number of products to ingest")
    args = parser.parse_args()

    print("Starting ingestion from:", args.collection_url)
    ingest_from_store(args.collection_url, limit=args.limit)
    print("Done.")

if __name__ == "__main__":
    main()
