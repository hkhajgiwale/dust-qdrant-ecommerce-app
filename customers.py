# customers.py
"""
Generate synthetic customer preference embeddings and store them in Qdrant.
Each customer has:
 - name, email
 - interests (sampled keywords)
 - embedded preference vector (text-based)
"""

import uuid
import random
from faker import Faker
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff, PointStruct
from client import qdrant_client
from embedder import embed_text, get_text_embedding_dimension

fake = Faker()

def ensure_customers_collection(collection_name="customers"):
    dim = get_text_embedding_dimension()
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={"text": VectorParams(size=dim, distance=Distance.COSINE)},
            hnsw_config=HnswConfigDiff(m=16, ef_construct=256),
        )
        print(f"✅ Created 'customers' collection (dim={dim})")
    else:
        print(f"ℹ️  'customers' collection already exists")


def generate_synthetic_customers(n=10):
    """
    Create synthetic customer profiles with random interests.
    """
    skincare_keywords = [
        "hydration", "anti-aging", "sunscreen", "brightening", "vitamin C",
        "retinol", "cleanser", "exfoliator", "serum", "vegan skincare",
        "moisturizer", "fragrance-free", "acne care", "sensitive skin"
    ]

    customers = []
    for _ in range(n):
        name = fake.name()
        email = fake.email()
        interests = random.sample(skincare_keywords, k=random.randint(2, 4))
        interest_text = " ".join(interests)
        customers.append({
            "id": str(uuid.uuid4()),
            "name": name,
            "email": email,
            "interests": interests,
            "interest_text": interest_text
        })
    return customers


def ingest_customers(customers, collection_name="customers"):
    """
    Embed interests and upsert into Qdrant.
    """
    points = []
    for c in customers:
        try:
            vec = embed_text(c["interest_text"])
            payload = {
                "name": c["name"],
                "email": c["email"],
                "interests": c["interests"]
            }
            points.append(PointStruct(id=c["id"], vector={"text": vec}, payload=payload))
        except Exception as e:
            print(f"⚠️ Failed to embed customer {c['name']}: {e}")

    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Upserted {len(points)} customer profiles into Qdrant.")


def setup_and_ingest_customers(n=10):
    ensure_customers_collection()
    customers = generate_synthetic_customers(n)
    ingest_customers(customers)
    print("🎯 Customer embedding setup complete.")
