import os
import json
import pandas as pd

def load_products_json(file_path):
    """Loads products list from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_products_csv(file_path):
    """Loads products list from CSV file and conforms it to the JSON structure."""
    df = pd.read_csv(file_path)
    products = []
    for _, row in df.iterrows():
        # Reconstruct the specs dictionary
        specs = {}
        if 'spec_feature_1' in row and pd.notna(row['spec_feature_1']):
            specs['feature_1'] = str(row['spec_feature_1'])
        if 'spec_ram' in row and pd.notna(row['spec_ram']):
            specs['ram'] = str(row['spec_ram'])
        if 'spec_storage' in row and pd.notna(row['spec_storage']):
            specs['storage'] = str(row['spec_storage'])

        product = {
            "product_id": str(row.get("product_id", "")),
            "category": str(row.get("category", "")),
            "product_name": str(row.get("product_name", "")),
            "brand": str(row.get("brand", "")),
            "price_inr": int(row.get("price_inr", 0)) if pd.notna(row.get("price_inr")) else 0,
            "specs": specs,
            "rating": float(row.get("rating", 0.0)) if pd.notna(row.get("rating")) else 0.0,
            "reviews_count": int(row.get("reviews_count", 0)) if pd.notna(row.get("reviews_count")) else 0,
            "customer_review": str(row.get("customer_review", "")) if pd.notna(row.get("customer_review")) else ""
        }
        products.append(product)
    return products

def load_dataset(directory="."):
    """
    Tries to load electronics dataset from the given directory.
    Prefers JSON, falls back to CSV.
    """
    json_path = os.path.join(directory, "electronics_catalog_100.json")
    csv_path = os.path.join(directory, "electronics_catalog_100.csv")

    if os.path.exists(json_path):
        try:
            return load_products_json(json_path)
        except Exception as e:
            print(f"Error loading JSON ({e}), falling back to CSV...")
    
    if os.path.exists(csv_path):
        return load_products_csv(csv_path)
    
    raise FileNotFoundError("Could not find electronics_catalog_100.json or electronics_catalog_100.csv in workspace.")

def construct_document(product):
    """
    Creates a rich text description of a product, combining brand, name, category,
    price, technical specs, and reviews. This text is what gets indexed and embedded.
    """
    name = product.get("product_name", "Unknown Product")
    category = product.get("category", "")
    brand = product.get("brand", "")
    price = product.get("price_inr", 0)
    
    # Specs formatting
    specs = product.get("specs", {})
    specs_list = []
    if isinstance(specs, dict):
        for k, v in specs.items():
            specs_list.append(f"{k}: {v}")
    specs_str = ", ".join(specs_list) if specs_list else "None"
    
    review = product.get("customer_review", "")
    
    # Build text content
    text = (
        f"Product Name: {name} | "
        f"Category: {category} | "
        f"Brand: {brand} | "
        f"Price: INR {price:,} | "
        f"Specifications: {specs_str} | "
        f"Reviews: {review}"
    )
    return text

def get_indexed_documents(products):
    """
    Returns a list of tuple (product_id, doc_text) for indexing.
    """
    return [(p["product_id"], construct_document(p)) for p in products]

if __name__ == "__main__":
    # Quick test
    try:
        products = load_dataset(".")
        print(f"Loaded {len(products)} products.")
        print("First product document example:")
        doc_id, doc_text = get_indexed_documents(products)[0]
        print(f"ID: {doc_id}\nContent: {doc_text}")
    except Exception as e:
        print(f"Test failed: {e}")
