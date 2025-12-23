from data_loader import DataLoader
from knowledge_base import KnowledgeBase

def run_step_2():
    # 1. Load Data (Reusing Step 1 logic)
    loader = DataLoader()
    loader.load_data()
    products_df = loader.get_products()
    
    # 2. Build Knowledge Base
    kb = KnowledgeBase()
    kb.build(products_df)
    
    # 3. Verification / Sanity Check
    # Let's print some samples to ensure normalization worked and extraction makes sense
    print("\n--- Knowledge Base Samples ---")
    print(f"Sample Brands: {list(kb.get_brands())[:5]}")
    print(f"Sample Product Types: {list(kb.get_product_types())[:5]}")
    
    # Test a known entity if it exists in the sample data (hypothetical check)
    test_brand = "سامسونگ" # Samsung
    if kb.is_brand(test_brand):
        print(f"SUCCESS: '{test_brand}' recognized as Brand.")
    else:
        print(f"WARNING: '{test_brand}' not found in Brands.")

if __name__ == "__main__":
    run_step_2()