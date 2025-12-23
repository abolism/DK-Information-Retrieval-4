from data_loader import DataLoader

def run_step_1():
    # Initialize Loader
    loader = DataLoader()
    
    # Load and Preprocess
    loader.load_data()
    
    # Verification/Sanity Check
    products = loader.get_products()
    train = loader.get_train_pairs()
    
    print("\n--- Data Stats ---")
    print(f"Total Products: {len(products)}")
    print(f"Total Train Pairs: {len(train)}")
    print(f"Total Test Queries: {len(loader.get_test_queries())}")
    
    print("\n--- Sample Preprocessing ---")
    print(f"Raw Product Title: {products.iloc[0]['title']}")
    print(f"Norm Product Title: {products.iloc[0]['norm_title']}")
    print(f"Search Text: {products.iloc[0]['search_text'][:50]}...")

if __name__ == "__main__":
    run_step_1()