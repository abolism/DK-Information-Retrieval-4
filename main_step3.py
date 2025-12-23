from data_loader import DataLoader
from retriever import BM25Retriever
import time

def run_step_3():
    # 1. Load Data
    loader = DataLoader()
    loader.load_data()
    products_df = loader.get_products()
    
    # 2. Initialize and Train Retriever
    retriever = BM25Retriever()
    start_time = time.time()
    retriever.fit(products_df)
    print(f"Indexing Time: {time.time() - start_time:.4f} seconds")
    
    # 3. Test Retrieval
    test_queries = [
        "گوشی سامسونگ",       # Samsung Phone
        "کفش ورزشی قرمز",     # Red Sports Shoes (Attribute heavy)
        "لپ تاپ ایسوس"        # Asus Laptop
    ]
    
    print("\n--- Retrieval Sanity Check ---")
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, top_k=5)
        
        if not results:
            print("No results found.")
            continue
            
        for i, res in enumerate(results):
            # Fetch product details for display
            product_row = products_df[products_df['p_id'] == res['p_id']].iloc[0]
            print(f"Rank {i+1}: [Score: {res['bm25_score']:.4f}] {product_row['title'][:60]}...")

if __name__ == "__main__":
    run_step_3()