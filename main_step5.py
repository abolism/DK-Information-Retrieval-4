from data_loader import DataLoader
from knowledge_base import KnowledgeBase
from intent_dictionary import IntentDictionary
from query_parser import QueryParser

def run_step_5():
    # 1. Load Data
    loader = DataLoader()
    loader.load_data()
    products_df = loader.get_products()
    train_df = loader.get_train_pairs()
    
    # 2. Setup Resources
    kb = KnowledgeBase()
    kb.build(products_df)
    
    intent_dict = IntentDictionary()
    intent_dict.fit(train_df, products_df)
    
    # 3. Initialize Parser
    parser = QueryParser(kb, intent_dict)
    
    # 4. Test Queries
    test_cases = [
        "گوشی سامسونگ سری اس", # Samsung S Series Phone
        "لپ تاپ لنوو مدل ایدیاپد", # Lenovo Ideapad Laptop
        "کفش ورزشی نایکی" # Nike Sports Shoes
    ]
    
    print("\n--- Query Parsing Results ---")
    for q in test_cases:
        res = parser.get_structured_query(q)
        print(f"Query: {q}")
        print(f"  Detected Brands: {res['brands']}")
        print(f"  Detected Types:  {res['product_types']}")
        print(f"  Detected Cats:   {res['categories']}")

if __name__ == "__main__":
    run_step_5()