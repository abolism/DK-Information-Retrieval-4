from data_loader import DataLoader
from knowledge_base import KnowledgeBase
from intent_dictionary import IntentDictionary
from query_parser import QueryParser
from feature_extractor import FeatureExtractor

def run_step_6():
    # 1. Load Data & Setup Parser (Standard Pipeline)
    loader = DataLoader()
    loader.load_data()
    products_df = loader.get_products()
    
    kb = KnowledgeBase()
    kb.build(products_df)
    intent_dict = IntentDictionary()
    intent_dict.fit(loader.get_train_pairs(), products_df)
    
    parser = QueryParser(kb, intent_dict)
    
    # 2. Initialize Feature Extractor
    fe = FeatureExtractor(parser)
    
    # 3. Test on a Query-Product Pair
    sample_query = "گوشی سامسونگ"
    # Find a product that actually is a Samsung phone for the test
    target_product = products_df[products_df['brand'].str.contains("سامسونگ", na=False)].iloc[0]
    
    features = fe.extract_features(sample_query, target_product)
    
    print(f"\n--- Feature Extraction Test ---")
    print(f"Query: {sample_query}")
    print(f"Product: {target_product['title']}")
    for k, v in features.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run_step_6()