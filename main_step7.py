from data_loader import DataLoader
from knowledge_base import KnowledgeBase
from intent_dictionary import IntentDictionary
from query_parser import QueryParser
from feature_extractor import FeatureExtractor
from retriever import BM25Retriever
from enhanced_features import EnhancedFeatureExtractor
import time

def run_step_7():
    print("Initializing components...")
    
    # 1. Load Data
    loader = DataLoader()
    loader.load_data()
    products_df = loader.get_products()
    train_df = loader.get_train_pairs()
    
    # 2. Setup Dependencies
    kb = KnowledgeBase()
    kb.build(products_df)
    
    intent_dict = IntentDictionary()
    intent_dict.fit(train_df, products_df)
    
    parser = QueryParser(kb, intent_dict)
    basic_fe = FeatureExtractor(parser)
    
    retriever = BM25Retriever()
    retriever.fit(products_df)
    
    # 3. Initialize Enhanced Extractor
    efe = EnhancedFeatureExtractor(basic_fe, retriever, products_df)
    
    # 4. Test on a Pair
    # Case: "Asus Laptop" query against an "Asus Laptop" product
    q = "لپ تاپ ایسوس"
    # Find an Asus product ID
    asus_pid = products_df[products_df['norm_brand'] == 'ایسوس'].iloc[0]['p_id']
    
    print(f"\n--- Enhanced Feature Extraction ---")
    print(f"Query: {q}")
    print(f"PID: {asus_pid}")
    
    start = time.time()
    feats = efe.extract_features(q, asus_pid)
    end = time.time()
    
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}")
        
    print(f"\nExtraction Time: {end - start:.4f} seconds")

if __name__ == "__main__":
    run_step_7()