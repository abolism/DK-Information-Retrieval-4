import pandas as pd
from data_loader import DataLoader
from knowledge_base import KnowledgeBase
from intent_dictionary import IntentDictionary
from query_parser import QueryParser
from feature_extractor import FeatureExtractor
from retriever import BM25Retriever
from enhanced_features import EnhancedFeatureExtractor
from ranker import PointwiseRanker
from inference_engine import InferenceEngine

def run_step_11():
    print("=== Initialize Search Engine Components ===")
    
    # 1. Load Data
    loader = DataLoader()
    loader.load_data()
    products_df = loader.get_products()
    train_df = loader.get_train_pairs() # Needed for Intent Dict
    test_queries = loader.get_test_queries()
    
    # 2. Build Knowledge Resources
    kb = KnowledgeBase()
    kb.build(products_df)
    
    intent_dict = IntentDictionary()
    intent_dict.fit(train_df, products_df)
    
    # 3. Initialize Processors
    parser = QueryParser(kb, intent_dict)
    basic_fe = FeatureExtractor(parser)
    
    retriever = BM25Retriever()
    retriever.fit(products_df)
    
    efe = EnhancedFeatureExtractor(basic_fe, retriever, products_df)
    
    # 4. Load Trained Model
    ranker = PointwiseRanker()
    try:
        ranker.load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have run Step 9 to train the model first.")
        return

    # 5. Run Inference
    engine = InferenceEngine(retriever, efe, ranker)
    
    # Run on all test queries
    # Using 'submission.csv' as the final output filename
    engine.generate_predictions(test_queries, output_path='submission.csv')
    
    print("\n=== Process Complete ===")
    print("You can now upload 'submission.csv' to the contest platform.")

if __name__ == "__main__":
    run_step_11()