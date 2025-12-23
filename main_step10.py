from data_loader import DataLoader
from knowledge_base import KnowledgeBase
from intent_dictionary import IntentDictionary
from query_parser import QueryParser
from feature_extractor import FeatureExtractor
from retriever import BM25Retriever
from enhanced_features import EnhancedFeatureExtractor
from ranker import PointwiseRanker
from evaluator import SearchEvaluator

def run_step_10():
    # 1. Setup full pipeline
    loader = DataLoader()
    loader.load_data()
    products_df = loader.get_products()
    train_df = loader.get_train_pairs()
    
    # Initialize components
    kb = KnowledgeBase()
    kb.build(products_df)
    intent_dict = IntentDictionary()
    intent_dict.fit(train_df, products_df)
    parser = QueryParser(kb, intent_dict)
    basic_fe = FeatureExtractor(parser)
    retriever = BM25Retriever()
    retriever.fit(products_df)
    efe = EnhancedFeatureExtractor(basic_fe, retriever, products_df)
    
    # Load trained model
    ranker = PointwiseRanker()
    ranker.load_model()
    
    # 2. Slice Data for Validation
    # Let's take 200 samples to keep evaluation fast
    val_sample = train_df.sample(200, random_state=42)
    
    # 3. Run Evaluation
    evaluator = SearchEvaluator(retriever, efe, ranker)
    results = evaluator.evaluate(val_sample)
    
    print("\n--- Validation Results ---")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    run_step_10()