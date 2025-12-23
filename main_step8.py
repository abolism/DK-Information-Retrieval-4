import pandas as pd
from data_loader import DataLoader
from knowledge_base import KnowledgeBase
from intent_dictionary import IntentDictionary
from query_parser import QueryParser
from feature_extractor import FeatureExtractor
from retriever import BM25Retriever
from enhanced_features import EnhancedFeatureExtractor
from dataset_builder import DatasetBuilder

def run_step_8():
    # 1. Initialization (Full Pipeline)
    print("Loading resources...")
    loader = DataLoader()
    loader.load_data()
    products_df = loader.get_products()
    train_df = loader.get_train_pairs()
    
    kb = KnowledgeBase()
    kb.build(products_df)
    
    intent_dict = IntentDictionary()
    intent_dict.fit(train_df, products_df)
    
    parser = QueryParser(kb, intent_dict)
    basic_fe = FeatureExtractor(parser)
    
    retriever = BM25Retriever()
    retriever.fit(products_df)
    
    efe = EnhancedFeatureExtractor(basic_fe, retriever, products_df)
    
    # 2. Build Dataset
    builder = DatasetBuilder(efe, retriever, products_df)
    
    # Using a smaller ratio (e.g., 3 negatives) is often enough and faster
    X, y, groups = builder.build_dataset(train_df, negative_ratio=3)
    
    # 3. Save for Model Training
    # Combine for saving
    full_training_data = X.copy()
    full_training_data['label'] = y
    
    output_filename = 'ltr_training_data2.csv'
    full_training_data.to_csv(output_filename, index=False)
    print(f"Training data saved to {output_filename}")
    
    # Show sample
    print("\n--- Training Data Sample ---")
    print(full_training_data.head())

if __name__ == "__main__":
    run_step_8()