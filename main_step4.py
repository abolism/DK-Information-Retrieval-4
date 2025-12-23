from data_loader import DataLoader
from intent_dictionary import IntentDictionary

def run_step_4():
    # 1. Load Data
    loader = DataLoader()
    loader.load_data()
    
    # 2. Build Intent Dictionary
    intent_dict = IntentDictionary(min_freq=2)
    intent_dict.fit(loader.get_train_pairs(), loader.get_products())
    
    # 3. Test Mapping
    test_tokens = ["گلکسی", "لنوو", "دیجیتال"] # Galaxy, Lenovo, Digital
    
    print("\n--- Intent Mapping Check ---")
    for t in test_tokens:
        intent = intent_dict.get_intent(t)
        print(f"Token: {t} -> Inferred Brand: {intent['brand']}, Inferred Cat: {intent['category']}")

if __name__ == "__main__":
    run_step_4()