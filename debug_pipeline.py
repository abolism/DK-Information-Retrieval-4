import re
from text_processor import PersianPreprocessor
from feature_extractor import FeatureExtractor
import pandas as pd

# Mock objects to test FeatureExtractor in isolation
class MockRetriever:
    def __init__(self):
        self.k1 = 1.5
        self.b = 0.75
        self.avg_dl = 20
        self.idf = {'گوشی': 0.5, 'سامسونگ': 1.0, 'a52': 2.0}

class MockParser:
    def __init__(self):
        self.preprocessor = PersianPreprocessor()
    def parse(self, text):
        norm = self.preprocessor.normalize(text)
        return {
            'detected_brands': {'سامسونگ'} if 'سامسونگ' in norm else set(),
            'detected_categories': {'گوشی'} if 'گوشی' in norm else set()
        }

def test_normalization():
    print("\n--- 1. Testing Normalization Logic ---")
    pp = PersianPreprocessor()
    
    test_cases = [
        ("English", "Samsung Galaxy S21"),
        ("Finglish", "gooshi samsung a22"),
        ("Persian Digits", "گوشی آیفون ۱۳"),  # Persian 13
        ("Mixed", "Galaxy A۱۲ gold"),       # Mixed digits
        ("Synonym", "mobile apple")         # Should map to gooshi
    ]
    
    for label, text in test_cases:
        norm = pp.normalize(text)
        print(f"[{label:15}] Raw: '{text}'  ->  Norm: '{norm}'")

def test_feature_logic():
    print("\n--- 2. Testing Feature Extraction Logic ---")
    # Setup mock data: A Product that is "Samsung A52"
    products_df = pd.DataFrame([{
        'p_id': '100',
        'title': 'گوشی موبایل سامسونگ مدل Galaxy A52', # Raw title has English A52
        'brand': 'Samsung',
        'category': 'Mobile',
        'search_text': 'dummy text'
    }])
    
    fe = FeatureExtractor(MockParser(), MockRetriever(), products_df)
    
    # Case: User searches with PERSIAN digits "a۵۲"
    query = "سامسونگ a۵۲"
    print(f"\nQuery: '{query}' vs Product: 'Galaxy A52'")
    
    # Run extraction
    feats = fe.extract_features(query, '100')
    
    if feats:
        print(f"Digit Match Score: {feats['digit_match']}")
        if feats['digit_match'] == 0.0:
            print("❌ FAILURE: Persian digits in query did not match English digits in title.")
        else:
            print("✅ SUCCESS: Digits matched correctly.")
    else:
        print("Error in extraction.")

if __name__ == "__main__":
    test_normalization()
    test_feature_logic()