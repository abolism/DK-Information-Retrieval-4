import numpy as np

class FeatureExtractor:
    def __init__(self, query_parser):
        self.parser = query_parser

    def extract_features(self, query_text, product_row):
        """
        Generates a feature dictionary for a query-product pair.
        product_row: A single row (Series) from the products dataframe.
        """
        parsed_q = self.parser.get_structured_query(query_text)
        query_tokens = set(self.parser.preprocessor.tokenize(query_text))
        title_tokens = set(self.parser.preprocessor.tokenize(product_row['norm_title']))
        
        features = {}

        # 1. Exact Attribute Matches
        # Check if any brand detected in query matches the product's actual brand
        features['brand_match'] = 1.0 if any(b in product_row['norm_brand'] for b in parsed_q['brands']) else 0.0
        
        # Check if product types/categories match
        features['type_match'] = 1.0 if any(t in product_row['norm_category'] for t in parsed_q['product_types']) else 0.0
        features['cat_match'] = 1.0 if any(c in product_row['norm_category'] for c in parsed_q['categories']) else 0.0

        # 2. Token Overlap Features
        intersection = query_tokens.intersection(title_tokens)
        features['token_overlap_count'] = float(len(intersection))
        features['token_overlap_ratio'] = len(intersection) / len(query_tokens) if len(query_tokens) > 0 else 0.0
        
        # 3. Positional Features
        # Does the first word of the query appear in the product title? (Often the most important word)
        first_token = self.parser.preprocessor.tokenize(query_text)[0] if query_tokens else ""
        features['first_token_match'] = 1.0 if first_token in title_tokens else 0.0

        # 4. Length Features
        features['query_len'] = float(len(query_tokens))
        features['title_len'] = float(len(title_tokens))
        
        # 5. Intent Presence (Metadata about the query itself)
        features['has_brand_intent'] = 1.0 if parsed_q['brands'] else 0.0
        
        return features

    def get_feature_names(self):
        # Helper to maintain consistent column ordering later
        return [
            'brand_match', 'type_match', 'cat_match', 
            'token_overlap_count', 'token_overlap_ratio', 
            'first_token_match', 'query_len', 'title_len', 'has_brand_intent'
        ]