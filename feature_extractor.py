# import numpy as np

# class FeatureExtractor:
#     def __init__(self, query_parser):
#         self.parser = query_parser

#     def extract_features(self, query_text, product_row):
#         """
#         Generates a feature dictionary for a query-product pair.
#         product_row: A single row (Series) from the products dataframe.
#         """
#         parsed_q = self.parser.get_structured_query(query_text)
#         query_tokens = set(self.parser.preprocessor.tokenize(query_text))
#         title_tokens = set(self.parser.preprocessor.tokenize(product_row['norm_title']))
        
#         features = {}

#         # 1. Exact Attribute Matches
#         # Check if any brand detected in query matches the product's actual brand
#         features['brand_match'] = 1.0 if any(b in product_row['norm_brand'] for b in parsed_q['brands']) else 0.0
        
#         # Check if product types/categories match
#         features['type_match'] = 1.0 if any(t in product_row['norm_category'] for t in parsed_q['product_types']) else 0.0
#         features['cat_match'] = 1.0 if any(c in product_row['norm_category'] for c in parsed_q['categories']) else 0.0

#         # 2. Token Overlap Features
#         intersection = query_tokens.intersection(title_tokens)
#         features['token_overlap_count'] = float(len(intersection))
#         features['token_overlap_ratio'] = len(intersection) / len(query_tokens) if len(query_tokens) > 0 else 0.0
        
#         # 3. Positional Features
#         # Does the first word of the query appear in the product title? (Often the most important word)
#         first_token = self.parser.preprocessor.tokenize(query_text)[0] if query_tokens else ""
#         features['first_token_match'] = 1.0 if first_token in title_tokens else 0.0

#         # 4. Length Features
#         features['query_len'] = float(len(query_tokens))
#         features['title_len'] = float(len(title_tokens))
        
#         # 5. Intent Presence (Metadata about the query itself)
#         features['has_brand_intent'] = 1.0 if parsed_q['brands'] else 0.0
        
#         return features

#     def get_feature_names(self):
#         # Helper to maintain consistent column ordering later
#         return [
#             'brand_match', 'type_match', 'cat_match', 
#             'token_overlap_count', 'token_overlap_ratio', 
#             'first_token_match', 'query_len', 'title_len', 'has_brand_intent'
#         ]

import Levenshtein # pip install python-Levenshtein

import re
import numpy as np

# # Try to import Levenshtein, fallback to simple implementation if missing
# try:
#     import Levenshtein
# except ImportError:
#     Levenshtein = None

class FeatureExtractor:
    def __init__(self, parser, retriever, products_df):
        self.parser = parser
        self.retriever = retriever
        self.products_df = products_df.set_index('p_id')
        
    def simple_ratio(self, s1, s2):
        # Fallback if python-Levenshtein is not installed
        if not s1 or not s2: return 0.0
        return 1.0 if s1 in s2 or s2 in s1 else 0.0

    def extract_features(self, query, pid):
        # Ensure PID is string
        pid = str(pid)
        if pid not in self.products_df.index: return None
        
        product = self.products_df.loc[pid]
        
        # --- NORMALIZATION ---
        # Normalize BOTH query and title to ensure common ground (English numbers, unified chars)
        norm_query = self.parser.preprocessor.normalize(query)
        norm_title = self.parser.preprocessor.normalize(product['title'])
        
        # 1. Parse Query
        query_info = self.parser.parse(query) # Parser uses normalized internally
        
        # 2. Token Sets (use Normalized versions)
        q_tokens = set(norm_query.split())
        t_tokens = set(norm_title.split())
        
        # 3. Precise Brand Match
        p_brand = self.parser.preprocessor.normalize(product['brand'])
        brand_match = 1.0 if any(b in p_brand for b in query_info['detected_brands']) else 0.0
        
        # 4. Digit/Model Number Match (FIXED)
        # Extract digits from the NORMALIZED query (where '۵' -> '5')
        q_digits = re.findall(r'\d+', norm_query)
        digit_match = 1.0
        if q_digits:
            # Check against NORMALIZED title (where '۵' -> '5')
            # This ensures '۵۲' matches '52'
            if not any(d in norm_title for d in q_digits):
                digit_match = 0.0
        
        # 5. Category Consistency
        p_cat = self.parser.preprocessor.normalize(product['category'])
        cat_match = 1.0 if any(c in p_cat for c in query_info['detected_categories']) else 0.0

        # 6. Semantic Similarity (Levenshtein on Normalized Text)
        if Levenshtein:
            lev_score = Levenshtein.ratio(norm_query, norm_title)
        else:
            lev_score = self.simple_ratio(norm_query, norm_title)

        # 7. BM25 Score
        bm25 = self.get_bm25_score(query, pid)

        return {
            'bm25_score': bm25,
            'brand_match': brand_match,
            'cat_match': cat_match,
            'digit_match': digit_match, # Now robust
            'levenshtein': lev_score,
            'token_overlap_ratio': len(q_tokens & t_tokens) / len(q_tokens) if q_tokens else 0,
            'title_len': float(len(t_tokens)),
            'query_len': float(len(q_tokens))
        }
    
    def get_bm25_score(self, query, pid):
        # Calculate BM25 for a specific pair on-the-fly
        # We access the retriever's internal structures
        score = 0.0
        tokens = self.parser.preprocessor.tokenize(query)
        
        # Find internal doc index for this PID
        # Note: Inverted index stores internal integer IDs, not PIDs.
        # We need a map if we want exact O(1), but for simplicity in this flow:
        # We rely on the retriever having pre-calculated scores OR recalculate.
        # Since recalculating is fast for one doc:
        
        prod_text = self.products_df.loc[pid]['search_text']
        doc_tokens = self.parser.preprocessor.tokenize(prod_text)
        doc_len = len(doc_tokens)
        from collections import Counter
        doc_counts = Counter(doc_tokens)
        
        for token in tokens:
            if token not in self.retriever.idf: continue
            idf = self.retriever.idf[token]
            tf = doc_counts[token]
            num = tf * (self.retriever.k1 + 1)
            den = tf + self.retriever.k1 * (1 - self.retriever.b + self.retriever.b * (doc_len / self.retriever.avg_dl))
            score += idf * (num / den)
            
        return score