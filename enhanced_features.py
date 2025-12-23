import numpy as np
from difflib import SequenceMatcher
# from feature_extractor import FeatureExtractor
from feature_extractor import SOTAFeatureExtractor
from collections import Counter

class EnhancedFeatureExtractor:
    def __init__(self, basic_extractor, retriever, products_df):
        """
        basic_extractor: Instance of FeatureExtractor (Step 6)
        retriever: Instance of BM25Retriever (Step 3)
        products_df: The products dataframe (for fast lookups)
        """
        self.basic_extractor = basic_extractor
        self.retriever = retriever
        self.products_df = products_df.set_index('p_id')
        self.preprocessor = self.basic_extractor.parser.preprocessor
        
        # Create a p_id -> internal_doc_idx map if needed, 
        # but for exact BM25 calc on single pairs, we can re-compute efficiently.
        # However, to use the retriever's IDF, we access it directly.

    def get_jaccard_sim(self, set_a, set_b):
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union

    def get_fuzzy_score(self, str_a, str_b):
        return SequenceMatcher(None, str_a, str_b).ratio()

    def calculate_bm25_pair(self, query_tokens, product_text):
        """
        Calculates BM25 score for a specific (Query, Product) pair on the fly.
        This is needed for Training Data where we have the ground truth pair 
        and need its score, even if we didn't run a full retrieval.
        """
        score = 0.0
        prod_tokens = self.preprocessor.tokenize(product_text)
        doc_len = len(prod_tokens)
        prod_counts = Counter(prod_tokens)
        
        for token in query_tokens:
            if token not in self.retriever.idf:
                continue
                
            idf = self.retriever.idf[token]
            tf = prod_counts[token]
            
            numerator = tf * (self.retriever.k1 + 1)
            denominator = tf + self.retriever.k1 * (1 - self.retriever.b + self.retriever.b * (doc_len / self.retriever.avg_dl))
            
            score += idf * (numerator / denominator)
            
        return score

    def extract_features(self, query_text, p_id):
        """
        Master feature extraction method.
        """
        # 1. Get Product Data
        if p_id not in self.products_df.index:
            return None # Should not happen in valid flow
        
        prod_row = self.products_df.loc[p_id]
        
        # 2. Basic Semantic Features (Step 6)
        base_feats = self.basic_extractor.extract_features(query_text, prod_row)
        
        # 3. Enhanced Statistical Features
        query_tokens = self.preprocessor.tokenize(query_text)
        query_token_set = set(query_tokens)
        title_token_set = set(self.preprocessor.tokenize(prod_row['norm_title']))
        
        # BM25 Score
        # We use the 'search_text' field which contains title+brand+cat+attr
        bm25 = self.calculate_bm25_pair(query_tokens, prod_row['search_text'])
        
        # Text Similarity (Title focus)
        jaccard = self.get_jaccard_sim(query_token_set, title_token_set)
        fuzzy = self.get_fuzzy_score(query_text, prod_row['norm_title'])
        
        # 4. Combine
        enhanced_feats = base_feats.copy()
        enhanced_feats['bm25_score'] = bm25
        enhanced_feats['title_jaccard'] = jaccard
        enhanced_feats['title_fuzzy'] = fuzzy
        
        return enhanced_feats

    def get_feature_names(self):
        return self.basic_extractor.get_feature_names() + ['bm25_score', 'title_jaccard', 'title_fuzzy']