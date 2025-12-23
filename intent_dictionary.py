import pandas as pd
from collections import defaultdict
from text_processor import PersianPreprocessor

class IntentDictionary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.preprocessor = PersianPreprocessor()
        self.token_to_brand = {}
        self.token_to_category = {}

    def fit(self, train_df, products_df):
        """
        Learns associations between query tokens and product attributes.
        """
        print("Mining Intent Dictionary from training pairs...")
        
        # Merge train pairs with product metadata to see "what attributes were intended"
        enriched_train = train_df.merge(products_df[['p_id', 'norm_brand', 'norm_category']], on='p_id')
        
        brand_counts = defaultdict(lambda: defaultdict(int))
        cat_counts = defaultdict(lambda: defaultdict(int))
        token_total_freq = defaultdict(int)

        for _, row in enriched_train.iterrows():
            tokens = self.preprocessor.tokenize(row['norm_query'])
            brand = row['norm_brand']
            category = row['norm_category']

            for token in tokens:
                token_total_freq[token] += 1
                if brand:
                    brand_counts[token][brand] += 1
                if category:
                    cat_counts[token][category] += 1

        # Calculate probabilities and pick the most likely intent
        for token, total in token_total_freq.items():
            if total < self.min_freq:
                continue
            
            # Find most frequent brand for this token
            if token in brand_counts:
                best_brand = max(brand_counts[token].items(), key=lambda x: x[1])
                # If a token points to a brand > 50% of the time, we map it
                if (best_brand[1] / total) > 0.5:
                    self.token_to_brand[token] = best_brand[0]

            # Find most frequent category for this token
            if token in cat_counts:
                best_cat = max(cat_counts[token].items(), key=lambda x: x[1])
                if (best_cat[1] / total) > 0.5:
                    self.token_to_category[token] = best_cat[0]

        print(f"Intent Dictionary Built. Mapped {len(self.token_to_brand)} brands and {len(self.token_to_category)} categories.")

    def get_intent(self, token):
        """Returns inferred brand and category for a single token."""
        return {
            'brand': self.token_to_brand.get(token, None),
            'category': self.token_to_category.get(token, None)
        }