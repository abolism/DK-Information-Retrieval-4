import pandas as pd
from text_processor import PersianPreprocessor

class KnowledgeBase:
    def __init__(self):
        self.preprocessor = PersianPreprocessor()
        self.brands = set()
        self.product_types = set()
        self.categories = set()
        
    def build(self, products_df):
        """
        Extracts entities from the product catalog.
        
        Args:
            products_df (pd.DataFrame): The dataframe containing product data.
        """
        print("Building Knowledge Base...")
        
        # 1. Extract Brands
        # We filter out empty brands and normalize them
        raw_brands = products_df['brand'].dropna().unique()
        for b in raw_brands:
            norm_b = self.preprocessor.normalize(str(b))
            if len(norm_b) > 1: # Ignore single char brands as noise
                self.brands.add(norm_b)
                
        # 2. Extract Categories and Product Types
        # Assuming category structure might be like "Main > Sub > Leaf" or similar
        # We will split by common delimiters to capture all granularities
        raw_cats = products_df['category'].dropna().unique()
        
        for c in raw_cats:
            # Normalize the full string first to handle delimiters cleanly if needed, 
            # but splitting raw first is usually safer for delimiters.
            # We assume standard delimiters found in datasets: '>', '|', '-'
            # We replace them all with a standard splitter
            clean_c = str(c).replace('|', '>').replace('-', '>')
            parts = [p.strip() for p in clean_c.split('>')]
            
            for i, part in enumerate(parts):
                norm_part = self.preprocessor.normalize(part)
                if len(norm_part) < 2: 
                    continue
                
                # Add to general categories
                self.categories.add(norm_part)
                
                # Heuristic: The last part of a category string is often the "Product Type"
                # e.g., "Digital Goods > Mobile > Samsung Phone" -> "Samsung Phone" is product type
                if i == len(parts) - 1:
                    self.product_types.add(norm_part)
                    
        print(f"Knowledge Base Built: {len(self.brands)} Brands, "
              f"{len(self.product_types)} Product Types, "
              f"{len(self.categories)} Total Category Tokens.")

    def is_brand(self, token):
        return token in self.brands

    def is_product_type(self, token):
        return token in self.product_types
    
    def is_category(self, token):
        return token in self.categories

    def get_brands(self):
        return self.brands

    def get_product_types(self):
        return self.product_types