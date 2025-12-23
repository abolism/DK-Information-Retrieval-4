# import pandas as pd
# import os
# from text_processor import PersianPreprocessor

# class DataLoader:
#     def __init__(self, data_path='data'):
#         self.data_path = data_path
#         self.preprocessor = PersianPreprocessor()
#         self.products = None
#         self.train_pairs = None
#         self.test_queries = None

#     def load_data(self):
#         """Loads CSVs and applies preprocessing to text columns."""
#         print("Loading data...")
        
#         # Load Products
#         self.products = pd.read_csv(os.path.join(self.data_path, 'products.csv'))
#         # Fill NaNs with empty strings to avoid errors
#         self.products.fillna('', inplace=True)
        
#         # Apply normalization to searchable fields
#         print("Preprocessing Products...")
#         self.products['norm_title'] = self.products['title'].apply(self.preprocessor.normalize)
#         self.products['norm_brand'] = self.products['brand'].apply(self.preprocessor.normalize)
#         self.products['norm_category'] = self.products['category'].apply(self.preprocessor.normalize)
#         self.products['norm_attributes'] = self.products['attributes'].apply(self.preprocessor.normalize)
#         # Create a combined text field for retrieval
#         self.products['search_text'] = (
#             self.products['norm_title'] + " " + 
#             self.products['norm_brand'] + " " + 
#             self.products['norm_category'] + " " + 
#             self.products['norm_attributes']
#         )

#         # Load Train Pairs
#         print("Preprocessing Train Queries...")
#         self.train_pairs = pd.read_csv(os.path.join(self.data_path, 'train_query_product_pairs.csv'))
#         self.train_pairs['norm_query'] = self.train_pairs['query'].apply(self.preprocessor.normalize)

#         # Load Test Queries
#         print("Preprocessing Test Queries...")
#         self.test_queries = pd.read_csv(os.path.join(self.data_path, 'test_queries.csv'))
#         self.test_queries['norm_query'] = self.test_queries['query'].apply(self.preprocessor.normalize)
        
#         print("Data Loading & Preprocessing Complete.")
        
#     def get_products(self):
#         return self.products

#     def get_train_pairs(self):
#         return self.train_pairs

#     def get_test_queries(self):
#         return self.test_queries


import pandas as pd
import os
from text_processor import PersianPreprocessor

class DataLoader:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.preprocessor = PersianPreprocessor()
        self.products = None
        self.train_pairs = None
        self.test_queries = None

    def load_data(self):
        print("Loading data with strict types...")
        
        # 1. Load Products - FORCE STRING ID
        self.products = pd.read_csv(os.path.join(self.data_path, 'products.csv'), dtype={'p_id': str})
        self.products.fillna('', inplace=True)
        
        print("Preprocessing Products...")
        # Normalize fields
        self.products['norm_title'] = self.products['title'].apply(self.preprocessor.normalize)
        self.products['norm_brand'] = self.products['brand'].apply(self.preprocessor.normalize)
        self.products['norm_category'] = self.products['category'].apply(self.preprocessor.normalize)
        self.products['norm_attributes'] = self.products['attributes'].apply(self.preprocessor.normalize)
        
        # ENRICHED SEARCH TEXT: Includes Normalized Text AND Original Title (for English keywords)
        self.products['search_text'] = (
            self.products['title'] + " " +  # Raw title (English/Finglish friendly)
            self.products['norm_title'] + " " + 
            self.products['norm_brand'] + " " + 
            self.products['norm_category'] + " " + 
            self.products['norm_attributes']
        )

        # 2. Load Train Pairs
        print("Preprocessing Train Queries...")
        self.train_pairs = pd.read_csv(os.path.join(self.data_path, 'train_query_product_pairs.csv'), dtype={'p_id': str})
        self.train_pairs['norm_query'] = self.train_pairs['query'].apply(self.preprocessor.normalize)

        # 3. Load Test Queries
        print("Preprocessing Test Queries...")
        self.test_queries = pd.read_csv(os.path.join(self.data_path, 'test_queries.csv'))
        self.test_queries['norm_query'] = self.test_queries['query'].apply(self.preprocessor.normalize)
        
        print("Data Loading Complete.")
        
    def get_products(self): return self.products
    def get_train_pairs(self): return self.train_pairs
    def get_test_queries(self): return self.test_queries