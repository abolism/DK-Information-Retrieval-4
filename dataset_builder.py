import pandas as pd
import random
import numpy as np
from tqdm import tqdm

class DatasetBuilder:
    def __init__(self, enhanced_extractor, retriever, products_df):
        self.extractor = enhanced_extractor
        self.retriever = retriever
        # Get list of all PIDs for random sampling backup
        self.all_pids = products_df['p_id'].tolist()
        
    def get_hard_negatives(self, query, positive_pid, count=4):
        """
        Retrieves top items for query and selects those that are NOT the positive_pid.
        """
        # Retrieve slightly more than we need to have a buffer
        candidates = self.retriever.retrieve(query, top_k=count + 5)
        
        negatives = []
        for cand in candidates:
            if cand['p_id'] != positive_pid:
                negatives.append(cand['p_id'])
                if len(negatives) >= count:
                    break
        
        # If we still don't have enough (rare for specific queries), fill with randoms
        while len(negatives) < count:
            rand_pid = random.choice(self.all_pids)
            if rand_pid != positive_pid and rand_pid not in negatives:
                negatives.append(rand_pid)
                
        return negatives

    def build_dataset(self, train_pairs, negative_ratio=4):
        """
        Constructs the full training DataFrame.
        """
        print(f"Building Dataset with 1:{negative_ratio} Positive/Negative ratio...")
        
        X_list = []
        y_list = []
        groups = [] # Useful for LGBM Ranker if we used GroupWise loss, keeping for structure
        
        # Iterate over pairs with a progress bar
        # We use a subset or full set. 10k pairs is fast enough.
        for idx, row in tqdm(train_pairs.iterrows(), total=len(train_pairs)):
            query = row['query'] # Use original query for retrieval/features
            pos_pid = row['p_id']
            
            # 1. Process Positive Example
            pos_feats = self.extractor.extract_features(query, pos_pid)
            if pos_feats:
                X_list.append(pos_feats)
                y_list.append(1)
            
            # 2. Process Negative Examples
            neg_pids = self.get_hard_negatives(query, pos_pid, count=negative_ratio)
            
            for neg_pid in neg_pids:
                neg_feats = self.extractor.extract_features(query, neg_pid)
                if neg_feats:
                    X_list.append(neg_feats)
                    y_list.append(0)
            
            # Record group size (1 pos + N negs)
            groups.append(1 + len(neg_pids))

        # Convert to DataFrame
        X_df = pd.DataFrame(X_list)
        y_series = pd.Series(y_list, name='label')
        
        print(f"Dataset Built. Total Samples: {len(X_df)}")
        return X_df, y_series, groups