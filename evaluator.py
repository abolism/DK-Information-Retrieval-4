import pandas as pd
import numpy as np
from tqdm import tqdm

class SearchEvaluator:
    def __init__(self, retriever, extractor, ranker):
        self.retriever = retriever
        self.extractor = extractor
        self.ranker = ranker

    def evaluate(self, query_product_pairs, top_k_retrieval=100):
        results = []
        p1_list = []
        p10_list = []
        
        print(f"Evaluating {len(query_product_pairs)} queries...")
        
        for _, row in tqdm(query_product_pairs.iterrows(), total=len(query_product_pairs)):
            query = row['query']
            true_pid = row['p_id']
            
            # 1. Retrieval Phase
            candidates = self.retriever.retrieve(query, top_k=top_k_retrieval)
            candidate_pids = [c['p_id'] for c in candidates]
            
            if not candidate_pids:
                p1_list.append(0)
                p10_list.append(0)
                continue

            # 2. Ranking Phase
            # Extract features for all candidates
            feat_list = []
            for pid in candidate_pids:
                feat = self.extractor.extract_features(query, pid)
                feat_list.append(feat)
            
            X_test = pd.DataFrame(feat_list)
            scores = self.ranker.predict(X_test)
            
            # Combine and sort by model score
            scored_candidates = sorted(zip(candidate_pids, scores), key=lambda x: x[1], reverse=True)
            top_10_pids = [x[0] for x in scored_candidates[:10]]
            
            # 3. Calculate Metrics
            # Precision@1
            p1 = 1.0 if top_10_pids[0] == true_pid else 0.0
            # Precision@10 (Boolean in this specific contest context: is it in the top 10?)
            p10 = 1.0 if true_pid in top_10_pids else 0.0
            
            p1_list.append(p1)
            p10_list.append(p10)
            
        metrics = {
            'Mean_Precision@1': np.mean(p1_list),
            'Mean_Precision@10': np.mean(p10_list)
        }
        return metrics