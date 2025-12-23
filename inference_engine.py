# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# # Update the saving logic in InferenceEngine.generate_predictions:
# def format_pid(pid):
#     # Convert '1234.0' or 1234.0 or '1234' -> '1234'
#     try:
#         return str(int(float(pid)))
#     except:
#         return str(pid)

# class InferenceEngine:
#     def __init__(self, retriever, extractor, ranker):
#         self.retriever = retriever
#         self.extractor = extractor
#         self.ranker = ranker

#     def generate_predictions(self, queries_df, output_path='submission.csv', top_k_candidates=100):
#         """
#         Runs the search pipeline for all queries and saves the CSV.
#         """
#         results = []
        
#         print(f"Starting Inference on {len(queries_df)} queries...")
        
#         # Iterate over test queries
#         for _, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
#             query_text = row['query']
            
#             # 1. Candidate Generation (Recall)
#             candidates = self.retriever.retrieve(query_text, top_k=top_k_candidates)
#             candidate_pids = [c['p_id'] for c in candidates]
            
#             # Handle cases with no results (fallback to popular or random if needed, 
#             # but usually return empty or what we have)
#             if not candidate_pids:
#                 # Fill with 0s or empty strings to maintain format
#                 results.append([query_text] + [0]*10)
#                 continue
                
#             # 2. Feature Extraction (Ranking Prep)
#             feat_list = []
#             valid_pids = []
            
#             for pid in candidate_pids:
#                 # Helper to handle missing features gracefully
#                 feats = self.extractor.extract_features(query_text, pid)
#                 if feats:
#                     feat_list.append(feats)
#                     valid_pids.append(pid)
            
#             if not feat_list:
#                  # Fallback: just return BM25 order if feature extraction fails completely
#                 results.append([query_text] + candidate_pids[:10])
#                 continue

#             # 3. Re-Ranking (Precision)
#             X_pred = pd.DataFrame(feat_list)
            
#             # Predict relevance scores
#             scores = self.ranker.predict(X_pred)
            
#             # Pair PIDs with scores and sort
#             ranked_candidates = sorted(zip(valid_pids, scores), key=lambda x: x[1], reverse=True)
            
#             # Select Top 10
#             # top_10 = [str(x[0]) for x in ranked_candidates[:10]]
#             top_10 = [format_pid(x[0]) for x in ranked_candidates[:10]]
            
#             # Pad with original retrieval if less than 10 (rare)
#             if len(top_10) < 10:
#                 remaining = [str(p) for p in candidate_pids if str(p) not in top_10]
#                 top_10.extend(remaining[:10 - len(top_10)])
                
#             # Format row: [Query Text, PID1, PID2, ..., PID10]
#             results.append([query_text] + top_10)
            
#         # 4. Save to CSV
#         print("Saving Results...")
#         # Columns: query, pid1, pid2, ... pid10
#         cols = ['query'] + [f'pid{i+1}' for i in range(10)]
#         submission_df = pd.DataFrame(results, columns=cols)
        
#         submission_df.to_csv(output_path, index=False)
#         print(f"Submission saved to {output_path}")
#         return submission_df

import pandas as pd
import numpy as np
from tqdm import tqdm

class InferenceEngine:
    def __init__(self, retriever, extractor, ranker):
        self.retriever = retriever
        self.extractor = extractor
        self.ranker = ranker

    def format_pid(self, pid):
        try:
            return str(int(float(str(pid))))
        except:
            return str(pid)

    def generate_predictions(self, queries_df, output_path='submission.csv', top_k_candidates=200): # Increased Recall
        results = []
        # Coefficients based on your suggestion
        ALPHA = 200.0  # Brand
        BETA = 150.0   # Digits (Increased slightly to act as a hard filter)
        THETA = 100.0  # Category
        
        print(f"Starting Ultra-Boost Inference on {len(queries_df)} queries...")
        
        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
            query_text = row['query']
            candidates = self.retriever.retrieve(query_text, top_k=top_k_candidates)
            candidate_pids = [self.format_pid(c['p_id']) for c in candidates]
            
            scored_candidates = []
            for pid in candidate_pids:
                feats = self.extractor.extract_features(query_text, pid)
                if feats:
                    # Start with the ML model's base ranking 
                    # (Usually a small number between -5 and 5)
                    ml_score = self.ranker.predict(pd.DataFrame([feats]))[0]
                    
                    # Apply your coefficients
                    brand_boost = ALPHA if feats.get('brand_match', 0) == 1.0 else 0.0
                    digit_boost = BETA if feats.get('digit_match', 0) == 1.0 else 0.0
                    cat_boost = THETA if feats.get('cat_match', 0) == 1.0 else 0.0
                    
                    # FINAL SCORE FUNCTION
                    final_score = ml_score + brand_boost + digit_boost + cat_boost
                    scored_candidates.append((pid, final_score))
            
            # Sort by the new weighted score
            ranked = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
            final_top_10 = [x[0] for x in ranked[:10]]
            
            # Fallback if extractor failed
            if not final_top_10:
                final_top_10 = candidate_pids[:10]
            
            results.append([query_text] + final_top_10)