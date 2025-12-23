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
        """Safely formats PID to integer string."""
        try:
            return str(int(float(str(pid))))
        except:
            return str(pid)

    def generate_predictions(self, queries_df, output_path='submission.csv', top_k_candidates=100):
        results = []
        print(f"Starting Inference on {len(queries_df)} queries...")
        
        # Fallback PID (e.g. a generic valid PID if absolute zero results found)
        # Using '0' or a known popular PID is safer than crashing
        fallback_pid = '100000' 
        
        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
            query_text = row['query']
            
            # 1. Retrieval
            candidates = self.retriever.retrieve(query_text, top_k=top_k_candidates)
            candidate_pids = [self.format_pid(c['p_id']) for c in candidates]
            
            final_top_10 = []

            # 2. Ranking (if we have candidates)
            if candidate_pids:
                feat_list = []
                valid_pids = []
                
                for pid in candidate_pids:
                    feats = self.extractor.extract_features(query_text, pid)
                    if feats:
                        feat_list.append(feats)
                        valid_pids.append(pid)
                
                if feat_list:
                    X_pred = pd.DataFrame(feat_list)
                    scores = self.ranker.predict(X_pred)
                    ranked = sorted(zip(valid_pids, scores), key=lambda x: x[1], reverse=True)
                    final_top_10 = [x[0] for x in ranked[:10]]
                else:
                    final_top_10 = candidate_pids[:10]
            
            # 3. PADDING (Crucial for perfect CSV)
            # Fill from original candidates if ranker filtered too many
            if len(final_top_10) < 10:
                remaining = [p for p in candidate_pids if p not in final_top_10]
                final_top_10.extend(remaining)
            
            # Fill with fallback if still empty
            if len(final_top_10) < 10:
                final_top_10.extend([fallback_pid] * (10 - len(final_top_10)))
                
            # Truncate strictly to 10
            final_top_10 = final_top_10[:10]
            
            results.append([query_text] + final_top_10)
            
        print("Saving Results...")
        cols = ['query'] + [f'pid{i+1}' for i in range(10)]
        df = pd.DataFrame(results, columns=cols)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")