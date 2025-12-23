# import pandas as pd
# import matplotlib.pyplot as plt
# from ranker import PointwiseRanker

# def run_step_9():
#     # 1. Load Training Data (Created in Step 8)
#     try:
#         data = pd.read_csv('ltr_training_data.csv')
#     except FileNotFoundError:
#         print("Training data not found. Please run Step 8.")
#         return

#     # Separate Features and Labels
#     X = data.drop(columns=['label'])
#     y = data['label']
    
#     # 2. Train Model
#     ranker = PointwiseRanker()
#     ranker.train(X, y)
    
#     # 3. Analyze Feature Importance
#     print("\n--- Feature Importance (Gain) ---")
#     importance = ranker.get_feature_importance()
    
#     # Sort and Print
#     sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
#     for feat, score in sorted_imp:
#         print(f"{feat:25}: {score}")

#     # Visualization (Optional textual representation)
#     top_feature = sorted_imp[0][0]
#     print(f"\nInsight: The most influential feature is '{top_feature}'.")
#     if 'bm25_score' in top_feature:
#         print("This confirms that Keyword Matching is the foundation.")
#     elif 'brand' in top_feature:
#         print("This confirms that Entity Matching is the foundation.")

# if __name__ == "__main__":
#     run_step_9()

import pandas as pd
from ranker import PointwiseRanker

def run_step_9():
    try:
        print("Loading Training Data...")
        data = pd.read_csv('ltr_training_data.csv')
    except FileNotFoundError:
        print("Training data not found. Please run Step 8.")
        return

    # Initialize Ranker (The class is named PointwiseRanker but logic is now LambdaRank/Listwise)
    ranker = PointwiseRanker()
    
    # Pass the full dataframe (Ranker handles splitting X, y, and qid internally)
    ranker.train(data)
    
    print("\n--- Feature Importance ---")
    imp = ranker.get_feature_importance()
    for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:20}: {v}")

if __name__ == "__main__":
    run_step_9()