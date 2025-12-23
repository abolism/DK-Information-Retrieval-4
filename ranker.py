# import lightgbm as lgb
# import pandas as pd
# import numpy as np
# import pickle
# import os
# from sklearn.model_selection import train_test_split

# class PointwiseRanker:
#     def __init__(self, model_path='lgbm_ranker.pkl'):
#         self.model_path = model_path
#         self.model = None
#         self.feature_names = None

#     def train(self, X_df, y_series):
#         """
#         Trains the LightGBM model.
#         """
#         print("Preparing Data for Training...")
#         self.feature_names = X_df.columns.tolist()
        
#         # Split into Train/Validation (80/20) to monitor overfitting
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
#         )
        
#         # Initialize LightGBM Classifier
#         # We use 'binary' objective for Pointwise ranking (Relevant vs Not Relevant)
#         self.model = lgb.LGBMClassifier(
#             objective='binary',
#             metric='binary_logloss',
#             n_estimators=1000,      # High cap, will stop early
#             learning_rate=0.05,     # Lower LR for better generalization
#             num_leaves=31,          # Standard complexity
#             max_depth=-1,
#             subsample=0.8,          # Row sampling to prevent overfitting
#             colsample_bytree=0.8,   # Feature sampling
#             random_state=42,
#             n_jobs=-1               # Use all CPU cores
#         )
        
#         print("Training LightGBM Model...")
#         self.model.fit(
#             X_train, y_train,
#             eval_set=[(X_val, y_val)],
#             # early_stopping_rounds is deprecated in newer sklearn API, 
#             # but usually handled via callbacks or explicit args. 
#             # We use the standard fit with eval_set.
#             callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
#         )
        
#         print("Training Complete.")
#         self.save_model()

#     def predict(self, features_df):
#         """
#         Returns probability scores for a dataframe of features.
#         """
#         if self.model is None:
#             self.load_model()
            
#         # Return probability of class 1 (Relevant)
#         return self.model.predict_proba(features_df)[:, 1]

#     def get_feature_importance(self):
#         if self.model is None:
#             return {}
        
#         importances = self.model.feature_importances_
#         return dict(zip(self.feature_names, importances))

#     def save_model(self):
#         with open(self.model_path, 'wb') as f:
#             pickle.dump({'model': self.model, 'features': self.feature_names}, f)
#         print(f"Model saved to {self.model_path}")

#     def load_model(self):
#         if os.path.exists(self.model_path):
#             with open(self.model_path, 'rb') as f:
#                 data = pickle.load(f)
#                 self.model = data['model']
#                 self.feature_names = data['features']
#         else:
#             raise FileNotFoundError("Model file not found. Train first.")


import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
import os

class PointwiseRanker:
    def __init__(self, model_path='lgbm_ranker.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_names = None

    def train(self, df):
        print("Preparing Data for LambdaRank...")
        
        # 1. Sort by QID (Required for grouping)
        df = df.sort_values('qid')
        
        # 2. Split by Query ID (Not by row!) to keep groups intact
        qids = df['qid'].unique()
        np.random.seed(42)
        np.random.shuffle(qids)
        
        split_idx = int(len(qids) * 0.8)
        train_qids = set(qids[:split_idx])
        
        train_df = df[df['qid'].isin(train_qids)]
        val_df = df[~df['qid'].isin(train_qids)]
        
        # 3. Create 'Group' arrays (Number of items per query)
        q_train = train_df.groupby('qid')['label'].count().values
        q_val = val_df.groupby('qid')['label'].count().values
        
        # 4. Separate Features
        drop_cols = ['label', 'qid']
        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df['label']
        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df['label']
        
        self.feature_names = X_train.columns.tolist()

        # 5. Initialize LambdaRank
        self.model = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        
        print("Training LambdaRank Model...")
        self.model.fit(
            X_train, y_train,
            group=q_train,
            eval_set=[(X_val, y_val)],
            eval_group=[q_val],
            eval_at=[10],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(50)
            ]
        )
        self.save_model()

    def predict(self, features_df):
        if self.model is None: self.load_model()
        # Ensure input has same columns as training (minus label/qid)
        # Filter strictly to trained features
        valid_feats = features_df[self.feature_names]
        return self.model.predict(valid_feats)

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'features': self.feature_names}, f)

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['features']
    
    def get_feature_importance(self):
        if self.model is None: return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))