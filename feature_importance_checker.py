import pickle
m = pickle.load(open('lgbm_ranker.pkl','rb'))  # adapt path
fi = m.feature_importance(importance_type='gain')
names = m.feature_name()
for n, v in sorted(zip(names, fi), key=lambda x:-x[1])[:20]:
    print(n, v)
