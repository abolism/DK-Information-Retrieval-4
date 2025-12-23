import pandas as pd
df = pd.read_csv('ltr_training_data.csv')   # or whichever LTR CSV you built last
pos = df[df['label']==1]
total_pos = len(pos)
zero_bm25 = pos[pos['bm25_score']==0]
print("Total positives:", total_pos)
print("Positives with bm25_score==0:", len(zero_bm25), f"({len(zero_bm25)/total_pos:.2%})")
# Also token overlap if you have it:
if 'token_overlap' in pos.columns:
    zero_overlap = pos[(pos['bm25_score']==0) & (pos['token_overlap']==0)]
    print("bm25==0 and token_overlap==0:", len(zero_overlap), f"({len(zero_overlap)/total_pos:.2%})")
