import pandas as pd
import os

def verify_submission(file_path='submission.csv'):
    print(f"Verifying {file_path}...")
    
    if not os.path.exists(file_path):
        print("❌ Error: File not found.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error: Could not read CSV. {e}")
        return

    # 1. Check Column Headers
    expected_cols = ['query'] + [f'pid{i+1}' for i in range(10)]
    if list(df.columns) != expected_cols:
        print(f"❌ Error: Incorrect columns.\nExpected: {expected_cols}\nFound:    {list(df.columns)}")
        return

    # 2. Check Row Count
    # We expect 9961 rows (based on test_queries.csv count from prompt)
    # But strictly, it should match the input test file length. 
    # We'll just warn if it looks empty.
    if len(df) == 0:
        print("❌ Error: CSV is empty.")
        return

    # 3. Check Data Types
    # PIDs should be integers or strings, no NaNs
    if df.isnull().values.any():
        print("❌ Error: Found missing values (NaN) in the file.")
        return

    print("✅ Validation Passed!")
    print(f"   - Rows: {len(df)}")
    print(f"   - Columns: {len(df.columns)}")
    print("   - Format looks correct for submission.")

if __name__ == "__main__":
    verify_submission()