"""
data_preprocessing.py
Load raw data, basic cleaning, save processed data.
"""
import pandas as pd
from pathlib import Path

RAW = Path('data/raw/manufacturing_defect_dataset.csv')
PROCESSED = Path('data/interim/cleaned_data.csv')


def load_data(path=RAW):
    return pd.read_csv(path)

def clean_data(df):
    # basic cleaning: drop duplicates and rows with missing values
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def main():
    if not RAW.exists():
        print(f'Place your raw CSV at {RAW} and re-run.')
        return
    df = load_data()
    df_clean = clean_data(df)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED, index=False)
    print('Processed data saved to', PROCESSED)

if __name__ == '__main__':
    main()