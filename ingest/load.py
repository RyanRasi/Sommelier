import pandas as pd

def preview_data(df):
    # Basic shape and columns
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSample row:")
    print(df.iloc[0])

    # Check for nulls
    print("\nNull counts:")
    print(df.isnull().sum())

    # Check key columns we'll use
    key_cols = ['description', 'country', 'variety', 'price', 'points', 'title', 'winery']
    print("\nKey columns preview:")
    print(df[key_cols].head(3))

    # Must-pass checks
    assert df.shape[0] > 100_000, "Expected ~130k rows"
    assert 'description' in df.columns, "Missing description column"
    assert df['description'].isnull().sum() == 0, "Descriptions have nulls — problem!"
    assert 'country' in df.columns
    assert 'variety' in df.columns

    print("✅ All checks passed.")