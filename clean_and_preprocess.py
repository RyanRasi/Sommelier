import pandas as pd

# Drop nulls and low-value rows
def drop_nulls(df):
    # Keep only rows where the fields we care about are present
    df = df.dropna(subset=['description', 'country', 'variety'])

    # Drop duplicate descriptions (some wines are reviewed twice)
    df = df.drop_duplicates(subset=['description'])

    # Reset index cleanly
    df = df.reset_index(drop=True)

    print("Shape after cleaning:", df.shape)
    return df

# Fill remaining nulls in soft fields
def fill_remaining_nulls(df):
    df['price'] = df['price'].fillna(0.0)
    df['province'] = df['province'].fillna('Unknown')
    df['region_1'] = df['region_1'].fillna('Unknown')
    df['designation'] = df['designation'].fillna('Unknown')

    print("Nulls remaining:")
    print(df.isnull().sum())
    return df

# Build the rich structured text & description field 
def build_rich_text(row):
    parts = [
        f"Variety: {row['variety']}",
        f"Country: {row['country']}",
        f"Province: {row['province']}",
        f"Price: ${row['price']}",
        f"Points: {row['points']}",
        f"Description: {row['description']}"
    ]
    return " | ".join(parts)

# Load fresh from saved file to confirm it round-trips correctly
def validate(filename):
    df_check = pd.read_csv(f"{filename}.csv")

    assert df_check['description'].isnull().sum() == 0
    assert df_check['country'].isnull().sum() == 0
    assert df_check['variety'].isnull().sum() == 0
    assert 'text' in df_check.columns
    assert df_check['text'].iloc[0].startswith("Variety:")
    assert len(df_check) > 100_000

    print(f"✅ All checks passed. {len(df_check)} clean wines ready.")


df = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
df = drop_nulls(df)
df = fill_remaining_nulls(df)

df['text'] = df.apply(build_rich_text, axis=1)

# Preview
print("\nSample text field:")
print(df['text'].iloc[0])

# Save cleaned dataset
filename = "wines_clean"
df.to_csv(f"{filename}.csv", index=False)
print(f"\n✅ Saved {filename}.csv with {len(df)} rows.")

validate(filename)