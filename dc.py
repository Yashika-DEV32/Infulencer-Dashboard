import pandas as pd
import numpy as np

# Load the Excel file
df = pd.read_excel("Book1.xlsx", sheet_name="Sheet1")

# Filter Instagram influencers
df = df[df["Platform Used"].str.contains("Instagram", case=False, na=False)]

# Clean Follower Count
def parse_followers(value):
    if isinstance(value, str):
        value = value.strip("~+").upper().replace(",", "").strip()
        try:
            if 'M' in value:
                return float(value.replace('M', '')) * 1_000_000
            elif 'K' in value:
                return float(value.replace('K', '')) * 1_000
            else:
                return float(value)
        except ValueError:
            return np.nan
    return value

df["Follower Count"] = df["Follower Count"].apply(parse_followers)

# Clean Engagement Rate
def parse_engagement_rate(rate):
    if isinstance(rate, str):
        rate = rate.strip("~%").strip()
        try:
            return float(rate)
        except:
            return np.nan
    return rate

df["Engagement Rate"] = df["Engagement Rate"].apply(parse_engagement_rate)

# Drop rows with missing values
df_cleaned = df.dropna(subset=["Follower Count", "Engagement Rate"]).reset_index(drop=True)

# View cleaned data
print(df_cleaned.head())
