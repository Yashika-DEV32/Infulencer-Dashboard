import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("influencer_model.pkl")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_excel("Book1.xlsx", sheet_name="Sheet1")

    # Parse follower count
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
            except:
                return np.nan
        return value

    df["Follower Count"] = df["Follower Count"].apply(parse_followers)

    # Parse engagement rate
    def parse_engagement_rate(rate):
        if isinstance(rate, str):
            rate = rate.strip("~%").strip()
            try:
                return float(rate)
            except:
                return np.nan
        return rate

    df["Engagement Rate"] = df["Engagement Rate"].apply(parse_engagement_rate)

    # Drop missing required values
    df = df.dropna(subset=["Follower Count", "Engagement Rate", "Category", "Country", "Platform Used"])
    
    return df

df = load_data()

# 🎯 Clean Platform Dropdown (only major ones)
def extract_platforms(df):
    major_platforms = ["Instagram", "YouTube", "TikTok", "Twitter"]
    found_platforms = set()

    for value in df["Platform Used"]:
        for platform in major_platforms:
            if platform.lower() in value.lower():
                found_platforms.add(platform)
    
    return ["All"] + sorted(found_platforms)

platform_options = extract_platforms(df)

# --- Streamlit UI ---
st.title("📊 Influencer Recommendation Dashboard")

with st.form("input_form"):
    platform = st.selectbox("Platform", platform_options)
    category = st.selectbox("Product Category", sorted(df["Category"].unique()))
    country = st.selectbox("Target Country", sorted(df["Country"].unique()))
    budget = st.slider("Estimated Promotion Budget ($)", 100, 100000, 5000, step=500)
    submitted = st.form_submit_button("Find Best Influencers")

if submitted:
    df_filtered = df.copy()
    
    # Apply platform filter if not All
    if platform != "All":
        df_filtered = df_filtered[df_filtered["Platform Used"].str.contains(platform, case=False, na=False)]

    # Apply category & country filter
    df_filtered = df_filtered[(df_filtered["Category"] == category) & (df_filtered["Country"] == country)]

    if df_filtered.empty:
        st.warning("No influencers found for the selected filters.")
    else:
        # Prepare data and predict score
        X = df_filtered[["Follower Count", "Engagement Rate", "Category", "Country", "Platform Used"]]
        df_filtered["Score"] = model.predict(X)

        # Estimate cost
        df_filtered["Estimated Cost"] = (df_filtered["Follower Count"] / 1000) * 10

        # Budget filter
        df_budgeted = df_filtered[df_filtered["Estimated Cost"] <= budget]

        if df_budgeted.empty:
            st.warning("No influencers within budget. Showing top 3 regardless of cost.")
            top3 = df_filtered.sort_values("Score", ascending=False).head(3)
        else:
            top3 = df_budgeted.sort_values("Score", ascending=False).head(3)

        # Show top influencers
        st.subheader("🎯 Top Recommended Influencers")
        st.dataframe(top3[[
            "Influencer Name", "Platform Used", "Follower Count", "Engagement Rate",
            "Category", "Country", "Score", "Estimated Cost"
        ]])
        st.success("Recommendation complete ✅")
