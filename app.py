import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("complete_travel_dataset.csv")

df = load_data()

st.title("🌍 Jatayuv AI - Travel Recommendation (Lite)")

# --- User Input Form ---
with st.form("input_form"):
    name = st.text_input("Your Name")
    age = st.number_input("Age", min_value=10, max_value=100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    budget = st.number_input("Your Budget (₹)", min_value=1000)
    duration = st.slider("Trip Duration (days)", 1, 15, key="duration_input")
    group_size = st.slider("Group Size", 1, 10, key="group_size_input")
    interests = st.multiselect("Your Interests", ["Food", "Adventure", "Wellness", "Culture", "Nature", "Spiritual"])
    travel_style = st.selectbox("Travel Style", ["Explorer", "Relaxed", "Urban", "Cultural"])
    climate = st.selectbox("Preferred Climate", ["Cool", "Warm", "Moderate"])
    submit = st.form_submit_button("Get Recommendations")

if submit:
    # Step 1: Build user & destination profiles
    user_profile = " ".join(interests) + " " + travel_style + " " + climate
    df["UserProfile"] = user_profile
    df["DestinationProfile"] = df["Type"] + " " + df["BestTimeToVisit"]

    # Step 2: Filter destinations by climate and budget
    filtered_df = df[
        (df["PreferredClimate"].str.lower() == climate.lower()) &
        (df["Budget"] <= budget)
    ].copy()

    if filtered_df.empty:
        st.warning("No destinations found based on your budget and climate. Showing general options.")
        filtered_df = df.copy()

    # Step 3: TF-IDF Similarity
    combined = pd.concat([filtered_df["UserProfile"], filtered_df["DestinationProfile"]], ignore_index=True)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(combined)

    user_vec = tfidf[0:len(filtered_df)]
    dest_vec = tfidf[len(filtered_df):]

    scores = cosine_similarity(user_vec, dest_vec)
    top_indices = scores[0].argsort()[-3:][::-1]

    top_destinations = filtered_df.iloc[top_indices][["DestinationID", "Name_x", "State", "Type"]] \
        if "Name_x" in filtered_df.columns else filtered_df.iloc[top_indices][["DestinationID", "State", "Type"]]

    st.subheader("🎯 Top 3 Travel Recommendations:")
    st.dataframe(top_destinations)

    # Step 4: Generate Itinerary & Cost
    activity_map = {
        "Explorer": ["Trekking", "Nature Walk", "Local Adventure"],
        "Relaxed": ["Beach Walk", "Spa Session", "Leisure Shopping"],
        "Urban": ["City Tour", "Café Hopping", "Nightlife"],
        "Cultural": ["Temple Visit", "Museum Tour", "Folk Performance"]
    }

    cost_map = {
        "Explorer": 2000,
        "Relaxed": 2500,
        "Urban": 3000,
        "Cultural": 1800
    }

    if not top_destinations.empty:
        selected_place = top_destinations.iloc[0]["Name_x"] \
            if "Name_x" in top_destinations.columns else top_destinations.iloc[0][0]

        activity_list = activity_map.get(travel_style, ["Local Exploration"])
        activities = [random.choice(activity_list) for _ in range(duration)]

        cost = duration * cost_map.get(travel_style, 2000) * group_size

        st.subheader(f"🗓️ Your {duration}-Day Itinerary for {selected_place}")
        for i, activity in enumerate(activities, 1):
            st.write(f"Day {i}: {activity}")

        st.success(f"💰 Estimated Cost for {group_size} people: ₹{cost}")

        # Step 5: Download as CSV
        import io
        download_df = top_destinations.copy()
        download_df["UserName"] = name
        download_df["Interests"] = ", ".join(interests)
        download_df["TravelStyle"] = travel_style
        download_df["DurationDays"] = duration
        download_df["EstimatedCost"] = cost

        csv = download_df.to_csv(index=False)
        st.download_button("📥 Download Recommendations as CSV", data=csv, file_name="travel_recommendations.csv", mime="text/csv")
