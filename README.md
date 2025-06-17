# 🌍 Jatayuv AI – Personalized Travel Recommender

Jatayuv AI is a smart travel recommendation system that helps users discover ideal destinations, plan itineraries, and estimate budgets based on their preferences, interests, and style — powered by machine learning and Streamlit.

---

## 💡 Problem Statement

In today's world, travelers are overwhelmed by endless choices and generic trip packages. 
People want **personalized experiences** tailored to their:

- Interests (adventure, food, culture, etc.)
- Travel style (explorer, relaxed, cultural, etc.)
- Budget, duration, and climate preferences

**Jatayuv AI** solves this by offering a smart, interactive recommender that suggests:

✅ Best-matching destinations  
✅ Day-wise itinerary  
✅ Budget estimation  
✅ Real-time evaluation of recommendations

---

## 📂 Dataset Overview

The dataset was prepared from mock real-world travel platforms and contains ~500 user profiles and destinations.

It includes:
- `UserID`, `Age`, `Gender`, `Budget`, `Duration`
- `Interests`, `TravelStyle`, `PreferredClimate`
- `DestinationID`, `placeName`, `State`, `Type`, `BestTimeToVisit`, `Popularity`
- Ratings and reviews (for optional NLP)
- Generated output fields like `Recommended Destinations` and `Itinerary Plan`

---

## 🔍 ML Pipeline Overview

### 🔧 Preprocessing
- Data cleaning and transformation
- Feature engineering: combining interests + climate + style

### 🎯 Content-Based Recommendation
- Using **TF-IDF Vectorization** + **Cosine Similarity** on user vs destination profiles
- Returns top 3 most relevant destinations

### 🧠 Clustering (KMeans)
- Users grouped by Age and Budget
- Evaluated with **Silhouette Score** and visualized using scatter plots

### 🗓️ Itinerary Generator
- Rule-based generator selects activities based on travel style
- Calculates estimated cost using travel duration and group size

---

## 📊 Evaluation Metrics

| Metric          | Description                              |
|-----------------|------------------------------------------|
| Precision@3     | How many top 3 destinations matched interests  
| Recall@3        | How well user interests were covered     
| MAE             | Cost prediction error (₹)                
| Silhouette Score| Cluster separation quality               

---

## 🧪 Demo

🔗 [Live Streamlit App](https://jatayuv-ai-project-vo6nw3nxmallzusynipkbk.streamlit.app/)  

## 🚀 Tech Stack

- Python 🐍
- Streamlit
- scikit-learn
- pandas
- matplotlib

---

## 🙌 Acknowledgment

This project was built as part of an internship assignment for **Jatayuv AI** and demonstrates skills in data science, ML, recommender systems, and full-stack AI app deployment.

---

## 📬 Contact

**Vinay Kumar Gorli**  
📍 Srikakulam | CSE (AI & ML) Graduate  
📧 [gorlivinaykumar9876@gmail.com] | 💼 🔗 [**LinkedIn – Vinay Kumar Gorli**](https://www.linkedin.com/in/vinay-kumar-gorli-263634266)


