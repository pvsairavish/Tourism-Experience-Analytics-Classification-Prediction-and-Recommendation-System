import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem !important;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.4rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .rec-card {
        background-color: #ffffff;
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        color: #1e293b;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        height: 3rem;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================== LOAD ARTIFACTS ======================
@st.cache_resource
def load_artifacts():
    clf_model = joblib.load("saved_models/best_classification_model.pkl")
    reg_model = joblib.load("saved_models/best_regression_model.pkl")
    le_dict = joblib.load("saved_models/label_encoders.pkl")
    feature_cols = joblib.load("saved_models/feature_columns.pkl")
    attraction_features = joblib.load("saved_models/attraction_features.pkl")
    df = pd.read_csv("saved_models/cleaned_tourism_data.csv")
    return clf_model, reg_model, le_dict, feature_cols, attraction_features, df

clf_model, reg_model, le_dict, feature_cols, attraction_features, df = load_artifacts()

# ====================== HEADER ======================
st.markdown('<p class="main-title">✈️ Tourism Experience Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Visit Mode Prediction • Rating Prediction • Personalized Recommendations</p>', unsafe_allow_html=True)
st.markdown("---")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("🔧 User Inputs")

    # Continent
    continents = sorted(df['Continent'].dropna().unique())
    continent = st.selectbox("🌍 Continent", continents)

    # Region
    filtered_regions = df[df['Continent'] == continent]['Region'].dropna().unique()
    region = st.selectbox("🗺️ Region", sorted(filtered_regions))

    # Country
    filtered_countries = df[(df['Continent'] == continent) & 
                            (df['Region'] == region)]['Country'].dropna().unique()
    country = st.selectbox("🏳️ Country", sorted(filtered_countries))

    # City
    filtered_cities = df[(df['Continent'] == continent) & 
                         (df['Region'] == region) & 
                         (df['Country'] == country)]['CityName'].dropna().unique()
    city = st.selectbox("🏙️ City / Town", sorted(filtered_cities))

    # Attraction Type
    attraction_types = sorted(df['AttractionType'].dropna().unique())
    attraction_type = st.selectbox("🏛️ Preferred Attraction Type", attraction_types)

    st.markdown("### 📅 Visit Details")

    # Calendar - Wide range (2010 to 2035)
    selected_date = st.date_input(
        "Select Visit Date",
        value=pd.to_datetime("2025-06-15"),
        min_value=pd.to_datetime("2010-01-01"),
        max_value=pd.to_datetime("2035-12-31")
    )

    visit_year = selected_date.year
    visit_month = selected_date.month

    st.caption(f"Selected → Year: **{visit_year}** | Month: **{visit_month}**")

    st.markdown("---")
    predict_btn = st.button("🚀 Generate Predictions")

# ====================== HELPER FUNCTION ======================
def create_input_features(continent, region, country, city, attraction_type, visit_year, visit_month):
    
    cont_enc = le_dict['Continent'].transform([continent])[0]
    reg_enc = le_dict['Region'].transform([region])[0]
    coun_enc = le_dict['Country'].transform([country])[0]
    city_enc = le_dict['CityName'].transform([city])[0]
    type_enc = le_dict['AttractionType'].transform([attraction_type])[0]

    user_avg_rating = df['Rating'].mean()
    attr_avg_rating = df[df['AttractionType'] == attraction_type]['Rating'].mean()
    attraction_popularity = len(df[df['AttractionType'] == attraction_type])

    input_dict = {
        'VisitYear': visit_year,
        'VisitMonth': visit_month,
        'IsPeakSeason': 1 if visit_month in [5, 6, 7, 8, 12] else 0,
        'Continent_encoded': cont_enc,
        'Region_encoded': reg_enc,
        'Country_encoded': coun_enc,
        'CityName_encoded': city_enc,
        'AttractionType_encoded': type_enc,
        'UserAvgRating': user_avg_rating,
        'UserVisitCount': 5,
        'UserRatingStd': 0.8,
        'UserMinRating': 3.0,
        'UserMaxRating': 5.0,
        'UserModeDiversity': 2,
        'UserModeMean': 1.5,
        'UserTypeDiversity': 3,
        'UserAvgMonth': visit_month,
        'UserMonthDiversity': 3,
        'AttractionAvgRating': attr_avg_rating,
        'AttractionPopularity': attraction_popularity,
        'AttractionRatingStd': 0.7,
        'AttractionModeDiversity': 3,
        'AttractionUniqueUsers': 50,
        'AttractionPopularityLog': np.log1p(attraction_popularity),
        'Rating_User_Attr_Diff': user_avg_rating - attr_avg_rating,
        'UserLoyalty': 0.3
    }

    return pd.DataFrame([input_dict])[feature_cols]

# ====================== MAIN LOGIC ======================
if predict_btn:

    input_df = create_input_features(
        continent, region, country, city, attraction_type, visit_year, visit_month
    )

    # Predictions
    visit_mode_encoded = clf_model.predict(input_df)[0]
    visit_mode = le_dict['VisitMode'].inverse_transform([visit_mode_encoded])[0]
    rating_pred = float(np.clip(reg_model.predict(input_df)[0], 1.0, 5.0))

    # Results
    st.subheader("📊 Prediction Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Predicted Visit Mode</h4>
            <h2>{visit_mode}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Predicted Rating</h4>
            <h2>{rating_pred:.2f} / 5.0</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Selected Type</h4>
            <h2>{attraction_type}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Recommendations
    st.subheader("🏞️ Recommended Attractions")

    content_recs = attraction_features[
        attraction_features['AttractionType'] == attraction_type
    ].head(8)

    if len(content_recs) == 0:
        st.warning("No attractions found for this type.")
    else:
        cols = st.columns(2)
        for i, (_, row) in enumerate(content_recs.iterrows()):
            with cols[i % 2]:
                attraction_name = row.get('Attraction', 'Unknown Attraction')
                attr_type = row.get('AttractionType', '')
                city_name = row.get('CityName', 'Unknown City')

                st.markdown(f"""
                <div class="rec-card">
                    <h4>📍 {attraction_name}</h4>
                    <p>
                        <b>Type:</b> {attr_type}<br>
                        <b>City:</b> {city_name}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    st.subheader("📈 Insights for Selected Attraction Type")
    col_a, col_b = st.columns(2)

    with col_a:
        type_data = df[df['AttractionType'] == attraction_type]
        fig1 = px.histogram(type_data, x='Rating', nbins=5,
                            title=f"Rating Distribution - {attraction_type}",
                            color_discrete_sequence=['#3B82F6'])
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        top_attr = (type_data.groupby('Attraction')['Rating']
                    .mean().sort_values(ascending=False).head(8))
        fig2 = px.bar(x=top_attr.values, y=top_attr.index, orientation='h',
                      title=f"Top Rated {attraction_type}",
                      color=top_attr.values, color_continuous_scale='Blues')
        fig2.update_layout(height=350, yaxis_title="", xaxis_title="Average Rating")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 Select Continent → Region → Country → City and click **Generate Predictions**")

    st.subheader("🌍 Overall Tourism Insights")
    c1, c2 = st.columns(2)

    with c1:
        mode_counts = df['VisitMode'].value_counts().reset_index()
        mode_counts.columns = ['VisitMode', 'Count']
        fig = px.pie(mode_counts, values='Count', names='VisitMode',
                     title="Visit Mode Distribution", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cont_counts = df['Continent'].value_counts().reset_index()
        cont_counts.columns = ['Continent', 'Count']
        fig = px.bar(cont_counts, x='Continent', y='Count',
                     title="Users by Continent", color='Count',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# ====================== FOOTER ======================
st.markdown("---")
st.caption("Tourism Experience Analytics • Classification + Regression + Recommendation • Built with Streamlit")