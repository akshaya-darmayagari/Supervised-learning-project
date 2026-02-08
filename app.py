import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Student Regression Dashboard",
                   page_icon="🎓",
                   layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Whole App Background */
.stApp {
    background: linear-gradient(to right, #e0f7fa, #f0fff4);
}

/* Main Title */
h1 {
    text-align: center;
    color: #2c3e50;
    font-weight: 800;
}

/* Subheaders */
h2, h3 {
    color: #34495e;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #fffaf0;
    border-right: 2px solid #e0f2f1;
}

/* Buttons */
.stButton>button {
    background-color: #ffb74d;   /* Light Orange */
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #ffa726;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border-radius: 18px;
    padding: 18px;
    border: 2px solid #b2dfdb;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
}

/* Tabs Styling */
button[role="tab"] {
    font-size: 16px;
    font-weight: 600;
    color: #2c3e50;
}

/* Horizontal Line */
hr {
    border: 1px solid #b2ebf2;
}

</style>
""", unsafe_allow_html=True)



st.title("🎓 Student Performance Regression Dashboard")

# ---------------- SESSION STATE ----------------
if "model" not in st.session_state:
    st.session_state.model = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "rmse" not in st.session_state:
    st.session_state.rmse = None
if "r2" not in st.session_state:
    st.session_state.r2 = None

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_excel("student_performance_100.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ---------------- HANDLE MISSING VALUES ----------------
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📂 Dataset Overview", "🤖 Model Training & Prediction"])

# ===================== TAB 1 =====================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Column Names")
    st.write(list(df.columns))

# ===================== TAB 2 =====================
with tab2:

    # -------- ENCODING --------
    le_dict = {}
    categorical_cols = ["Gender", "Ethnicity",
                        "Parental Education",
                        "Test Preparation"]

    df_encoded = df.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        le_dict[col] = le

    # -------- FEATURES & TARGET --------
    X = df_encoded.drop(["Student ID", "Math Score"], axis=1)
    y = df_encoded["Math Score"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # -------- SIDEBAR SETTINGS --------
    st.sidebar.header("⚙️ Regression Settings")

    model_name = st.sidebar.selectbox(
        "Select Model",
        ["Linear Regression",
         "Decision Tree",
         "Random Forest",
         "SVR",
         "KNN"]
    )

    if model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeRegressor(max_depth=max_depth)

    elif model_name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 300, 100)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    elif model_name == "SVR":
        C = st.sidebar.slider("C Value", 0.1, 10.0, 1.0)
        model = SVR(C=C)

    elif model_name == "KNN":
        k = st.sidebar.slider("Neighbors", 1, 20, 5)
        model = KNeighborsRegressor(n_neighbors=k)

    # -------- TRAIN MODEL --------
    if st.sidebar.button("🚀 Train Model"):

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.trained = True
        st.session_state.rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.session_state.r2 = r2_score(y_test, y_pred)

    # -------- SHOW RESULTS --------
    if st.session_state.trained:

        st.subheader(f"📊 {model_name} Performance")

        col1, col2 = st.columns(2)
        col1.metric("📉 RMSE", round(st.session_state.rmse, 2))
        col2.metric("📈 R² Score", round(st.session_state.r2, 3))

        # -------- PREDICTION --------
        st.subheader("🔮 Predict Math Score")

        gender = st.selectbox("Gender", le_dict["Gender"].classes_)
        ethnicity = st.selectbox("Ethnicity", le_dict["Ethnicity"].classes_)
        parent = st.selectbox("Parental Education",
                              le_dict["Parental Education"].classes_)
        prep = st.selectbox("Test Preparation",
                            le_dict["Test Preparation"].classes_)

        reading = st.slider("Reading Score", 0, 100, 50)
        writing = st.slider("Writing Score", 0, 100, 50)

        if st.button("✨ Predict Score"):

            input_data = np.array([[
                le_dict["Gender"].transform([gender])[0],
                le_dict["Ethnicity"].transform([ethnicity])[0],
                le_dict["Parental Education"].transform([parent])[0],
                le_dict["Test Preparation"].transform([prep])[0],
                reading,
                writing
            ]])

            input_scaled = scaler.transform(input_data)

            prediction = st.session_state.model.predict(input_scaled)[0]

            st.success(f"🎯 Predicted Math Score: {round(prediction, 2)}")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center style='color:white'>
Supervised Learning Regression Project | Built with Streamlit 🎓
</center>
""", unsafe_allow_html=True)
