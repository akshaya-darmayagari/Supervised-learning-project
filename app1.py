import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Iris Classification Dashboard",
    page_icon="🌸",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM LIGHT THEME (Lavender + Pink)
# ---------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #fdfbff, #f3e8ff);
}

/* Main Title */
h1 {
    text-align: center;
    color: #5b21b6;
    font-weight: 800;
}

/* Subheaders */
h2, h3 {
    color: #6b21a8;
}

/* Sidebar - Light Pink */
section[data-testid="stSidebar"] {
    background-color: #ffe4ec;
    border-right: 2px solid #fbcfe8;
}

/* Buttons */
.stButton>button {
    background-color: #c084fc;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #a855f7;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border-radius: 18px;
    padding: 18px;
    border: 2px solid #e9d5ff;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

st.title("🌸 Iris Flower Classification Dashboard")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------------
# SIDEBAR - MODEL SELECTION
# ---------------------------------------------------
st.sidebar.header("⚙ Model Selection")

model_name = st.sidebar.selectbox(
    "Choose Classification Model",
    ["Logistic Regression",
     "Decision Tree",
     "Random Forest",
     "SVM",
     "KNN"]
)

# Hyperparameters
if model_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)

elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)

elif model_name == "KNN":
    n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 15, 5)

elif model_name == "SVM":
    C = st.sidebar.slider("Regularization (C)", 0.1, 10.0, 1.0)

# ---------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------
X = df.drop("species", axis=1)
y = df["species"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# MODEL INITIALIZATION
# ---------------------------------------------------
if model_name == "Logistic Regression":
    model = LogisticRegression()

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=max_depth)

elif model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=n_estimators)

elif model_name == "SVM":
    model = SVC(C=C)

elif model_name == "KNN":
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("Accuracy", round(accuracy, 4))
col2.metric("Model Used", model_name)

# ---------------------------------------------------
# SMALL CONFUSION MATRIX
# ---------------------------------------------------
# ---------------------------------------------------
# ULTRA SMALL CONFUSION MATRIX
# ---------------------------------------------------
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

# Create columns to center small plot
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    fig, ax = plt.subplots(figsize=(1.8, 1.8))  # very small

    ax.imshow(cm)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])

    ax.set_xticklabels(iris.target_names, fontsize=6, rotation=45)
    ax.set_yticklabels(iris.target_names, fontsize=6)

    # Small numbers inside boxes
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=6)

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout(pad=0.3)

    st.pyplot(fig)


# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------
st.sidebar.header("🌼 Predict New Flower")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)

if st.sidebar.button("Predict Species"):

    input_data = np.array([[sepal_length,
                            sepal_width,
                            petal_length,
                            petal_width]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    species_name = iris.target_names[prediction]

    st.subheader("🌺 Prediction Result")

    st.markdown(f"""
    <div style="
        padding:20px;
        background-color:#fce7f3;
        border-left:8px solid #f472b6;
        border-radius:12px;
        font-size:20px;
        font-weight:600;
        color:#7e22ce;">
        🌸 Predicted Species: {species_name}
    </div>
    """, unsafe_allow_html=True)
