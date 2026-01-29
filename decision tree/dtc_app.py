import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error

st.set_page_config(page_title="Decision Tree App", layout="centered")

st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(to right, #fdfbfb, #ebedee);
}

/* Main title (st.title) */
h1, .stTitle {
    color: #6C63FF !important;   /* Purple */
    text-align: center;
}

/* Headers (st.header) */
h2, .stHeader {
    color: #2ECC71 !important;   /* Green */
}

/* Subheaders (st.subheader) */
h3, .stSubheader {
    color: #E67E22 !important;   /* Orange */
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1E1E2F;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: white;
}

/* Alerts */
.stAlert {
    background-color: #E8F5E9;
    color: #1B5E20;
    border-left: 6px solid #4CAF50;
}

/* DataFrame */
.stDataFrame {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)



st.title("Decision Tree – Classification & Regression")

st.sidebar.header("Choose Problem Type")
option = st.sidebar.selectbox(
    "Select",
    ["Classification (Iris)", "Regression"]
)


if option == "Classification (Iris)":

    st.subheader("Iris Flower Classification")

    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    df = pd.DataFrame(X, columns=feature_names)
    df["Target"] = y
    df["Flower"] = df["Target"].apply(lambda x: target_names[x])

    st.sidebar.header("Model Settings")
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)

    st.sidebar.header("Input Features")
    inputs = []
    for i, feature in enumerate(feature_names):
        value = st.sidebar.slider(
            feature,
            float(X[:, i].min()),
            float(X[:, i].max()),
            float(X[:, i].mean())
        )
        inputs.append(value)

    input_data = np.array([inputs])

    prediction = model.predict(input_data)[0]
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Predicted Flower: **{target_names[prediction]}**")
    st.markdown(f"<h4 style='color:#34495E'>Accuracy: {accuracy*100:.2f}%</h4>", unsafe_allow_html=True)

    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    st.dataframe(cm_df)

    
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        ax=ax
    )
    st.pyplot(fig)

    with st.expander("View Dataset"):
        st.dataframe(df)


else:

    st.subheader("Decision Tree Regression")

    np.random.seed(0)
    X = np.random.rand(100, 1) * 10
    y = 3 * X.squeeze() + np.random.randn(100) * 3

    df = pd.DataFrame({"X": X.squeeze(), "Y": y})

    st.sidebar.header("Model Settings")
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)

    st.sidebar.header("Input")
    x_input = st.sidebar.slider(
        "Enter X value",
        float(X.min()),
        float(X.max()),
        float(X.mean())
    )

    input_data = np.array([[x_input]])

    prediction = model.predict(input_data)[0]
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.success(f"Predicted Output: **{prediction:.2f}**")
    st.markdown(
        f"<h4 style='color:#34495E'>R²: {r2:.2f} | MAE: {mae:.2f}</h4>",
        unsafe_allow_html=True
    )

    
    st.subheader("Actual vs Predicted")
    fig1, ax1 = plt.subplots()
    ax1.scatter(x_test, y_test, label="Actual")
    ax1.scatter(x_test, y_pred, label="Predicted")
    ax1.legend()
    st.pyplot(fig1)

    
    st.subheader("Decision Tree Visualization")
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    plot_tree(model, feature_names=["X"], filled=True, ax=ax2)
    st.pyplot(fig2)

    with st.expander("View Dataset"):
        st.dataframe(df)