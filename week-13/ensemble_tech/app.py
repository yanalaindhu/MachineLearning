import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import tree

st.set_page_config(page_title="Decision Tree – Pre and Post Pruning", layout="wide")
st.title("🏠 California Housing – Decision Tree (Pre and Post Pruning)")

file = st.sidebar.file_uploader("Upload california_housing_test.csv", type="csv")

if file:
    df = pd.read_csv(file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = df.drop(columns="median_house_value")
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.sidebar.header("Model Options")
    option = st.sidebar.selectbox(
        "Select Mode",
        ["Pre-Pruning", "Post-Pruning"]
    )

    if option == "Pre-Pruning":
        st.subheader("Pre-Pruning (Before Training)")

        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("R² Score:", r2_score(y_test, y_pred))
        st.write("RMSE:", mean_squared_error(y_test, y_pred))
        st.write("MAE:", mean_absolute_error(y_test, y_pred))

        fig, ax = plt.subplots(figsize=(20,10))
        tree.plot_tree(
            model,
            feature_names=X.columns,
            filled=True,
            max_depth=3,
            ax=ax
        )
        st.pyplot(fig)
    elif option == "Post-Pruning":
        st.subheader("Post-Pruning (Cost Complexity Pruning)")

        base_model = DecisionTreeRegressor(random_state=42)
        path = base_model.cost_complexity_pruning_path(X_train, y_train)

        ccp_alpha = st.sidebar.slider(
            "ccp_alpha",
            float(path.ccp_alphas.min()),
            float(path.ccp_alphas.max()),
            step=0.0005
        )

        model = DecisionTreeRegressor(
            random_state=42,
            ccp_alpha=ccp_alpha
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("R² Score:", r2_score(y_test, y_pred))
        st.write("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

        fig, ax = plt.subplots(figsize=(20,10))
        tree.plot_tree(
            model,
            feature_names=X.columns,
            filled=True,
            max_depth=3,
            ax=ax
        )
        st.pyplot(fig)