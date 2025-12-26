import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# page config
st.set_page_config("Multiple Linear Regression", layout="centered")

# load css
def load_css(filename):
    base_path = os.path.dirname(__file__)
    css_path = os.path.join(base_path, filename)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# title
st.markdown("""
<div class="card">
<h1>Multiple Linear Regression</h1>
<p>Predict <b>tip amount</b> using <b>total bill</b> and <b>party size</b></p>
</div>
""", unsafe_allow_html=True)

# load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# prepare data (MULTIPLE FEATURES)
X = df[["total_bill", "size"]]
y = df["tip"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)

# visualization (scatter only)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")
fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# coefficients
st.markdown(f"""
<div class="card">
<h3>Model Coefficients</h3>
<p>
<b>Total Bill Coefficient:</b> {model.coef_[0]:.3f}<br>
<b>Size Coefficient:</b> {model.coef_[1]:.3f}<br>
<b>Intercept:</b> {model.intercept_:.3f}
</p>
</div>
""", unsafe_allow_html=True)

# prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

size = st.slider("Party Size", 1, 6, 2)

tip = model.predict(
    scaler.transform([[bill, size]])
)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
