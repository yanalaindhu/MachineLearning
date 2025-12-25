import streamlit as st
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#page config#
st.set_page_config("Linear Regression",layout="centered")
#load css
def load_css(filename):
    base_path = os.path.dirname(__file__)   # regression folder
    css_path = os.path.join(base_path, filename)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")
# title#
st.markdown(""" 
<div class="card">
<h1>Linear Regression</h1>
<p>predict<b> tip amount</b> from <b> total bill </b> using linear  regression..</p>
</div>
 """,unsafe_allow_html=True)

#Load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

st.markdown('<div class="card',unsafe_allow_html=True)
st.subheader("dataset preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)

#prepare the data
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#train model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#metrics
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)

#visualisation
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("total bill vs tip")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6)
ax.plot(x, model.predict(scaler.transform(x)), color="red")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

#performance
st.markdown('<div class=card',unsafe_allow_html=True)
st.subheader("model performance")
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R2",f"{r2:.3f}")
c4.metric("adg R2",f"{adj_r2:.3f}")
st.markdown('</div>',unsafe_allow_html=True)

#m & c
st.markdown(f"""
<div class="card">
<h3>Model Intercept & Co-efficient</h3>
<p>
<b>Coefficient:</b> {model.coef_[0]:.3f}<br>
<b>Intercept:</b> {model.intercept_:.3f}
</p>
</div>
""", unsafe_allow_html=True)

#prediction

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill $",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

tip = model.predict(scaler.transform([[bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predict Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

