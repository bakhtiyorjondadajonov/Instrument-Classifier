import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

temp = pathlib.PosixPath
pathlib.PosixPath =  pathlib.WindowsPath if pathlib.WindowsPath else pathlib.PosixPath

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
  
    </style>
   
    <h1 class="centered-title">Instrument Detector</h1>
    <p>This image classification model effectively distinguishes between medical instruments🩺🩻, musical instruments🥁🎸, and kitchen utensils🥣🍽️ using deep learning techniques. 
    </br>
    </br>
    By leveraging the fastai library and the comprehensive Open Images Dataset, the model achieves high accuracy and robustness. 
    </br>
    </br>
    This project demonstrates the power of transfer learning and modern neural network architectures in solving practical image classification tasks.
    </p>
    """, 
    unsafe_allow_html=True
)
st.subheader("How to use:")
st.write("Just upload an image into one of the three categories above, and it will determine which category it matches.")
file=st.file_uploader("Upload your image",["jpg","jpeg","png","gif","svg"])
model=load_learner("tools_classifier.pkl")
if(file):
    img=PILImage.create(file)
    prediction,prediction_id,probability=model.predict(img)
    st.image(file)
    st.success(f'Prediction result: {prediction}')
    prob=f'Probability: {probability[prediction_id]*100:.1f}%'
    st.info(prob)
    fig=px.bar(x=probability*100,y=model.dls.vocab)
    st.plotly_chart(fig)