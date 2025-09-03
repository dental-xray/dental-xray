import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Main page",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed")

CSS = """
iframe {
    width: 100%;
    height: 700px;
}
"""

st.sidebar.markdown("# 🦷 Main page ")

# Main page content
st.title(" The Dental Disease  Recognition App")
st.write("")

st.markdown("### This app uses advanced machine learning to detect *dental diseases, treatments, anatomical structures* from x-ray images, making diagnosis faster and more accessible.")

# SPACER

st.divider()


img_path = "images/Screenshot 2025-09-01 170022.png"
image = Image.open(img_path)  # <-- Add this line
new_size = (920, 720)
resized_image = image.resize(new_size)
st.write("")


url_mlflow = "https://mlflow.nasebanal.com/#/experiments/278657606559884997?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D"
url_kaggle = "https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset"
url_gcs = "https://storage.googleapis.com/disease-recognition/"

col1, col2, col3 = st.columns([1, 6, 1])  # Wider center column for centering
with col2:

    st.markdown("""
    ##### 🦷 **Main Page**: Explore key statistics, visualizations, and a confusion matrix that shows how well the model distinguishes different conditions. Also links for MLflow, GCS bucket and Dataset for detailed information.
    ######
    ##### 🦷 **Prediction Page**: Upload your own dental x-ray to receive an instant prediction, along with the model’s confidence level.
    ######
    ##### 🦷 **Conditions Page**: Browse the most common dental diseases detected by the app, see real examples, and access links to medical resources for further explanation.
    """)
    st.divider()
    st.image(resized_image, caption='Confusion matrix for best model')
    st.divider()

    st.markdown("## Check out these links for more information:")
    st.write("")
    st.markdown("#### - [link](%s)" % url_mlflow + " for MLflow setup")
    st.write("")
    st.markdown("#### - [link](%s)" % url_kaggle + " for Dataset used to train the model")
    st.write("")
    st.markdown("#### - [link](%s)" % url_gcs + " for GCS bucket")
