import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Dental X-ray Disease Analysis App",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed")

CSS = """
iframe {
    width: 100%;
    height: 700px;
}
"""

# Main page content
st.title("The Dental X-Ray Disease Analysis")
st.write("")

st.markdown("##### This app uses advanced machine learning to detect *dental diseases, treatments, anatomical structures* from x-ray images, making diagnosis faster and more accessible.")

# SPACER

st.divider()


img_path = "frontend/images/confusion_matrix.png"
image = Image.open(img_path)  # <-- Add this line
new_size = (920, 720)
resized_image = image.resize(new_size)
st.write("")

url_kaggle = "https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset"
url_gcs = "https://storage.googleapis.com/disease-recognition/"
url_docs = "https://dental-xray.github.io/dental-xray/"
url_backend = "https://dental-xray-858779445866.europe-west1.run.app/"
url_backend_docs = url_backend + "docs"

col1, col2, col3 = st.columns([1, 6, 1])  # Wider center column for centering
with col2:

    st.markdown("""
    **Main Page**: Explore key statistics, visualizations, and a confusion matrix that shows how well the model distinguishes different conditions. Also links for MLflow, GCS bucket and Dataset for detailed information.

    **Prediction Page**: Upload your own dental x-ray to receive an instant prediction, along with the model’s confidence level.

    **Conditions Page**: Browse the most common dental diseases detected by the app, see real examples, and access links to medical resources for further explanation.
    """)
    st.divider()
    st.image(resized_image, caption='Confusion matrix of our model')
    st.divider()

    st.markdown("#### Resources:")
    st.write("")
    st.markdown("**Source Code**: [Github Pages](%s)" % url_docs)
    st.write("")
    st.markdown("**Shared Library Docs**: [Github Pages](%s)" % url_docs)
    st.write("")
    st.markdown("**Dataset**: [Dental Xray Panoramic Dataset on Kaggle](%s)" % url_kaggle)
    st.write("")
    st.markdown("**Model Storage**: [Google Cloud Storage](%s)" % url_gcs)
    st.write("")
    st.markdown("**Backend Endpoint**: [Google Cloud Run](%s)" % url_backend)
    st.write("")
    st.markdown("**Backend API Reference**: [Swagger](%s)" % url_backend_docs)
