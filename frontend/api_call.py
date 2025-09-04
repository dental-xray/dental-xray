import hashlib
import streamlit as st
import httpx

@st.cache_data(show_spinner=False)
def call_api_cached(uploaded_file):
    """
    Send file to API and cache response based on file content hash.
    """
    # Generate a hash for the file so caching works
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    url = "https://dental-recognition-api-best-yolo-702910251809.us-west1.run.app/predict"
    response = httpx.post(url, files=files, timeout=60.0)

    return response.json()

def call_api(uploaded_file):
    """Send file to API and return response JSON."""
    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    url = "https://dental-recognition-api-best-yolo-702910251809.us-west1.run.app/predict"
    response = httpx.post(url, files=files, timeout=60.0)
    return response.json()
