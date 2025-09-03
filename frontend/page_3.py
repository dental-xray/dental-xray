import streamlit as st
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import folium
from streamlit_folium import folium_static
from folium.plugins import TagFilterButton
import tempfile
import httpx
import numpy as np
from ultralytics.utils.ops import xyxy2xywh
from PIL import Image

st.set_page_config(
    page_title="Folium Map",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="collapsed")

CSS = """
iframe {
    width: 100%;
    height: 700px;
}
"""

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 2, 2])
with col2:
    st.markdown("# Common conditions")
st.write("")
st.write("")
st.write("")
st.divider()

## FILLING ##
col1, col2 = st.columns([1, 1])
with col1:
    url_root_filling = "https://www.nidcr.nih.gov/health-info/dental-fillings"
    img_path = "frontend/images/Screenshot 2025-09-02 105304.png"
    image = Image.open(img_path)  # <-- Add this line
    new_size = (520, 320)
    resized_image_filling = image.resize(new_size)

    st.image(resized_image_filling,
            caption="filling example"
            )

with col2:
    st.write("## **Filling**")
    st.markdown("##### - A dental procedure to restore a tooth damaged by decay")
    st.markdown("##### - Prevents further progression of caries and restores tooth function")
    st.markdown("##### - [link](%s)" % url_root_filling + " for more information")
st.divider()

## ROOT PIECE ##
col1, col2 = st.columns([1, 1])
with col1:
    url_root_piece = "https://medlineplus.gov/ency/article/001055.htm"
    img_path = "frontend/images/Screenshot 2025-09-02 105345.png"
    image = Image.open(img_path)  # <-- Add this line
    new_size = (520, 320)
    resized_image_rootpiece = image.resize(new_size)

    st.image(resized_image_rootpiece,
            caption="root piece example"
            )

with col2:
    st.write("## **Root Piece**")
    st.markdown("##### - A fragment of a tooth root left in the bone after breakage or incomplete extraction")
    st.markdown("##### - Can lead to infection and usually requires removal")
    st.markdown("##### - [link](%s)" % url_root_piece + " for more information")
st.divider()

## CARIES ##
col1, col2 = st.columns([1, 1])
with col1:
    url_caries = "https://medlineplus.gov/toothdecay.html"
    img_path = "frontend/images/Screenshot 2025-09-02 105405.png"
    image = Image.open(img_path)  # <-- Add this line
    new_size = (520, 320)
    resized_image_caries = image.resize(new_size)

    st.image(resized_image_caries,
            caption="caries example"
            )

with col2:
    st.write("## **Caries**")
    st.markdown("#### - A bacterial disease that causes demineralization of enamel and dentin")
    st.markdown("#### - If untreated, it can lead to pain, infection, and tooth loss")
    st.markdown("#### - [link](%s)" % url_caries + " for more information")
st.divider()

## CROWN ##
col1, col2 = st.columns([1, 1])
with col1:
    url_crown = "https://medlineplus.gov/ency/article/007631.htm"
    img_path = "frontend/images/Screenshot 2025-09-02 105713.png"
    image = Image.open(img_path)  # <-- Add this line
    new_size = (520, 320)
    resized_image_crown = image.resize(new_size)

    st.image(resized_image_crown,
            caption="crown example"
            )

with col2:
    st.write("## **Crown**")
    st.markdown("#### - A fixed prosthetic cap placed over a natural tooth")
    st.markdown("#### - Used for large cavities, after root canal treatment, or for aesthetic purposes")
    st.markdown("#### - [link](%s)" % url_crown + " for more information")
st.divider()

## ROOT CANAL ##
col1, col2 = st.columns([1, 1])
with col1:
    url_rootcanal ="https://medlineplus.gov/ency/article/007275.htm"
    img_path = "frontend/images/Screenshot 2025-09-02 110229.png"
    image = Image.open(img_path)  # <-- Add this line
    new_size = (520, 320)
    resized_image_rootcanal = image.resize(new_size)

    st.image(resized_image_rootcanal,
            caption="Root Canal Treatment example"
            )

with col2:
    st.write("## **Root Canal Treatment**")
    st.markdown("#### - A procedure to remove infected or damaged pulp from root canals")
    st.markdown("#### - Canals are cleaned, disinfected, and filled to prevent reinfection")
    st.markdown("#### - [link](%s)" % url_rootcanal + " for more information")
st.divider()

## IMPACTED TOOTH ##
col1, col2 = st.columns([1, 1])
with col1:
    url_impacted = "https://medlineplus.gov/ency/article/001057.htm"
    img_path = "frontend/images/Screenshot 2025-09-02 112406.png"
    image = Image.open(img_path)  # <-- Add this line
    new_size = (520, 320)
    resized_image_impacted_tooth = image.resize(new_size)

    st.image(resized_image_impacted_tooth,
            caption="Impacted Tooth example"
            )


with col2:
    st.write("## **Impacted Tooth**")
    st.markdown("#### - A tooth that fails to fully erupt and remains partially or completely in the bone")
    st.markdown("#### - Most often affects wisdom teeth and may cause pain, pressure, or inflammation")
    st.markdown("#### - [link](%s)" % url_impacted + " for more information")

st.divider()
