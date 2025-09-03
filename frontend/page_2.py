import streamlit as st
from streamlit_image_comparison import image_comparison
from frontend.api_call import call_api, call_api_cached
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import folium
from streamlit_folium import folium_static
from folium.plugins import TagFilterButton
import tempfile
import httpx
import numpy as np
from collections import defaultdict
import pandas as pd


st.markdown("# Prediction Page 🦷")
# SPACER
st.text("")
st.text("")
st.text("")

st.sidebar.markdown("# 🦷Prediction Page")
# SPACER
st.text("")
st.text("")

st.set_page_config(
    page_title="Prediction",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed")

CSS = """
iframe {
    width: 100%;
    height: 700px;
}
"""

# Upload file
uploaded_file = st.file_uploader("Choose a file")

# SPACER
st.text("")
st.text("")


if uploaded_file is None:
    st.markdown("## Download Images Example")

# Load your images
image_paths = ["images/1154870000-jpg_png_jpg.rf.b58bc86b0009ff155104cacef61b4e4f.jpg",
               "images/3623320000-jpg_png_jpg.rf.d4ca918d94910914cba47a574ddbba08.jpg",
               "images/e6067300-Shahmohamadi_Roghayeh_2022-05-14195349_jpg.rf.acd7cb2e09d40c824c524481cb123035.jpg"]

cols = st.columns(len(image_paths) + 4)

# Loop through the images and create download buttons
n=1
for col, image_path in zip(cols, image_paths):
    with col:
        with open(image_path, "rb") as file:
            st.download_button(
                label=f"Download Sample {n}",
                data=file,
                file_name=image_path,
                mime="image/png" if image_path.endswith(".png") else "image/jpeg"
            )
        n += 1


if uploaded_file is not None:

    st.write(f"### File size: {uploaded_file.size} bytes")
    # SPACER
    st.text("")
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
    # Write the content of the UploadedFile to the temporary file
        tmp_file.write(uploaded_file.getvalue())
    # Get the path of the temporary file
        file_path = tmp_file.name

    img = mpimg.imread(uploaded_file)
    height, width = img.shape[:2]

    scale = 1.7
    # Create a Folium map, with unvisible map to add image
    m_2 = folium.Map(
        location=[0, 0],
        zoom_start=0.25,
        crs="Simple",
        tiles=None
    )

    # add the image as background
    bounds = [[0, 0], [int(height * scale), int(width * scale)]]
    folium.raster_layers.ImageOverlay(
        image=img,
        bounds=bounds,
        opacity=1
        ).add_to(m_2)

    m_2.fit_bounds(bounds)

    col1, col2, col3 = st.columns([1, 6, 1])  # Wider center column for centering
    with col2:
        st.markdown("### Original X-ray Image")
        folium_static(m_2, width=int(width * scale), height=int(height * scale))

    # SPACER
    st.text("")

    st.markdown("### Select minimum confidence score:")
    minimum_confidence_score = st.slider(' ', 0.0, 1.0, 0.1)

    # SPACER
    st.text("")

    if minimum_confidence_score:
        st.markdown("### Select detection annotation:")
        type_of_annotation = st.radio(' ',
                                    ('mask', 'bounding box', 'both'))
            # SPACER
        st.text("")
        st.text("")
        if type_of_annotation == 'mask':



##### FIRST BUTTON #####
            if st.button("Initialize model"):
                st.divider()
                with st.spinner("Uploading and processing..."):

                    #call_api method to get data
                    data = call_api_cached(uploaded_file)

                    number_of_detections = len(data["detections"]["box"])

                    img = mpimg.imread(uploaded_file)

                    # Taking class IDs from json response
                    cls = []
                    for i in range(number_of_detections):
                        class_id = data["detections"]["box"][i]["classification"]
                        cls.append(class_id)
                    cls = np.array(cls)

                    # Taking mask coordinates from json response
                    masks = []
                    for i in range(number_of_detections):
                        mask_coords = data["detections"]["mask"][i]["xy"]
                        mask_coords = np.array(mask_coords)
                        masks.append(mask_coords)

                    # Taking confidence level from json response
                    confidence_level = []
                    for i in range(number_of_detections):
                        confidence_level.append(data["detections"]["box"][i]["confidence"])
                    confidence_level = np.array(confidence_level)

                    # Create a Folium map, with unvisible map to add image
                    scale = 1.7

                    height, width = img.shape[:2]
                    m = folium.Map(
                        location=[0, 0],
                        zoom_start=0.25,
                        crs="Simple",
                        tiles=None
                    )
                    # add the image as background
                    bounds = [[0, 0], [int(height * scale), int(width * scale)]]
                    folium.raster_layers.ImageOverlay(
                        image=img,
                        bounds=bounds,
                        opacity=1
                    ).add_to(m)

                    def get_class_name(class_id):
                        class_dict = {0:"Caries",
                                1:"Crown",
                                2:"Filling",
                                3:"Implant",
                                4:"Malaligned",
                                5:"Mandibular Canal",
                                6:"Missing teeth",
                                7:"Periapical lesion",
                                8:"Retained root",
                                9:"Root Canal Treatment",
                                10:"Root Piece",
                                11:"impacted tooth",
                                12:"maxillary sinus",
                                13:"Bone Loss",
                                14:"Fracture teeth",
                                15:"Permanent Teeth",
                                16:"Supra Eruption",
                                17:"TAD",
                                18:"abutment",
                                19:"attrition",
                                20:"bone defect",
                                21:"gingival former",
                                22:"metal band",
                                23:"orthodontic brackets",
                                24:"permanent retainer",
                                25:"post - core",
                                26:"plating",
                                27:"wire",
                                28:"Cyst",
                                29:"Root resorption",
                                30:"Primary teeth"}
                        class_id = int(class_id.item())
                        return class_dict.get(class_id)

                    def get_color(class_id):
                        class_id = int(class_id.item())
                        colors_dict = {
                            0: "#FF0000",   # red
                            1: "#0000FF",   # blue
                            2: "#008000",   # green
                            3: "#FFFF00",   # yellow
                            4: "#800080",   # purple
                            5: "#FFA500",   # orange
                            6: "#FFC0CB",   # pink
                            7: "#A52A2A",   # brown
                            8: "#808080",   # gray
                            9: "#00FFFF",   # cyan
                            10: "#FF00FF",  # magenta
                            11: "#00FF00",  # lime
                            12: "#008080",  # teal
                            13: "#000080",  # navy
                            14: "#800000",  # maroon
                            15: "#808000",  # olive
                            16: "#FF7F50",  # coral
                            17: "#FA8072",  # salmon
                            18: "#FFD700",  # gold
                            19: "#F0E68C",  # khaki
                            20: "#DDA0DD",  # plum
                            21: "#DA70D6",  # orchid
                            22: "#ADD8E6",  # lightblue
                            23: "#90EE90",  # lightgreen
                            24: "#FFB6C1",  # lightpink
                            25: "#FFA07A",  # lightsalmon
                            26: "#FFFFE0",  # lightyellow
                            27: "#D3D3D3",  # lightgray
                            28: "#F08080",  # lightcoral
                            29: "#E0FFFF",  # lightcyan
                            30: "#FF77FF"   # lightmagenta (not standard, custom approximation)
                        }
                        return colors_dict.get(class_id)

                    names_for_filter = []
                    # Display masks on image with assigned class and class color
                    for mask, class_id, conf_lvl in zip(masks, cls, confidence_level):
                        if conf_lvl > minimum_confidence_score:
                            x = mask[:, 0] * scale
                            y = mask[:, 1] * scale
                            flipped_y = (height*scale) - y  # Flip y-coordinates
                            coords = list(zip(flipped_y, x))

                            name = get_class_name(class_id)
                            color = get_color(class_id)

                            if name in names_for_filter:
                                names_for_filter = names_for_filter
                            else:
                                names_for_filter.append(name)

                            folium.Polygon(
                                coords,
                                color=color,
                                fill=True,
                                weight=0.3 ,
                                tooltip=f"{name}, confidence level: {conf_lvl:.2f}",
                                fill_opacity=0.2,
                                tags=[name]
                            ).add_to(m)

                    TagFilterButton(names_for_filter).add_to(m)
                    m.fit_bounds(bounds)
                    col1, col2, col3 = st.columns([1, 6, 1])  # Wider center column for centering
                    with col2:

                        st.success("Map generated successfully!")

                        # Replace the detection summary block in each annotation type with:
                        st.markdown("# Detection summary:")
                        st.subheader(f"{number_of_detections} detections found")

                        if number_of_detections > 0:
                            summary = defaultdict(list)
                            for i in range(number_of_detections):
                                name = get_class_name(cls[i])
                                summary[name].append(confidence_level[i])
                            total = sum(len(confidences) for confidences in summary.values())
                            summary_data = []
                            for disease, confidences in summary.items():
                                confidences = np.array(confidences)
                                summary_data.append({
                                    "Disease/Class": disease,
                                    "Count": len(confidences),
                                    "Detection Rate (%)": round(100 * len(confidences) / total, 2),
                                    "Mean Confidence": round(np.mean(confidences), 2),
                                    "Median Confidence": round(np.median(confidences), 2),
                                    "Std Confidence": round(np.std(confidences), 2),
                                    "Min Confidence": round(np.min(confidences), 2),
                                    "Max Confidence": round(np.max(confidences), 2),
                                    "Confidence Scores": [round(float(c), 2) for c in confidences]
                                })
                            df_summary = pd.DataFrame(summary_data)
                            st.dataframe(df_summary, use_container_width=True)


                        st.divider()
                        st.markdown("### X-ray Image with masks predictions")
                        folium_static(m, width=int(width * scale), height=int(height * scale))
                        st.divider()

        elif type_of_annotation == 'bounding box':

##### SECOND BUTTON #####
            if st.button("Initialize model"):
                st.divider()
                with st.spinner("Uploading and processing..."):

                    #call_api method to get data
                    data = call_api_cached(uploaded_file)

                    number_of_detections = len(data["detections"]["box"])
                    img = mpimg.imread(uploaded_file)
                    height, width = img.shape[:2]

                    # Taking confidence level from json response
                    confidence_level = []
                    for i in range(number_of_detections):
                        confidence_level.append(data["detections"]["box"][i]["confidence"])
                    confidence_level = np.array(confidence_level)

                    # Taking class IDs from json response
                    cls = []
                    for i in range(number_of_detections):
                        class_id = data["detections"]["box"][i]["classification"]
                        cls.append(class_id)
                    cls = np.array(cls)

                    # Taking boxes coords from json response
                    boxes = []
                    for i in range(number_of_detections):
                        box = data["detections"]["box"][i]["boxes"][0]
                        boxes.append(box)
                    boxes = np.array(boxes)
                    # boxes = xyxy2xywh(boxes)

                    def get_class_name(class_id):
                        class_dict = {0:"Caries", 1:"Crown", 2:"Filling", 3:"Implant", 4:"Malaligned",
                                5:"Mandibular Canal", 6:"Missing teeth", 7:"Periapical lesion", 8:"Retained root",
                                9:"Root Canal Treatment", 10:"Root Piece", 11:"impacted tooth", 12:"maxillary sinus",
                                13:"Bone Loss", 14:"Fracture teeth", 15:"Permanent Teeth", 16:"Supra Eruption",
                                17:"TAD", 18:"abutment", 19:"attrition", 20:"bone defect", 21:"gingival former",
                                22:"metal band", 23:"orthodontic brackets", 24:"permanent retainer", 25:"post - core",
                                26:"plating", 27:"wire", 28:"Cyst", 29:"Root resorption", 30:"Primary teeth"}
                        class_id = int(class_id.item())
                        return class_dict.get(class_id)

                    def get_color(class_id):
                        class_id = int(class_id.item())
                        colors_dict = {0: "#FF0000", 1: "#0000FF", 2: "#008000", 3: "#FFFF00", 4: "#800080",
                            5: "#FFA500", 6: "#FFC0CB", 7: "#A52A2A", 8: "#808080", 9: "#00FFFF",
                            10: "#FF00FF", 11: "#00FF00", 12: "#008080", 13: "#000080", 14: "#800000",
                            15: "#808000", 16: "#FF7F50", 17: "#FA8072", 18: "#FFD700", 19: "#F0E68C",
                            20: "#DDA0DD", 21: "#DA70D6", 22: "#ADD8E6", 23: "#90EE90", 24: "#FFB6C1",
                            25: "#FFA07A", 26: "#FFFFE0", 27: "#D3D3D3", 28: "#F08080", 29: "#E0FFFF", 30: "#FF77FF"}
                        return colors_dict.get(class_id)

                    scale = 1.7

                    # Create Folium map
                    m_box = folium.Map(
                        location=[0, 0],
                        zoom_start=1,
                        crs="Simple",
                        tiles=None
                    )

                    # Add image overlay
                    bounds = [[0, 0], [int(height * scale), int(width * scale)]]
                    folium.raster_layers.ImageOverlay(
                        image=img,
                        bounds=bounds,
                        opacity=1
                    ).add_to(m_box)

                    names_for_filter = []

                    # Draw bounding boxes using Rectangle
                    for i, (box, class_id, conf_lvl) in enumerate(zip(boxes, cls, confidence_level)):
                        if conf_lvl > 0.3:
                            name = get_class_name(class_id)
                            color = get_color(class_id)
                            if name in names_for_filter:
                                names_for_filter = names_for_filter
                            else:
                                names_for_filter.append(name)

                            # Unpack normalized coordinates
                            x1, y1, x2, y2 = box

                            x1 = x1 * scale
                            x2 = x2 * scale
                            y1 = y1 * scale
                            y2 = y2 * scale

                            # Flip y-axis
                            y1_flipped = (height*scale) - y1
                            y2_flipped = (height*scale) - y2

                            # Draw rectangle with flipped y-axis
                            folium.Rectangle(
                                bounds=[[y1_flipped, x1], [y2_flipped, x2]],
                                color=color,
                                weight=0.2,
                                opacity=0.2,
                                fill=True,
                                fillColor=color,
                                fillOpacity=0.15,
                                tooltip=f"{name}: {conf_lvl:.2f}",
                                popup=f"<b>{name}</b><br>Confidence: {conf_lvl:.2f}",
                                tags=[name]
                            ).add_to(m_box)

                    TagFilterButton(list(set(names_for_filter))).add_to(m_box)
                    m_box.fit_bounds(bounds)

                    col1, col2, col3 = st.columns([1, 6, 1])
                    with col2:
                        st.success("Map generated successfully!")

                        st.markdown("# Detection summary:")
                        st.subheader(f"{number_of_detections} detections found")

                        if number_of_detections > 0:
                            summary = defaultdict(list)
                            for i in range(number_of_detections):
                                name = get_class_name(cls[i])
                                summary[name].append(confidence_level[i])
                            total = sum(len(confidences) for confidences in summary.values())
                            summary_data = []
                            for disease, confidences in summary.items():
                                confidences = np.array(confidences)
                                summary_data.append({
                                    "Disease/Class": disease,
                                    "Count": len(confidences),
                                    "Detection Rate (%)": round(100 * len(confidences) / total, 2),
                                    "Mean Confidence": round(np.mean(confidences), 2),
                                    "Median Confidence": round(np.median(confidences), 2),
                                    "Std Confidence": round(np.std(confidences), 2),
                                    "Min Confidence": round(np.min(confidences), 2),
                                    "Max Confidence": round(np.max(confidences), 2),
                                    "Confidence Scores": [round(float(c), 2) for c in confidences]
                                })
                            df_summary = pd.DataFrame(summary_data)
                            st.dataframe(df_summary, use_container_width=True)

                        st.divider()
                        st.markdown("### X-ray Image with bounding boxes predictions")
                        folium_static(m_box, width=int(width * scale), height=int(height * scale))
                        st.divider()

#### BOTH ####
        elif type_of_annotation == 'both':
            if st.button("Initialize model"):
                st.divider()
                with st.spinner("Uploading and processing..."):

                    #call_api method to get data
                    data = call_api_cached(uploaded_file)

                    number_of_detections = len(data["detections"]["box"])

                    img = mpimg.imread(uploaded_file)

                    # Taking class IDs from json response
                    cls = []
                    for i in range(number_of_detections):
                        class_id = data["detections"]["box"][i]["classification"]
                        cls.append(class_id)
                    cls = np.array(cls)

                    # Taking mask coordinates from json response
                    masks = []
                    for i in range(number_of_detections):
                        mask_coords = data["detections"]["mask"][i]["xy"]
                        mask_coords = np.array(mask_coords)
                        masks.append(mask_coords)

                    # Taking confidence level from json response
                    confidence_level = []
                    for i in range(number_of_detections):
                        confidence_level.append(data["detections"]["box"][i]["confidence"])
                    confidence_level = np.array(confidence_level)

                    # Create a Folium map, with unvisible map to add image
                    scale = 1.7

                    height, width = img.shape[:2]
                    m = folium.Map(
                        location=[0, 0],
                        zoom_start=0.25,
                        crs="Simple",
                        tiles=None
                    )
                    # add the image as background
                    bounds = [[0, 0], [int(height * scale), int(width * scale)]]
                    folium.raster_layers.ImageOverlay(
                        image=img,
                        bounds=bounds,
                        opacity=1
                    ).add_to(m)

                    def get_class_name(class_id):
                        class_dict = {0:"Caries",
                                1:"Crown",
                                2:"Filling",
                                3:"Implant",
                                4:"Malaligned",
                                5:"Mandibular Canal",
                                6:"Missing teeth",
                                7:"Periapical lesion",
                                8:"Retained root",
                                9:"Root Canal Treatment",
                                10:"Root Piece",
                                11:"impacted tooth",
                                12:"maxillary sinus",
                                13:"Bone Loss",
                                14:"Fracture teeth",
                                15:"Permanent Teeth",
                                16:"Supra Eruption",
                                17:"TAD",
                                18:"abutment",
                                19:"attrition",
                                20:"bone defect",
                                21:"gingival former",
                                22:"metal band",
                                23:"orthodontic brackets",
                                24:"permanent retainer",
                                25:"post - core",
                                26:"plating",
                                27:"wire",
                                28:"Cyst",
                                29:"Root resorption",
                                30:"Primary teeth"}
                        class_id = int(class_id.item())
                        return class_dict.get(class_id)

                    def get_color(class_id):
                        class_id = int(class_id.item())
                        colors_dict = {
                            0: "#FF0000",   # red
                            1: "#0000FF",   # blue
                            2: "#008000",   # green
                            3: "#FFFF00",   # yellow
                            4: "#800080",   # purple
                            5: "#FFA500",   # orange
                            6: "#FFC0CB",   # pink
                            7: "#A52A2A",   # brown
                            8: "#808080",   # gray
                            9: "#00FFFF",   # cyan
                            10: "#FF00FF",  # magenta
                            11: "#00FF00",  # lime
                            12: "#008080",  # teal
                            13: "#000080",  # navy
                            14: "#800000",  # maroon
                            15: "#808000",  # olive
                            16: "#FF7F50",  # coral
                            17: "#FA8072",  # salmon
                            18: "#FFD700",  # gold
                            19: "#F0E68C",  # khaki
                            20: "#DDA0DD",  # plum
                            21: "#DA70D6",  # orchid
                            22: "#ADD8E6",  # lightblue
                            23: "#90EE90",  # lightgreen
                            24: "#FFB6C1",  # lightpink
                            25: "#FFA07A",  # lightsalmon
                            26: "#FFFFE0",  # lightyellow
                            27: "#D3D3D3",  # lightgray
                            28: "#F08080",  # lightcoral
                            29: "#E0FFFF",  # lightcyan
                            30: "#FF77FF"   # lightmagenta (not standard, custom approximation)
                        }
                        return colors_dict.get(class_id)

                    names_for_filter = []
                    # Display masks on image with assigned class and class color
                    for mask, class_id, conf_lvl in zip(masks, cls, confidence_level):
                        if conf_lvl > minimum_confidence_score:
                            x = mask[:, 0] * scale
                            y = mask[:, 1] * scale
                            flipped_y = (height*scale) - y  # Flip y-coordinates
                            coords = list(zip(flipped_y, x))

                            name = get_class_name(class_id)
                            color = get_color(class_id)

                            if name in names_for_filter:
                                names_for_filter = names_for_filter
                            else:
                                names_for_filter.append(name)

                            folium.Polygon(
                                coords,
                                color=color,
                                fill=True,
                                weight=0.3,
                                tooltip=f"{name}, confidence level: {conf_lvl:.2f}",
                                fill_opacity=0.15,
                                tags=[name]
                            ).add_to(m)

                    TagFilterButton(names_for_filter).add_to(m)
                    m.fit_bounds(bounds)
                    col1, col2, col3 = st.columns([1, 6, 1])  # Wider center column for centering
                    with col2:
                        st.success("Map generated successfully!")
                        st.write("")
                        if number_of_detections > 0:
                            # Group detections by disease name

                            # Detection summary
                            st.markdown("# Detection summary:")
                            st.subheader(f"{number_of_detections} detections found")

                            if number_of_detections > 0:
                                summary = defaultdict(list)
                                for i in range(number_of_detections):
                                    name = get_class_name(cls[i])
                                    summary[name].append(confidence_level[i])
                                total = sum(len(confidences) for confidences in summary.values())
                                summary_data = []
                                for disease, confidences in summary.items():
                                    confidences = np.array(confidences)
                                    summary_data.append({
                                        "Disease/Class": disease,
                                        "Count": len(confidences),
                                        "Detection Rate (%)": round(100 * len(confidences) / total, 2),
                                        "Mean Confidence": round(np.mean(confidences), 2),
                                        "Median Confidence": round(np.median(confidences), 2),
                                        "Std Confidence": round(np.std(confidences), 2),
                                        "Min Confidence": round(np.min(confidences), 2),
                                        "Max Confidence": round(np.max(confidences), 2),
                                        "Confidence Scores": [round(float(c), 2) for c in confidences]
                                    })
                                df_summary = pd.DataFrame(summary_data)
                                st.dataframe(df_summary, use_container_width=True)

                        st.divider()
                        st.markdown("### X-ray Image with masks predictions")
                        folium_static(m, width=int(width * scale), height=int(height * scale))
                        st.divider()


#### SECOND IMAGE - BOTH #####

                    # Taking confidence level from json response
                    confidence_level = []
                    for i in range(number_of_detections):
                        confidence_level.append(data["detections"]["box"][i]["confidence"])
                    confidence_level = np.array(confidence_level)

                    # Taking class IDs from json response
                    cls = []
                    for i in range(number_of_detections):
                        class_id = data["detections"]["box"][i]["classification"]
                        cls.append(class_id)
                    cls = np.array(cls)

                    # Taking boxes coords from json response
                    boxes = []
                    for i in range(number_of_detections):
                        box = data["detections"]["box"][i]["boxes"][0]
                        boxes.append(box)
                    boxes = np.array(boxes)

                    scale = 1.7

                    # Create Folium map
                    m_box = folium.Map(
                        location=[0, 0],
                        zoom_start=1,
                        crs="Simple",
                        tiles=None
                    )

                    # Add image overlay
                    bounds = [[0, 0], [int(height * scale), int(width * scale)]]
                    folium.raster_layers.ImageOverlay(
                        image=img,
                        bounds=bounds,
                        opacity=1
                    ).add_to(m_box)

                    names_for_filter = []

                    # Draw bounding boxes using Rectangle
                    for i, (box, class_id, conf_lvl) in enumerate(zip(boxes, cls, confidence_level)):
                        if conf_lvl > 0.3:
                            name = get_class_name(class_id)
                            color = get_color(class_id)
                            if name in names_for_filter:
                                names_for_filter = names_for_filter
                            else:
                                names_for_filter.append(name)

                            # Unpack normalized coordinates
                            x1, y1, x2, y2 = box

                            x1 = x1 * scale
                            x2 = x2 * scale
                            y1 = y1 * scale
                            y2 = y2 * scale

                            # Flip y-axis
                            y1_flipped = (height*scale) - y1
                            y2_flipped = (height*scale) - y2

                            # Draw rectangle with flipped y-axis
                            folium.Rectangle(
                                bounds=[[y1_flipped, x1], [y2_flipped, x2]],
                                color=color,
                                weight=0.2,
                                opacity=0.3,
                                fill=True,
                                fillColor=color,
                                fillOpacity=0.15,
                                tooltip=f"{name}: {conf_lvl:.2f}",
                                popup=f"<b>{name}</b><br>Confidence: {conf_lvl:.2f}",
                                tags=[name]
                            ).add_to(m_box)

                    TagFilterButton(list(set(names_for_filter))).add_to(m_box)
                    m_box.fit_bounds(bounds)

                    col1, col2, col3 = st.columns([1, 6, 1])
                    with col2:
                        st.markdown("### X-ray Image with bounding boxes predictions")
                        folium_static(m_box, width=int(width * scale), height=int(height * scale))
                        st.divider()

            else:
                pass
