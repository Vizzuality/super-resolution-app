import os
import sys
import json
from PIL import Image

import ee
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# Include local library paths if you have ../src/utils.py
sys.path.append("src/")

from src.maps import FoliumMap
from src.models import srgan
from src.predict import Predictor
from src.verification import selected_bbox_too_large, selected_bbox_in_boundary


MAP_CENTER = [40.4786, -3.4556]
MAP_ZOOM = 13
MAX_ALLOWED_AREA_SIZE = 20.0
BTN_LABEL = "Submit"
WEIGHTS_DIR = 'data/Models/L8_S2_SR_x3/srgan_generator_L8_to_S2_x3'

# Initialize GEE
private_key = dict(st.secrets["EE_PRIVATE_KEY"])

ee_credentials = ee.ServiceAccountCredentials(email=private_key['client_email'], key_data=json.dumps(private_key))
ee.Initialize(credentials=ee_credentials)

# Create model
weights_file = lambda filename: os.path.join(WEIGHTS_DIR, filename)
pre_generator = srgan.Generator(input_shape=(None, None, 3), scale=3).generator()
pre_generator.load_weights(weights_file('model_weights.h5'))

# Load icon
footer = Image.open("images/redes.png")

# Create the Streamlit app and define the main code:
def main():
    st.set_page_config(
        page_title="Super_resolution-app",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Super Resolution App")

    m = FoliumMap(center=MAP_CENTER, zoom=MAP_ZOOM)

    output = st_folium(m, key="init", width=1200, height=600)

    st.image(footer)

    geojson = None
    if output["all_drawings"] is not None:
        if len(output["all_drawings"]) != 0:
            if output["last_active_drawing"] is not None:
                # get latest modified drawing
                geojson = output["last_active_drawing"]

    # ensure progress bar resides at top of sidebar and is invisible initially
    progress_bar = st.sidebar.progress(0)
    progress_bar.empty()


    # Getting Started container
    with st.sidebar.container():
        # Getting started
        st.subheader("Getting Started")
        st.markdown(
            f"""
                        1. Click the black square on the map
                        2. Draw a rectangle on the map
                        3. Click on <kbd>{BTN_LABEL}</kbd>
                        4. Wait for the computation to finish
                        """,
            unsafe_allow_html=True,
        )

        # Add the button and its callback
        if st.button(
            BTN_LABEL,
            key="compute_zs",
            disabled=False if geojson is not None else True,
        ):
            # Check if the geometry is valid
            geometry = geojson['geometry']
            if selected_bbox_too_large(geometry, threshold=MAX_ALLOWED_AREA_SIZE):
                st.sidebar.warning(
                    "Selected region is too large, fetching data for this area would consume too many resources. "
                    "Please select a smaller region."
                )
            elif not selected_bbox_in_boundary(geometry):
                st.sidebar.warning(
                    "Selected rectangle is not within the allowed region of the world map. "
                    "Do not scroll too far to the left or right. "
                    "Ensure to use the initial center view of the world for drawing your rectangle."
                )
            else:
                # Create input image
                array = m.create_input_image(geojson)

                # Get prediction
                predictor = Predictor(folder_path = 'data/Models/', dataset_name = 'L8_S2_SR_x3', model = pre_generator)
                prediction = predictor.predict(input_array=array, norm_range=[[0,1], [-1,1]])

                # Display input/output
                st.subheader("Model input:")

                fig, ax = plt.subplots(1, 1, figsize=(10,10))
                ax.imshow(array[0,:,:,:3])
                ax.axis('off')
                st.pyplot(fig)

                st.subheader("Model output:")

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(prediction[0,:,:,:3])
                ax.axis('off')
                st.pyplot(fig)


if __name__ == "__main__":
    main()
