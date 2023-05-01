import streamlit as st
import multiprocessing

import os

from predict_image import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    st.title("Image Prediction")

    menu = ["Home", "Image Prediction", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Image prediction":
        st.subheader("Home")
        image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
        if image_file is not None:
            file_details = {"Filename": image_file.name, "FileType": image_file.type, "FileSize": image_file.size}
            st.write(file_details)

            image = load_image(image_file)
            pred = image_prediction(image)
            st.write(pred)
    else:
        st.subheader("About")
        st.info("DL-Ops Project")
        st.text("Udayan Ghosh, Souvik Pal, Sourav Banerjee")


if __name__ == '__main__':
    main()
