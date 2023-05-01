import streamlit as st
import multiprocessing
import pandas as pd
import os

from predict_image import *
import numpy as np
from mapping import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    st.title("Image Prediction System ")

    menu = ["Image Prediction", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Image Prediction":
        st.subheader("Image Prediction")
        image_file = st.file_uploader("Upload Image", type=['jpeg', 'jpg'])
        if st.button("Process"):
            if image_file is not None:
                file_details = {"Filename": image_file.name, "FileType": image_file.type, "FileSize": image_file.size}
                st.write(file_details)

                arr = []

                image = load_image(image_file)
                pred = image_prediction(image)
                for key in pred:
                    arr.append(np.array[key, get_image_class(int(pred.get(key)))])
                arr = np.array(arr)
                index_values = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
                col_values = ['Probability in %', 'Class']
                df = pd.DataFrame(data=arr, index=index_values, columns=col_values)
                st.table(df)
                #st.write(pred)
    else:
        st.subheader("About")
        st.info("DL-Ops Project")
        st.text("Udayan Ghosh, Souvik Pal, Sourav Banerjee")


if __name__ == '__main__':
    main()
