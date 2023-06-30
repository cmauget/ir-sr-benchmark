import streamlit as st #type: ignore
import cv2 #type: ignore
import os
import sys
from models.EDSR import EDSR
from models.ESPCN import ESPCN
from models.FSRCNN import FSRCNN
from models.LapSRN import LapSRN
from models.ESRGAN import ESRGAN
from models.ESRGAN2 import ESRGAN2
from models.PSRGAN import PSRGAN

folder = os.path.dirname(__file__)
sys.path.append(folder+"/..")

upscaler_classes = [FSRCNN, EDSR, PSRGAN, ESRGAN]
device_ = ["cpu", "cuda", "mps"]

def upscale_image(image, upscaler, device):

    upscaler_instance = upscaler(device_=device)

    upscaled_image, _ = upscaler_instance.upscale(image)

    return upscaled_image

def main():
    st.title("App de mise à l'échelle d'images")

    selected_upscalers = st.sidebar.multiselect("Sélectionner les upscalers", upscaler_classes, default=FSRCNN, format_func=lambda x: x.__name__)

    device_selcted = st.sidebar.radio("Selectioner l'accelerateur",device_)

    uploaded_files = st.sidebar.file_uploader("Uploader des images", type=["png", "jpg", "tif"], accept_multiple_files=True)

    save_path_r = "sr_image/"

    if st.button("Upscale !"):
        if uploaded_files != []:
            with st.spinner("Upscaling please wait..."):
                i=0
                mybar = st.progress(i)
                for upscaler_class in selected_upscalers:
                    for uploaded_file in uploaded_files:
                        i+=1

                        image_path = "temp_image.tif"
                        with open(image_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        image = cv2.imread(image_path)
                        upscaled_image = upscale_image(image, upscaler_class, device_selcted)
                        print("upscale done")
                    
                        save_path = save_path_r + f"{uploaded_file.name.split('.')[0]}_{upscaler_class.__name__}.tif"
                        cv2.imwrite(save_path, upscaled_image)
                        mybar.progress(int((i/((len(uploaded_files)*len(selected_upscalers)))*100)))
                st.success(f"Image upscale enregistrée : {save_path_r}")
                st.image(upscaled_image, caption=f"Image upscale avec {upscaler_class.__name__}", use_column_width=True)
                os.remove(image_path)
        else:
            st.error("Erreur : Aucune image uploaded")

if __name__ == "__main__":
    main()
