import streamlit as st #type: ignore
from utils.utils import Image 
from models.EDSR import EDSR
from models.ESPCN import ESPCN
from models.FSRCNN import FSRCNN
from models.LapSRN import LapSRN
from models.ESRGAN import ESRGAN
from models.ESRGAN2 import ESRGAN2
from models.PSRGAN import PSRGAN


upscaler_classes = [FSRCNN, EDSR, PSRGAN, ESRGAN2]
device_list = ["cpu", "cuda", "mps"]


def main():

    st.title("Upscaler")

    st.write("Choisissez un algrotihme, l'accélérateur (cpu par défaut, cuda si gpu NVIDIA, mps si apple sillicone), et enfin les images à uploader")

    selected_upscalers = st.sidebar.multiselect("Sélectionner les upscalers", upscaler_classes, default=FSRCNN, format_func=lambda x: x.__name__)

    device = st.sidebar.radio("Selectioner l'accelerateur",device_list)

    uploaded_files = st.sidebar.file_uploader("Uploader des images", type=["png", "jpg", "tif"], accept_multiple_files=True)

    save_path_r = "sr_image/"
    
    if uploaded_files != []:

        if st.button("Upscale !"):
        
            with st.spinner("Upscaling please wait..."):

                i=0
                mybar = st.progress(i)

                for upscaler_class in selected_upscalers:
                    upscaler_instance = upscaler_class(device_=device)

                    for uploaded_file in uploaded_files:
                        i+=1

                        image = Image.load_streamlit(uploaded_file)

                        upscaled_image, _ = upscaler_instance.upscale(image)
                    
                        save_path = save_path_r + f"{uploaded_file.name.split('.')[0]}_{upscaler_class.__name__}.tif"

                        Image.save(save_path, upscaled_image)

                        mybar.progress(int((i/((len(uploaded_files)*len(selected_upscalers)))*100)))

                st.success(f"Image upscale enregistrée : {save_path_r}")
                st.image(upscaled_image, caption=f"Image upscale avec {upscaler_class.__name__}", use_column_width=True)


if __name__ == "__main__":
    main()

    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
