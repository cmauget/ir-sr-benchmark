import streamlit as st#type: ignore
import numpy as np #type: ignore
from utils.fuse import Fuse
from utils.utils import Image

                        
def main():

    save_path_r = "IMG/"

    st.title("Fuse")
    st.write("Sélectionnez la méthode de fusion, upload l'image visible (en premier) puis la thermique. Si le résultat n'est pas satisfaisant cochez Invert Thermal Image.")

    fusion_type = st.sidebar.selectbox("Select fusion type", ["Wavelet", "Multiplicative", "Additive"])

    uploaded_files = st.sidebar.file_uploader("Upload visible and thermal images", type=["png", "jpg", "tif"], accept_multiple_files=True)

    invert_thermal = st.sidebar.checkbox("Invert thermal Image")
    
    if len(uploaded_files) == 2:
        visible_image = Image.load_streamlit(uploaded_files[0], bw = True)
        thermal_image = Image.load_streamlit(uploaded_files[1], bw = True)

        if invert_thermal:
            thermal_image = Image.invert_image(thermal_image)
        

        visible_array = np.array(visible_image)
        thermal_array = np.array(thermal_image)

        if fusion_type == "Additive":
            fused_image = Fuse.additive_fusion(thermal_array, visible_array)
        elif fusion_type == "Multiplicative":
            fused_image = Fuse.multiplicative_fusion(thermal_array, visible_array)
        elif fusion_type == "Wavelet":
            fused_image = Fuse.wavelet_fusion(thermal_array, visible_array)

        
        save_path = save_path_r + f"{uploaded_files[0].name.split('.')[0]}_{fusion_type}.png"                
        Image.save(save_path, fused_image)
        st.sidebar.write("Constraste : ", np.std(fused_image))
        st.image(fused_image, caption=f"Image fuse avec {fusion_type}", use_column_width=False)
        st.success(f"Image enregistrée : {save_path}")
    

if __name__ == "__main__":
    main()

    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
