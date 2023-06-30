import streamlit as st#type: ignore
import cv2 #type: ignore
import numpy as np #type: ignore
from utils.fuse import Fuse

def load_image(uploaded_file):

    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    image = cv2.imread(image_path, 0)

    return image
                        
def main():

    save_path_r = "IMG/"


    st.title("Image Fusion App")
    st.write("Choose the fusion type and upload the visible and thermal images.")

    fusion_type = st.sidebar.selectbox("Select fusion type", ["Additive", "Multiplicative", "Wavelet"])

    invert_thermal = st.sidebar.checkbox("Invert thermal Image")

    uploaded_files = st.sidebar.file_uploader("Upload visible and thermal images", type=["png", "jpg", "tif"], accept_multiple_files=True)
    
    if len(uploaded_files) == 2:
        visible_image = load_image(uploaded_files[0])
        thermal_image = load_image(uploaded_files[1])

        if invert_thermal:
            thermal_image = cv2.bitwise_not(thermal_image)
        

        visible_array = np.array(visible_image)
        thermal_array = np.array(thermal_image)

        if fusion_type == "Additive":
            fused_image = Fuse.additive_fusion(thermal_array, visible_array)
        elif fusion_type == "Multiplicative":
            fused_image = Fuse.multiplicative_fusion(thermal_array, visible_array)
        elif fusion_type == "Wavelet":
            fused_image = Fuse.wavelet_fusion(thermal_array, visible_array)

        
        save_path = save_path_r + f"{uploaded_files[0].name.split('.')[0]}_{fusion_type}.png"                
        cv2.imwrite(save_path, fused_image)
        st.sidebar.write("Constraste : ", np.std(fused_image))
        st.image(fused_image, caption=f"Image fuse avec {fusion_type}", use_column_width=False)
        st.success(f"Image enregistrée : {save_path}")
    else:
        st.write("Please upload both visible and thermal images.")

if __name__ == "__main__":
    main()
