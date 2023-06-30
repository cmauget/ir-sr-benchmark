import streamlit as st #type: ignore
from utils.utils_models import utils as u

import os

def main():
    st.title("Détection de fissures")
    
    model = u.create_model()
    model.load_weights("models/crack_classifier/cp.ckpt")

    uploaded_file = st.sidebar.file_uploader("Importer une image", type=["png", "jpg", "jpeg","tif"])
    
    if uploaded_file is not None:

        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Détecter"):
            image = u.predict_and_draw_boxes(image_path, model)
            st.image(image, use_column_width=True, caption="Image prédite avec les fissures détectées")
    
    if uploaded_file is not None:
        os.remove(image_path)

if __name__ == "__main__":
    main()