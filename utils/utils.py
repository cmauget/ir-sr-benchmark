import matplotlib.pyplot as plt #type: ignore
import cv2 #type:ignore
import os
import imageio.v2 as imageio #type:ignore
import tensorflow as tf #type: ignore
import numpy as np #type:ignore

class Image:

    @staticmethod
    def save(image_path:str, image):
        cv2.imwrite(image_path, image)

    @staticmethod
    def load(image_path:str):
        if (os.path.basename(image_path)==".DS_Store"):
            print("Fichier .Ds_store trouvé, relancer le programme")
            img = None
            os.remove(image_path)
        else:
            img = cv2.imread(image_path)
        return img
    
    @staticmethod
    def loadio(image_path:str):
        if (os.path.basename(image_path)==".DS_Store"):
            print("Fichier .Ds_store trouvé, relancer le programme")
            img = None
            os.remove(image_path)
        else:
            img = imageio.imread(image_path)
        return img
    
    @staticmethod
    def load_streamlit(uploaded_file, image_path = "temp_image.tif", bw = False):
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if bw:
            image = cv2.imread(image_path, 0)
        image = cv2.imread(image_path)

        return image
    
    @staticmethod
    def invert_image(image):
        return cv2.bitwise_not(image)
    
    @staticmethod
    def resize_img(image_hr, ratio=4):

        largeur_hr = image_hr.shape[1]
        hauteur_hr = image_hr.shape[0]

        largeur_hr2 = (largeur_hr // ratio) * ratio
        hauteur_hr2 = (hauteur_hr // ratio) * ratio

        image_hr2 = cv2.resize(image_hr, (largeur_hr2, hauteur_hr2))
        
        largeur_lr = largeur_hr2 // ratio
        hauteur_lr = hauteur_hr2 // ratio
        image_lr = cv2.resize(image_hr2, (largeur_lr, hauteur_lr))

        return image_hr2, image_lr
    
    @staticmethod
    def crop_img(input_image, crop_coords):
        cropped_img = input_image[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
        return cropped_img

    
class Data_Utils :

    @staticmethod
    def graphe(liste_image, liste_titre)->None:
        size = len(liste_image)
        if size%3==0:
            height = size//3
        else:
            height = len(liste_image)//3 + 1
        for i, (image, titre) in enumerate(zip(liste_image, liste_titre),start=1):
                plt.subplot(height,3,i)
                # Original image
                plt.imshow(image[:,:,::-1], origin="lower")
                plt.title(titre)
                plt.xlim(image.shape[1]//8, 2*image.shape[1]//8)  
                plt.ylim(2*image.shape[0]//8,image.shape[0]//8) 
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def find_path(folder_path:str) -> list[str]:
        chemin = []
        for nom_fichier in os.listdir(folder_path):
            chemin_ = os.path.join(folder_path, nom_fichier)
            chemin.append(chemin_)
        
        return chemin
    
    @staticmethod
    def create_folder(fodler_path:str, rm=True)->None :
         
        if not os.path.exists(fodler_path):
            os.makedirs(fodler_path)

        if rm:
            for chemin_fichier in Data_Utils.find_path(fodler_path): 
                if os.path.isfile(chemin_fichier):  
                    os.remove(chemin_fichier)  
         
    @staticmethod
    def resize(dossier_hr:str,  dossier_hr2:str, dossier_lr:str, ratio=4)->None:
         
        Data_Utils.create_folder(dossier_hr2)
        Data_Utils.create_folder(dossier_lr)

        for nom_fichier in os.listdir(dossier_hr):
            chemin_hr = os.path.join(dossier_hr, nom_fichier)
            
            image_hr = cv2.imread(chemin_hr)

            largeur_hr = image_hr.shape[1]
            hauteur_hr = image_hr.shape[0]

            largeur_hr2 = (largeur_hr // ratio) * ratio
            hauteur_hr2 = (hauteur_hr // ratio) * ratio

            image_hr2 = cv2.resize(image_hr, (largeur_hr2, hauteur_hr2))
            
            chemin_hr2 = os.path.join(dossier_hr2, nom_fichier)
            
            cv2.imwrite(chemin_hr2, image_hr2)
            
            largeur_lr = largeur_hr2 // ratio
            hauteur_lr = hauteur_hr2 // ratio
            image_lr = cv2.resize(image_hr2, (largeur_lr, hauteur_lr))
            
            chemin_lr = os.path.join(dossier_lr, nom_fichier)
            
            cv2.imwrite(chemin_lr, image_lr)

        print("La réduction de résolution des images est terminée.")

    @staticmethod    
    def crop(input_image:str, output_image:str, crop_coords):
        img = cv2.imread(input_image)
        cropped_img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
        cv2.imwrite(output_image, cropped_img)

    @staticmethod
    def convert(dossier_ir:str, dossier_hr:str)->None:
        chemin_ir = []
        Data_Utils.create_folder(dossier_hr)
        chemin_ir = Data_Utils.find_path(dossier_ir)

        for chemin in chemin_ir:

            image = Data_Utils.loadio(chemin)

            image_scaled = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            image_rgb = cv2.cvtColor(image_scaled, cv2.COLOR_GRAY2RGB)

            nom_fichier_ = os.path.basename(chemin)
            chemin_ = os.path.join(dossier_hr, nom_fichier_)
            cv2.imwrite(chemin_, image_rgb)


tf.get_logger().setLevel('ERROR')

class Model_utils:

    @staticmethod
    def create_model():
        data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ]
        )

        model = tf.keras.models.Sequential([
            
            data_augmentation,
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
        
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2)
        ])

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"])
                            #tf.keras.metrics.FalseNegatives()])

        return model
    
    @staticmethod
    def predict_and_draw_boxes(image, model):
        
        # Découper l'image en carrés de 96x96
        height, width, _ = image.shape
        num_crops = min(height, width) // 96
        crops = []
        
        for i in range(num_crops):
            for j in range(num_crops):
                crop = image[i*96:(i+1)*96, j*96:(j+1)*96]
                crops.append(crop)
        
        # Convertir les crops en un tableau NumPy
        crops = np.array(crops)
        
        # Normaliser les valeurs des pixels entre 0 et 1
        crops = crops 
        
        # Effectuer les prédictions
        predictions = model.predict(crops)
        #print(predictions)
        
        # Dessiner des carrés autour des prédictions positives
        for idx, pred in enumerate(predictions):
            i = idx // num_crops
            j = idx % num_crops
            x = j * 96
            y = i * 96

            if np.argmax(pred) == 1:
                col = (0,255,0)
                cv2.rectangle(image, (x, y), (x+96, y+96), col, 2)
            else:
                col = (255,0,0)
            
        
        # Afficher l'image avec les carrés dessinés
        return image