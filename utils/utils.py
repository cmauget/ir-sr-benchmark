import matplotlib.pyplot as plt #type: ignore
import cv2 #type:ignore
import os
import imageio.v2 as imageio #type:ignore
import tensorflow as tf #type: ignore
import numpy as np #type:ignore

#---------------------------Image----------------------------#

class Image:

    @staticmethod
    def save(image_path:str, image)->None:
        """Permet d'enregister une image

        Parameters
        ----------
        image_path : str
            Le chemin ou enregistrer l'image
        image : numpy.ndarray
            L'image a enregister
        """
        cv2.imwrite(image_path, image)

    @staticmethod
    def load(image_path:str):
        """Permet de charger une image standard
        
        Parameters
        ----------
        image_path : str
            Le chemin de l'image à charger

        Returns
        ----------
        img : numpy.ndarray
            L'image chargée
        """
        if (os.path.basename(image_path)==".DS_Store"):
            print("Fichier .Ds_store trouvé, relancer le programme")
            img = None
            os.remove(image_path)
        else:
            img = cv2.imread(image_path)
        return img
    
    @staticmethod
    def loadio(image_path:str):
        """Permet d'enregister une image avec 1 channel
        
        Parameters
        ----------
        iamge_path : str
            Le chemin de l'image à charger

        Returns
        ----------
        img : numpy.ndarray
            L'image chargée
        """
        if (os.path.basename(image_path)==".DS_Store"):
            print("Fichier .Ds_store trouvé, relancer le programme")
            img = None
            os.remove(image_path)
        else:
            img = imageio.imread(image_path)
        return img
    
    @staticmethod
    def load_streamlit(uploaded_file, image_path = "temp_image.tif", bw = False):
        """Permet d'enregister une image issue du widget file uploader
        de streamlit
        
        Parameters
        ----------
        uploaded_file : 
            Sortie du widget file_uploader
        image_path : str, optional
            Fichier temporaire ou l'image sera enregistré
        bw : boolean, optional
            Si vrai, l'image sera chargée en noir et blanc
        
        Returns
        ----------
        img : numpy.ndarray
            L'image chargée
        """
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if bw:
            img = cv2.imread(image_path, 0)
        else: 
            img = cv2.imread(image_path)

        return img
    
    @staticmethod
    def invert_image(image):
        """Permet d'inverser une image
        
        Parameters
        ----------
        image : numpy.ndarray
            L'image à inverser
        
        Returns
        ----------
        img : numpy.ndarray
            L'image chargée
        """
        img = cv2.bitwise_not(image)
        return img
    
    @staticmethod
    def resize_img(image_hr, ratio=4):
        """Resize l'image pour la super-résolution

        1/ Il crée une image dans la dimensiion et divisible par 4
        2/ Il divise par quatre la résolution de l'image
        
        Parameters
        ----------
        image : numpy.ndarray
            L'image à inverser
        ratio : int, optional
            Le ration pour reduire l'image
        
        Returns
        ----------
        image_hr2: numpy.ndarray
            L'image divisble par 4
        image_lr: numpy.ndarray
            L'image divisée
        """
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
    def crop_img(image, crop_coords):
        """Crop l'image en fonction des coordonées entrées
        
        Parameters
        ----------
        image : numpy.ndarray
            L'image à crop
        croop_coords : tuples
            Les coordonées ou cropper dans l'ordre [x1, x2, y1, y2]
        
        Returns
        ----------
        img: numpy.ndarray
            L'image crop
        """
        img = image[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
        return img


#-------------------------Data_Utils--------------------------#

class Data_Utils :

    @staticmethod
    def graphe(liste_image, liste_titre)->None:
        """Affiche des images en ligne de 3

        Les deux premiers titres sont affichés en gras
        
        Parameters
        ----------
        liste_image : list[numpy.ndarray]
            Liste des images à afficher
        liste_titre : list[str]
            Liste des titres associées aux images
        
        """
        size = len(liste_image)
        if size%3==0:
            height = size//3
        else:
            height = len(liste_image)//3 + 1
        for i, (image, titre) in enumerate(zip(liste_image, liste_titre),start=1):
                dec = 40
                plt.subplot(height,3,i)
                if i == 1:
                    dec = dec/4
                    plt.title(titre, fontweight='bold')
                elif i == 2:
                    plt.title(titre, fontweight='bold')
                else:
                    plt.title(titre)
                # Original image
                plt.imshow(image[:,:,::-1], origin="lower")
                
                plt.xlim(image.shape[1]//8 + dec, 2*image.shape[1]//8 +dec)  
                plt.ylim(2*image.shape[0]//8,image.shape[0]//8) 
                plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def find_path(folder_path:str) -> list[str]:
        """Retrouve les chemins des images d'un dossier
        
        Parameters
        ----------
        folder_path : str
            Le dossier ou se trouve les images

        Returns
        ----------
        liste_chemin: list[str]
            La liste de tout les chemis des images
        """
        liste_chemin = []
        for nom_fichier in os.listdir(folder_path):
            chemin_ = os.path.join(folder_path, nom_fichier)
            liste_chemin.append(chemin_)
        
        return liste_chemin
    
    @staticmethod
    def create_folder(folder_path:str, rm=False)->None :
        """Crée un dossier

        Il vérifie que le dossier existe, le crée sinon et supprime
        son contenue avant si besoin
        
        Parameters
        ----------
        folder_path : str
            Le chemin du dossier à créer
        rm : Boolean, optional
            Si vrai, le contenu du dossier sera supprimé avant
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if rm:
            for chemin_fichier in Data_Utils.find_path(folder_path): 
                if os.path.isfile(chemin_fichier):  
                    os.remove(chemin_fichier)  
         
    @staticmethod
    def resize(dossier_hr:str,  dossier_hr2:str, dossier_lr:str, ratio=4)->None:
        """Resize l'image pour la super-résolution (legacy)

        Prend en entrée les chemins des dossiers
        """
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
        """Crop l'image (legacy)

        Prend en entrée les chemins des dossiers
        """
        img = cv2.imread(input_image)
        cropped_img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
        cv2.imwrite(output_image, cropped_img)

    @staticmethod
    def convert(dossier_ir:str, dossier_hr:str)->None:
        """Convertit l'image 1 channel en RGB (legacy)

        Prend en entrée les chemins des dossiers
        """
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




#-------------------------Model_Utils--------------------------#

tf.get_logger().setLevel('ERROR')

class Model_utils:
    
    @staticmethod
    def create_model():
        """Crée le modèle pour la classification

        Returns
        ----------
        model: tf.model
            La structure du model de classification
        """
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
        """Prédit la position des fissures

        Récupére l'image, la découpe en carrée de 96px, fait passer les carrées
        dans le modèle, affiche un carrée vert si une fissure est détéctée
        
        Parameters
        ----------
        image: numpy.ndarray
            L'image 
        model: tf.model
            Le model pour les prédictions

        Returns
        ----------
        image: numpy.ndarray
            L'image avec les prédictions
        """
        height, width, _ = image.shape
        num_crops = min(height, width) // 96
        crops = []
        
        for i in range(num_crops):
            for j in range(num_crops):
                crop = image[i*96:(i+1)*96, j*96:(j+1)*96]
                crops.append(crop)

        crops = np.array(crops)

        crops = crops 

        predictions = model.predict(crops)
        
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
            
        return image