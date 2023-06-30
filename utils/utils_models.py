import tensorflow as tf #type: ignore
import cv2 #type: ignore
import numpy as np #type: ignore

tf.get_logger().setLevel('ERROR')

class utils:

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
    def predict_and_draw_boxes(image_path, model):

        image = cv2.imread(image_path)
        
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