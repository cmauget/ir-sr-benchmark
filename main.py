import os
import cv2 #type:ignore
from data_utils import Data_Utils
import matplotlib.pyplot as plt #type:ignore
import numpy as np #type:ignore
from skimage.metrics import normalized_mutual_information #type:ignore
from skimage.metrics import structural_similarity #type:ignore
from models.EDSR import EDSR
from models.ESPCN import ESPCN
from models.FSRCNN import FSRCNN
from models.LapSRN import LapSRN
from models.ESRGAN import ESRGAN
from models.ESRGAN2 import ESRGAN2
from models.PSRGAN import PSRGAN
from tqdm import tqdm #type:ignore
import random
import time

dossier_lr = "lr_image"
dossier_sr = "sr_image"
dossier_hr = "hr2_image"

chemins_lr = []
chemins_hr = []
chemins_sr = []

resultats = []
psnr_values = []
temps_execution_classe = []
ifm_values_classe = []


blur = (1,1)


if not os.path.exists(dossier_sr):
    os.makedirs(dossier_sr)


for nom_fichier in os.listdir(dossier_lr):
    chemin_ = os.path.join(dossier_lr, nom_fichier)
    chemins_lr.append(chemin_)

for nom_fichier in os.listdir(dossier_hr):
    chemin_ = os.path.join(dossier_hr, nom_fichier)
    chemins_hr.append(chemin_)

seed = int(time.time())

random.seed(seed) 
random.shuffle(chemins_hr)

random.seed(seed) 
random.shuffle(chemins_lr)


classes = [ESPCN, FSRCNN,  ESRGAN, ESRGAN2, PSRGAN, EDSR, LapSRN ]
nom_classes = []
for classe in classes:  
    model = classe()
    nom_classe = classe.__name__
    nom_classes.append(nom_classe)
    psnr_values_classe = []
    temps_execution = []
    ifm_values = []

    for chemin_lr, chemin_hr in zip(chemins_lr, tqdm(chemins_hr, total=len(chemins_hr)-1, initial=0)):

        image_lr = Data_Utils.load(chemin_lr)
        image_hr = Data_Utils.load(chemin_hr)
        
        image_sr, execution_time = model.upscale(image_lr)
        
        nom_fichier_ = os.path.basename(chemin_lr)
        nom_fichier_ = nom_classe+"_"+nom_fichier_
        chemin_ = os.path.join(dossier_sr, nom_fichier_)
        chemins_sr.append(chemin_)

        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        
        resultats.append(image_sr)
        temps_execution.append(execution_time)

        image_hr = cv2.resize(image_hr, (image_sr.shape[1], image_sr.shape[0]))
        #les deux images doivent avoir la même talle pour le PSNR

        psnr = cv2.PSNR(image_hr, image_sr)
        psnr_values_classe.append(psnr)

        ifm = structural_similarity(image_hr.flatten(), image_sr.flatten())
        ifm_values.append(ifm)

        image_sr = cv2.filter2D(image_sr, -1, kernel)
        cv2.imwrite(chemin_, image_sr)

    psnr_values.append(psnr_values_classe)
    mean = np.mean(temps_execution) 
    print("\033[1mTemps d'execution moyen :\033[0m",mean, "s" )
    print("\033[1mPNSR moyen :\033[0m",np.mean(psnr_values_classe),"db" )
    print("\033[1mSSIM moyen :\033[0m",np.mean(ifm_values) )
    temps_execution_classe.append(mean)
    ifm_values_classe.append(ifm_values)


for i in range(len(classes)):
    plt.plot(psnr_values[i], label=classes[i].__name__)
plt.xlabel('Image')
plt.ylabel('PSNR')
plt.title('PSNR Comparison')
plt.legend()
plt.show()

for i in range(len(classes)):
    plt.plot(ifm_values_classe[i], label=classes[i].__name__)
plt.xlabel('Image')
plt.ylabel('SSIM')
plt.title('SSIM Comparison')
plt.legend()
plt.show()

plt.bar(nom_classes, temps_execution_classe)
plt.xlabel('Classe')
plt.ylabel('Temps d\'exécution moyen (s)')
plt.title('Temps d\'exécution moyen par méthodes')
plt.show()

