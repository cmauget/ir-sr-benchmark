import os
import cv2 #type:ignore
from utils.utils import Data_Utils as d
from utils.utils import Image
import matplotlib.pyplot as plt #type:ignore
import numpy as np #type:ignore
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

verbose = True

classes = [EDSR, ESPCN, FSRCNN, LapSRN, PSRGAN, ESRGAN2, ESRGAN]


dossier_lr = "../Images/lr_image"
dossier_sr = "../Images/sr_image"
dossier_hr = "../Images/hr2_image"

chemins_lr = []
chemins_hr = []
chemins_sr = []

resultats = []
psnr_values = []
temps_execution_classe = []
ssim_values_classe = []
mean_psnr_values = []
mean_ssim_values = []
liste_image = []
liste_titre = []
nom_classes = []

blur = (1,1)

d.create_folder(dossier_sr)

chemins_lr = d.find_path(dossier_lr)

chemins_hr = d.find_path(dossier_hr)

seed = int(time.time())

random.seed(seed) 
random.shuffle(chemins_hr)

random.seed(seed) 
random.shuffle(chemins_lr)

for classe in classes:  
    model = classe()
    nom_classe = classe.__name__
    nom_classes.append(nom_classe)
    psnr_values_classe = []
    temps_execution = []
    ssim_values = []

    for chemin_lr, chemin_hr in zip(chemins_lr, tqdm(chemins_hr, total=len(chemins_hr), initial=1)):

        image_lr = Image.load(chemin_lr)
        image_hr = Image.load(chemin_hr)

        print(type(image_hr))
        
        image_sr, execution_time = model.upscale(image_lr)
        
        nom_fichier_ = os.path.basename(chemin_lr)
        nom_fichier_ = nom_classe+"_"+nom_fichier_
        chemin_ = os.path.join(dossier_sr, nom_fichier_)
        chemins_sr.append(chemin_)
        
        resultats.append(image_sr)
        temps_execution.append(execution_time)

        if verbose:
            image_hr = cv2.resize(image_hr, (image_sr.shape[1], image_sr.shape[0]))

            psnr = cv2.PSNR(image_hr, image_sr)
            psnr_values_classe.append(psnr)

            ssim = structural_similarity(image_hr.flatten(), image_sr.flatten())
            ssim_values.append(ssim)

        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        #image_sr = cv2.filter2D(image_sr, -1, kernel)
        cv2.imwrite(chemin_, image_sr)


    if verbose:

        liste_image.append(image_sr)
        liste_titre.append(nom_classe)
        psnr_values.append(psnr_values_classe)
        mean = np.mean(temps_execution) 
        temps_execution_classe.append(mean)
        ssim_values_classe.append(ssim_values)
        mean_psnr_values.append(np.mean(psnr_values_classe))
        mean_ssim_values.append(np.mean(ssim_values))
        
        print("\033[1mTemps d'execution moyen :\033[0m",mean, "s" )
        print("\033[1mPNSR moyen :\033[0m",np.mean(psnr_values_classe),"db" )
        print("\033[1mSSIM moyen :\033[0m",np.mean(ssim_values) )


if verbose:

    liste_image.insert(0, image_hr)
    liste_image.insert(0, image_lr)
    liste_titre.insert(0, "Image haute \n résolution") 
    liste_titre.insert(0, "Image basse \n résolution")

    d.graphe(liste_image, liste_titre)

    for i in range(len(classes)):
        plt.plot(psnr_values[i], label=nom_classes[i])
    plt.xticks(np.arange(len(nom_classes)))
    plt.xlabel('Image')
    plt.ylabel('PSNR')
    plt.title('PSNR Comparison')
    plt.legend()
    plt.show()

    for i, methode in enumerate(nom_classes):
        plt.scatter(mean_psnr_values[i], mean_ssim_values[i], label = methode)
    plt.xlabel('PSNR (dB)')
    plt.ylabel('SSIM')
    plt.title('PSNR/SSIM')
    plt.legend()
    plt.show()

    for i in range(len(classes)):
        plt.plot(ssim_values_classe[i], label=classes[i].__name__)
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

