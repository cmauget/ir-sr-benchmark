import os
import cv2 #type:ignore
from data_utils import Data_Utils
import matplotlib.pyplot as plt #type:ignore
import numpy as np #type:ignore
from skimage.metrics import normalized_mutual_information #type:ignore
from skimage.metrics import structural_similarity #type:ignore
from EDSR import EDSR
from ESPCN import ESPCN
from FSRCNN import FSRCNN
from LapSRN import LapSRN
from ESRGAN import ESRGAN
from ESRGAN2 import ESRGAN2
from models.PSRGAN import PSRGAN
from tqdm import tqdm #type:ignore
import random
import time

if __name__ == "__main__" :

    lim_x = [0 ,800]
    lim_y = [0,800]
    div = 2

    img = Data_Utils.load("image.jpg")
    model = ESRGAN()
    result, _ = model.upscale(img)
    resized = cv2.resize(img,dsize=None,fx=2,fy=2)
    liste_image = [img, result, resized]
    liste_titre = ["image", "sr", "opencv"]

    Data_Utils.graphe(liste_image, liste_titre)

    cv2.imwrite("image2.jpg", result)