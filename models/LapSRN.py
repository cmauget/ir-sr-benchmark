import cv2 #type: ignore
from data_utils import Data_Utils
import time

class LapSRN:

    def __init__(self, path_ = "models/models/LapSRN_x4.pb"):
        self.path = path_
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(self.path) 
        self.sr.setModel("lapsrn", 4) 
        print("Model", __name__, "loaded")

    def upscale(self, img):
        #print("Beginning upscaling...")

        start_time = time.time()
        res = self.sr.upsample(img)
        end_time = time.time()
        execution_time = end_time - start_time
        #print("Done in", execution_time, "s")
        return res, execution_time

if __name__ == "__main__" :

    lim_x = [0 ,800]
    lim_y = [0,800]
    div = 2
    img = Data_Utils.load("image2.tif")
    model = LapSRN()
    result, _ = model.upscale(img)
    resized = cv2.resize(img,dsize=None,fx=2,fy=2)
    liste_image = [img, result, resized]
    liste_titre = ["image", "sr", "opencv"]

    Data_Utils.graphe(liste_image, liste_titre)

    cv2.imwrite("image2.jpg", result)
    