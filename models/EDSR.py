import cv2 #type: ignore
from data_utils import Data_Utils
import time

class EDSR:

    def __init__(self, path_ = "models/", mult_=4):
        if mult_==4:
            self.path=path_+"EDSR_x4.pb"
        else :
            self.path=path_+"ESDR_x2.pb"
            mult_ = 2

        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(self.path) 
        self.sr.setModel("edsr", mult_) 
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
    img = Data_Utils.load("image.jpg")
    model = EDSR()
    result, _ = model.upscale(img)
    resized = cv2.resize(img,dsize=None,fx=2,fy=2)
    liste_image = [img, result, resized]
    liste_titre = ["image", "sr", "opencv"]

    Data_Utils.graphe(liste_image, liste_titre)
    