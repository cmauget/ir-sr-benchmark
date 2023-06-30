import cv2 #type: ignore
from utils.data_utils import Data_Utils
import time

class FSRCNN:

    def __init__(self, device_="", path_ = "models/models/FSRCNN_x4.pb"):
        self.path = path_
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(self.path) 
        self.sr.setModel("fsrcnn", 4) 
        print("Model", __name__, "loaded")

    def upscale(self, img):
        #print("Beginning upscaling...")

        start_time = time.time()
        res = self.sr.upsample(img)
        end_time = time.time()
        execution_time = end_time - start_time
       # print("Done in", execution_time, "s")
        return res, execution_time

    