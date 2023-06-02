import cv2 #type: ignore
import time

class ESPCN:

    def __init__(self, path_ = "models/models/ESPCN_x4.pb"):
        self.path = path_
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(self.path) 
        self.sr.setModel("espcn", 4) 
        print("Model", __name__, "loaded")

    def upscale(self, img):
        #print("Beginning upscaling...")

        start_time = time.time()
        res = self.sr.upsample(img)
        end_time = time.time()
        execution_time = end_time - start_time
       # print("Done in", execution_time, "s")
        return res, execution_time
