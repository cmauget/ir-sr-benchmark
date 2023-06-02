import cv2 #type:ignore
import numpy as np #type:ignore
import torch #type:ignore
from data_utils import Data_Utils
import models.models_utils.RRDBNet_arch as arch
import time


class ESRGAN:

    def __init__(self, model_path='models/models/RRDB_ESRGAN_x4.pth', device_='mps'):

        self.device = torch.device(device_)

        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

        print("Model", __name__, "loaded")
        

    def upscale(self, img):

        start_time = time.time()
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        with torch.no_grad():
            output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).clip(0, 255).round().astype(np.uint8)
        end_time = time.time()
        execution_time = end_time - start_time


        return output, execution_time


    