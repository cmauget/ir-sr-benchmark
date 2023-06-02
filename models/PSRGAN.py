import time
import numpy as np #type:ignore
import torch #type:ignore
from models.models_utils.network_msrresnet import D_msrresnet0 as net

class PSRGAN:

    def __init__(self, model_path='models/models/75000_G.pth', device_='mps'):

        self.device = torch.device(device_) # cuda / cpu

        self.model = net(in_nc=3, out_nc=3, nc=64, nb=16, upscale=4)
        self.model.load_state_dict(torch.load(model_path), strict=False)

        self.model.eval()

        for _ , v in self.model.named_parameters():
            v.requires_grad = False

        self.model = self.model.to(self.device)

        print("Model", __name__, "loaded")
        

    def upscale(self, img):

        start_time = time.time()
        img = np.float32(img/255.)
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
        img = img.to(self.device)
        output = self.model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).clip(0, 255).round().astype(np.uint8)

        end_time = time.time()
        execution_time = end_time - start_time

        return output, execution_time

