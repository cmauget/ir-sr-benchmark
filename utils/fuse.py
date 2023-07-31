import cv2 #type: ignore
import pywt #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
from mpl_toolkits.mplot3d import Axes3D #type: ignore


class Fuse:

    @staticmethod   
    def wavelet_fusion(thermal, visible, wavelet='db1', level=1, weight_thermal=0.5):

        coeffs_thermal = pywt.wavedec2(thermal, wavelet, level=level)
        coeffs_visible = pywt.wavedec2(visible, wavelet, level=level)

        fused_coeffs = []
        for (coeffs_thermal_level, coeffs_visible_level) in zip(coeffs_thermal, coeffs_visible):
            fused_coeffs_level = []
            for (coeffs_thermal_band, coeffs_visible_band) in zip(coeffs_thermal_level, coeffs_visible_level):
                fused_band = weight_thermal * coeffs_thermal_band + (1 - weight_thermal) * coeffs_visible_band
                fused_coeffs_level.append(fused_band)
            fused_coeffs.append(fused_coeffs_level)

        fused_image = pywt.waverec2(fused_coeffs, wavelet)

        fused_image = (fused_image - fused_image.min()) * 255 / (fused_image.max() - fused_image.min())
        fused_image = fused_image.astype('uint8')

        return fused_image

    @staticmethod  
    def additive_fusion(thermal, visible):

        thermal = cv2.resize(thermal, (visible.shape[1], visible.shape[0]))

        thermal_normalized = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX)

        fused_image = cv2.addWeighted(visible, 0.5, thermal_normalized, 0.5, 0)

        return fused_image

    @staticmethod  
    def multiplicative_fusion(thermal, visible):

        thermal_resized = cv2.resize(thermal, (visible.shape[1], visible.shape[0]))

        thermal_normalized = cv2.normalize(thermal_resized, None, 0, 1, cv2.NORM_MINMAX)

        fused_image = np.multiply(visible, thermal_normalized)

        return fused_image

    @staticmethod  
    def temperature_contours(thermal, visible,  temperature_threshold=200):

        _, thresholded = cv2.threshold(thermal, temperature_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_image = cv2.drawContours(visible, contours, -1, (0, 255, 0), 2)

        return contours_image

    @staticmethod
    def add_channel(thermal, visible):

        thermal_image = cv2.resize(thermal, (visible.shape[1], visible.shape[0]))
        merged_image = cv2.merge((visible[:, :, 0:3], thermal_image))

        return merged_image

    @staticmethod  
    def temperature_3d_representation(thermal):

        thermal = cv2.resize(thermal, (thermal.shape[1]//4, thermal.shape[0]//4))

        y_coords, x_coords = np.mgrid[0:thermal.shape[0], 0:thermal.shape[1]]
        temperatures = thermal.flatten()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        ax.scatter(x_coords.flatten(), y_coords.flatten(), temperatures, c=temperatures, cmap='jet', s=0.5)

        cbar = plt.colorbar(ax.scatter([], [], [], c=[], cmap='jet'))
        cbar.set_label('Température')

        plt.show()


if __name__ == "__main__":

    list_func = [Fuse.additive_fusion, Fuse.multiplicative_fusion, Fuse.wavelet_fusion, Fuse.temperature_contours]

    thermal_image_path = '../IMG/fiss_irt.png'
    visible_image_path = '../IMG/fiss_col.png'

    thermal = cv2.imread(thermal_image_path,0)
    visible = cv2.imread(visible_image_path,0)

    for func in list_func:
        fused_image = func(thermal, visible)
        print(func.__name__+".png")
        cv2.imwrite(func.__name__+".png", fused_image)

