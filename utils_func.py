from utils.data_utils import Data_Utils as d

val = input("convert (1), crop (2), resize(3) : ")

try:
    val = int(val)
except ValueError :
    val=0

if (val == 1):

    dossier_ir = "ir_image"
    dossier_hr = "gray_image"

    d.convert(dossier_ir, dossier_hr)

elif (val == 2):

    crop_coords = [1000, 1500, 1500, 2000]
    d.crop_image('IMG_3_VIS.tif', 'IMG_3_VIS_crop.tif', crop_coords)

elif (val == 3):

    dossier_hr = "gray_image"
    dossier_hr2 = "hr2_image"

    dossier_lr = "lr_image"

    d.create_folder(dossier_hr2)
    d.create_folder(dossier_lr)

    d.resize(dossier_hr, dossier_hr2, dossier_lr)

else :

    print("Incorrect value")
