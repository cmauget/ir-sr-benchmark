from utils.utils import Data_Utils as d
from PIL import Image as I #type: ignore


def crop_image(image_path, output_folder):
    image = I.open(image_path)
    image_width, image_height = image.size
    square_size = 96

    # Vérifie si l'image est plus petite que la taille du carré
    if image_width < square_size or image_height < square_size:
        print("L'image est trop petite pour être découpée en carrés de 96x96 pixels.")
        return

    # Calcule le nombre de carrés sur chaque axe
    num_rows = image_height // square_size
    num_cols = image_width // square_size

    # Parcourt chaque carré
    for row in range(num_rows):
        for col in range(num_cols):
            # Calcule les coordonnées de découpe du carré
            left = col * square_size
            top = row * square_size
            right = left + square_size
            bottom = top + square_size

            # Découpe le carré de l'image
            square = image.crop((left, top, right, bottom))

            # Enregistre le carré dans un fichier
            filename = f"square3_{row}_{col}.jpg"
            output_path = f"{output_folder}/{filename}"
            square.save(output_path)

            print(f"Le carré {filename} a été découpé et enregistré avec succès.")


val = input("convert (1), crop (2), resize(3), square(4) : ")

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
    d.crop('IMG_3_VIS.tif', 'IMG_3_VIS_crop.tif', crop_coords)

elif (val == 3):

    dossier_hr = "../Images/hr_image"
    dossier_hr2 = "../Images/hr2_image"

    dossier_lr = "../Images/lr_image"

    d.create_folder(dossier_hr2)
    d.create_folder(dossier_lr)

    d.resize(dossier_hr, dossier_hr2, dossier_lr)

elif (val == 4):

    image_path = "hr2_image/AH012556.tif"
    output_folder = "square/"
    crop_image(image_path, output_folder)

else :

    print("Incorrect value")
