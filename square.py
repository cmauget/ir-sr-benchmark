from PIL import Image

def crop_image(image_path, output_folder):
    image = Image.open(image_path)
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

# Exemple d'utilisation
image_path = "hr2_image/AH012556.tif"
output_folder = "square/"
crop_image(image_path, output_folder)
