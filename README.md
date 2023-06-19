# ir-sr-benchmark
A python tool to easily implement and benchmark super resolution solutions, especially for thermal images

## Setup

First, you will need to download the zip, extract it, and move to the root of this repo using the _cd_ command. It is recommended to use it in a venv (using conda for example). To install the dependencies, you will need to run :

    pip install -r requirements.txt
  

# How to use it  

You will need to create the hr_image folder as below (if you do not want to use this folder, make sure to change the 'dossier_hr' value in the code:

```
📂 ir-sr-benchmark # this is root
├── 📂 hr_image/
|       ├── 📜 image1.jpg
|       |...
│...
```  
You will then have the choice to run multiple python scripts :

## utils.py

    python3 utils.py

This script allows you to make multiple operation such as converting thermal images to RGB and resizing the high resolution images.


## main.py

    python3 main.py

This is the script which runs the super resolution of the images and creates all the graphs regarding the results

## fuse.py

    python3 fuse.py

This script offers different methods of image fusion

# Methods 

This program currently implement 6 methods defined in the model folder. Each method has its own class with a constructor and an upscale() function taking an image (already loaded) as an input. If you want to add other methods, you can simply create a new class following the same naming scheme and add that class in the main.py script. You will have to import it and add it in the class list.
