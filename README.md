# ir-sr-benchmark
A python tool to easily implement and benchmark super resolution solutions, especially for thermal images

# Setup

First, you will need to download the zip, extract it, and move to the root of this repo using the _cd_ command. It is recommended to use it in a venv (using conda for example). To install the dependencies, you will need to run :

    pip install -r requirements.txt
  

# How to use it  

## Using the GUI

You just have to run the following command :

    streamlit run streamlit_app/Home.py

## Python scripts  

You will need to create a folder where you want and use the name you want, then you have to change the dossier_hr value in the following scripts with the path to this folder. You then need to put your thermal images in it. 

You will then have the choice to run multiple python scripts :

### utils_func.py

    python3 utils_func.py

This script allows you to make multiple operation such as converting thermal images to RGB and resizing the high resolution images.


### main.py

    python3 main.py

This is the script which runs the super resolution of the images and creates all the graphs regarding the results


# Methods 

This program currently implement 6 methods defined in the model folder. Each method has its own class with a constructor and an upscale() function taking an image (already loaded) as an input. If you want to add other methods, you can simply create a new class following the same naming scheme and add that class in the main.py script. You will have to import it and add it in the class list.

# Code organisation

```
📂 ir-sr-benchmark/ 
├── 📂 models/
|       ├── 📂 models/ #models files
|       ├── 📜 EDSR.py
|       |...
├── 📂 utils/
|       ├── 📜 utils.py #functions for Image, Folder and Classification
|       ├── 📜 fuse.py #functions fo Fusion
|       |...
├── 📂 streamlit_app/
|       ├── 📜 Home.py #entry point for the GUI
|       ├── 📂 pages/ #rest of the pages for the GUI 
├── 📜 main.py
├── 📜 utils_func.py
│...
```
