import imageio
from labControl.tweezers.tweezersTCP import Magnetic, Optical, calculateField
from flycapture import Camera
import numpy as np
from time import sleep
from tkinter import messagebox
import json
import cv2
import os
from radius import get_radius
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter import messagebox

FPS = input("Enter FPS:\n")
exposure = input("Enter exposure: \n")
current = input("Enter current:\n")
mag = input("Enter objective magnification:\n")

properties = {}
FPS = int(FPS)
properties["FPS"] = FPS
properties["current"] = current
properties["mag"] = mag
properties["exposure"] = exposure

properties = json.dumps(properties)

root = Tk()
root.update() 

path = askdirectory(title='Select Folder')
root.destroy()

with open(path + "\\properties.json", "w") as f:
    f.write(properties)

# TMP_PATH = "E:\\Martin\\Kapljice\\meritve\\"

opt = Camera()

i = 0

while True:
    messagebox.showinfo("Press enter to take a picture", "Press enter to take a picture")

    fname = "{}".format(i)
    fpath = path + "\\" + fname + '.jpg'
    opt.setFolder(path)

    print("Saving image to file {}".format(fpath))

    opt.takeImage(fname)
    sleep(2/FPS)

    i += 1