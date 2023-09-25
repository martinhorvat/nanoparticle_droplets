import numpy as np
from datetime import datetime
from measurement import measure, detect_roi
import os
import json
from tkinter import messagebox
from measurement import prepare
import imageio

# PATH SETUP

PATH = "E:\\Martin\\Kapljice\\meritve"
IP = "141.255.216.193"

today = datetime.now()

today = today.strftime('%d-%m-%Y')

while True:
    measurement_name = input("Enter measurement name\n")
    IMG_PATH = PATH + "\\" + today + "\\" + measurement_name + "\\"

    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
        break
    else:
        print("Measurement already exists!")

# ---------------------------------

# PROPERTIES

mag = input("Enter objective lens magnification\n")

properties = {"objective" : mag}
properties["comment"] = input("Input comment\n")

FPS = input("Enter FPS:\n")
FPS = int(FPS)

properties["FPS"] = FPS

# ---------------------------------

# MEASUREMENT SETUP

EXPOSURE_BASE = 5
Ix_MAX = 2.5
Iy_MAX = 2.5
Ix_STEPS = 20
Iy_STEPS = 20

Ix = np.linspace(-Ix_MAX, Ix_MAX, Ix_STEPS)
Ix2 = np.linspace(-1, 1, 20)
Ix3 = np.linspace(-0.2, 0.2, 10)
Ix = np.hstack([Ix, Ix2, Ix3])
Ix = np.sort(Ix)

mask = Ix > 0
Ix = np.hstack([Ix[mask], np.flip(Ix[np.logical_not(mask)])])

Iy = np.zeros_like(Ix)

# Iy = np.linspace(-Iy_MAX, Iy_MAX, Iy_STEPS)
# Iy2 = np.linspace(-1, 1, 20)

# Iy = np.hstack([Iy, Iy2])
# Iy = np.sort(Iy)

# Ix = np.zeros_like(Iy)

Iz = np.zeros_like(Ix)

# ---------------------------------

img, ROIsignal, mask, properties = prepare(FPS, properties)

with open(IMG_PATH + "properties.json", "w+") as f:
    f.write(properties)

imageio.imwrite(IMG_PATH + "uncrossed.jpg", img)

messagebox.showinfo("Cross polarizers", "Cross polarizers")

measure(Ix, Iy, Iz, IMG_PATH, IP, FPS, ROIsignal, mask)