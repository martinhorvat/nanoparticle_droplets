import numpy as np
from datetime import datetime
from measurement import run_multiple
import os
import json

# PATH SETUP

PATH = "E:\\Martin\\Kapljice\\meritve"
IP = "141.255.216.200"

today = datetime.now()
today = today.strftime('%d-%m-%Y')

paths = []
n = input("Enter measurement number\n")
# measurement_names = [n+"_Bx", n+"_By", n+"_Bxy", 
#                      n+"_Bx_hist", n+"_By_hist", n+"_Bxy_hist"]
measurement_names = [n+"_Bx", n+"_By"]

for measurement_name in measurement_names:
    path = (PATH + "\\" + today + "\\" + measurement_name + "\\")

    if not os.path.exists(path):
        os.makedirs(path)

    paths.append(path)

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

# Ix_MAX = 2.5
# Iy_MAX = 2.5
# Ix_STEPS = 20
# Iy_STEPS = 20

# Ix = np.linspace(-Ix_MAX, Ix_MAX, Ix_STEPS)
# Ix2 = np.linspace(-1, 1, 20)
# Ix3 = np.linspace(-0.2, 0.2, 10)
# Ix = np.hstack([Ix, Ix2, Ix3])
# Ix = np.sort(Ix)

# mask = Ix > 0
# Ix1 = np.hstack([Ix[mask], np.flip(Ix[np.logical_not(mask)])])
# Ix2 = np.hstack([Ix[mask], np.flip(Ix[mask])[1:]])

# Ixs = [Ix1, np.zeros_like(Ix1), Ix1, Ix2, np.zeros_like(Ix2), Ix2]

# Iy = np.linspace(-Iy_MAX, Iy_MAX, Iy_STEPS)
# Iy2 = np.linspace(-1, 1, 20)
# Iy3 = np.linspace(-0.2, 0.2, 10)
# Iy = np.hstack([Iy, Iy2, Iy3])
# Iy = np.sort(Iy)

# mask = Iy > 0
# Iy1 = np.hstack([Iy[mask], np.flip(Iy[np.logical_not(mask)])])
# Iy2 = np.hstack([Iy[mask], np.flip(Iy[mask])[1:]])

# Iys = [np.zeros_like(Iy1), Iy1, Iy1, np.zeros_like(Iy2), Iy2, Iy2]

# Iz1 = np.zeros_like(Ix1)
# Iz2 = np.zeros_like(Ix2)

# Izs = [Iz1]*3 + [Iz2]*3

# types = ["normal"]*3 + ["hysteresis"]*3

# ---------------------------------

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
Ix1 = np.hstack([Ix[mask], np.flip(Ix[np.logical_not(mask)])])

Ixs = [Ix1, np.zeros_like(Ix1)]

Iy = np.linspace(-Iy_MAX, Iy_MAX, Iy_STEPS)
Iy2 = np.linspace(-1, 1, 20)
Iy3 = np.linspace(-0.2, 0.2, 10)
Iy = np.hstack([Iy, Iy2, Iy3])
Iy = np.sort(Iy)

mask = Iy > 0
Iy1 = np.hstack([Iy[mask], np.flip(Iy[np.logical_not(mask)])])

Iys = [np.zeros_like(Iy1), Iy1]

Iz1 = np.zeros_like(Ix1)

Izs = [Iz1]*2

types = ["normal"]*2

# ---------------------------------

run_multiple(paths, properties, Ixs, Iys, Izs, IP, FPS, types)

