import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
from pathlib import Path
import imageio
import json
from radius import get_radius
from measurement import get_mask
from sys import platform
from itertools import chain

if platform == "linux" or platform == "linux2":
    DELIMITER = "/"
elif platform == "win32":
    DELIMITER = "\\"

def parse(fname):
    a = fname.split("_")

    B = []

    for i in range(3):
        B.append(float(a[i*2 + 1].replace("uT", "")))

    exp = a[7]
    factor = 1
    if "us" in exp:
        factor = 1000

    exp = exp.replace("us", "").replace("ms", "")
    exp = float(exp) / factor

    return B, exp

def select_ROI(file):
    img = imageio.imread(file)

    cv2.namedWindow("select signal ROI", cv2.WINDOW_NORMAL)
    
    ROIsignal = cv2.selectROI("select signal ROI", img, False)
    cv2.destroyWindow("select signal ROI")
    print("Signal ROI selected:")

    ROIbackground = cv2.selectROI("select background ROI", img, False)
    cv2.destroyWindow("select background ROI")
    print("Background ROI selected:")

    return ROIsignal, ROIbackground

def getROI(path):
    ROIsignal = None
    ROIbackground = None

    with open(path + DELIMITER + "properties.json", "r") as f:
        properties = json.loads(f.read())

        if "ROIsignal" in properties and "ROIbackground" in properties:
            ROIsignal = properties["ROIsignal"]
            ROIbackground = properties["ROIbackground"]

    if ROIsignal == None and ROIbackground == None:
    
        files = Path(path).glob('*.jpg')

        for file in files:
            ROIsignal, ROIbackground = select_ROI(file)

            if ROIsignal != (0, 0, 0, 0) and ROIbackground != (0, 0, 0, 0):
                break

        properties["ROIsignal"] = ROIsignal
        properties["ROIbackground"] = ROIbackground

        with open(path + DELIMITER + "properties.json", "w") as f:
            f.write(json.dumps(properties))

    return ROIsignal, ROIbackground

def extract2(path):
    with open(path + DELIMITER + "properties.json", "r") as f:
        properties = json.loads(f.read())

    ROIsignal = properties["ROIsignal"]
    radius = properties["radius"]
    # factor = properties["factor"]
    factor = 1/2
    center = properties["center"]

    mask = get_mask(radius, factor)

    try:
        img = imageio.imread(path + DELIMITER + "uncrossed.jpg")
    except FileNotFoundError:
        img = imageio.imread(path + DELIMITER + "uncrossed.tif")
    img2 = np.copy(img)

    cv2.circle(img2, center, int(radius * factor), (255,0,0), 2)
    cv2.rectangle(img2, ROIsignal, (255,0,0), 2)
    cv2.imshow("Image", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if "ROIbackground" not in properties:
        cv2.namedWindow("select background ROI", cv2.WINDOW_NORMAL)
        ROIbackground = cv2.selectROI("select background ROI", img, False)
        cv2.destroyWindow("select background ROI")

        properties["ROIbackground"] = ROIbackground

        with open(path + DELIMITER + "properties.json", "w") as f:
            f.write(json.dumps(properties))

    else:
        ROIbackground = properties["ROIbackground"]

    B = []
    I = []

    files = Path(path).glob('*.jpg')
    files = chain(files, Path(path).glob('*.tif'))

    i = 0

    for file in files:
        print(i)
        i += 1
        img = imageio.imread(file)

        fname = os.path.basename(file).split(".")[0]
        if fname == "uncrossed":
            continue

        Bi, exp = parse(fname)
        B.append(Bi)

        imCrop = img[int(ROIsignal[1]):int(ROIsignal[1]+ROIsignal[3]), int(ROIsignal[0]):int(ROIsignal[0]+ROIsignal[2])]
        imBackground = img[int(ROIbackground[1]):int(ROIbackground[1]+ROIbackground[3]), int(ROIbackground[0]):int(ROIbackground[0]+ROIbackground[2])]

        avg_brightness_signal = np.average(imCrop[mask])
        avg_brightness_background = np.mean(imBackground)

        I.append((avg_brightness_signal - avg_brightness_background) / exp)

    return B, I, radius

def extract(path):
    ROIsignal, ROIbackground = getROI(path)

    B = []
    I = []

    files = Path(path).glob('*.jpg')
    files = chain(files, Path(path).glob('*.tif'))

    for file in files:
        img = imageio.imread(file)

        fname = os.path.basename(file).split(".")[0]
        if fname == "uncrossed":
            continue

        Bi, exp = parse(fname)
        B.append(Bi)

        imCrop = img[int(ROIsignal[1]):int(ROIsignal[1]+ROIsignal[3]), int(ROIsignal[0]):int(ROIsignal[0]+ROIsignal[2])]
        imBackground = img[int(ROIbackground[1]):int(ROIbackground[1]+ROIbackground[3]), int(ROIbackground[0]):int(ROIbackground[0]+ROIbackground[2])]

        avg_brightness_signal = np.mean(imCrop)
        avg_brightness_background = np.mean(imBackground)

        I.append((avg_brightness_signal - avg_brightness_background) / exp)

    return B, I

def draw_radius(pathname):
    try:
        path = Path(pathname + DELIMITER + "uncrossed.jpg")
        img = imageio.imread(path)
    except FileNotFoundError:
        path = Path(pathname + DELIMITER + "uncrossed.tif")
        img = imageio.imread(path)

    path = Path(pathname + DELIMITER + "properties.json")

    with open(path, "r") as f:
        properties = json.loads(f.read())

        if "ROIsignal_radius" in properties:
            ROIsignal = properties["ROIsignal_radius"]
        else:
            cv2.namedWindow("select radius ROI", cv2.WINDOW_NORMAL)
            ROIsignal = cv2.selectROI("select radius ROI", img, False)
            cv2.destroyWindow("select radius ROI")

    imCrop = img[int(ROIsignal[1]):int(ROIsignal[1]+ROIsignal[3]), 
        int(ROIsignal[0]):int(ROIsignal[0]+ROIsignal[2])]
    
    center, radius = get_radius(imCrop)

    with open(path, "w") as f:
        properties["ROIsignal_radius"] = ROIsignal
        properties["ROIsignal"] = [center[0]-radius+ROIsignal[0], center[1]-radius+ROIsignal[1],
                                   2*radius+1, 2*radius+1]
        properties["radius"] = radius
        properties["center"] = (center[0] + ROIsignal[0], center[1] + ROIsignal[1])
        f.write(json.dumps(properties))

    return radius

def get_immediate_subdirectories(a_dir):
    return [a_dir + DELIMITER + name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# ------------------------------------------------------------------------------

def main_single(path=None):
    if path == None:
        root = Tk()
        root.update() 
        
        path = askdirectory(title='Select Folder')
        root.destroy()

    # B0, I = extract2(path)
    try:
        B0, I, _ = extract2(path)
    except KeyError:
        B0, I = extract(path)

    B0 = np.array(B0)
    I = np.array(I)

    B = B0**2
    B = np.sum(B, axis=1)
    B = np.sqrt(B)

    sign = np.sign(B0[:, 0])
    if np.sum(np.abs(sign)) == 0:
        sign = np.sign(B0[:, 1])

    B *= sign

    fig, ax = plt.subplots()

    ax.plot(B, I, ".")
    ax.set_xlabel("uT")
    ax.set_ylabel("I/ms")

    plt.show()

    # draw_radius(path)

    return B, I

def plot_k(B, I, radiis):
    

    k = []

    for b, i in zip(B, I):
        mask = np.abs(b) >= 2000

        b = b[mask]
        b = np.abs(b)
        i = i[mask]

        A = np.hstack([b[:, None], np.ones_like(b)[:, None]])
        b = i[:, None]

        k.append(np.linalg.lstsq(A, b)[0][0])

    fig, ax = plt.subplots()

    ax.plot(radiis, k, ".")

    plt.show()

def main_multiple():
    root = Tk()
    root.update() 
    
    path = askdirectory(title='Select Folder')
    root.destroy()

    imdir = get_immediate_subdirectories(path)

    B0s, Is, radiis = [], [], []

    for dir in imdir:
        # try:
        #     B0, I = extract2(dir)
        # except:
        #     B0, I = extract(dir)
        # draw_radius(dir)
        B0, I, radius = extract2(dir)

        B0s.append(B0)
        Is.append(I)
        radiis.append(radius)# + " " + dir)

    Is = np.array(Is)
    B0s = np.array(B0s)

    B = B0s**2
    B = np.sum(B, axis=2)
    B = np.sqrt(B)
    B *= np.sign(B0s[:, :, 0])

    fig, ax = plt.subplots()

    ax.plot(B.T, Is.T, ".", label=radiis)
    ax.set_xlabel("uT")
    ax.set_ylabel("I/ms")

    # ax.set_yscale("log")

    ax.legend(loc="center")

    plt.show()

    print(B.shape, Is.shape)

    plot_k(B, Is, radiis)

if __name__ == "__main__":
    main_single()
    # main_multiple()