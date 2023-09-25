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

RELAXATION_SLEEP = 2
SLEEP_TIME = lambda fps: 2/fps
gen_fname = lambda x, y, z, e: 'x_{:.0f}uT_y_{:.0f}uT_z_{:.0f}uT_exp_{:.0f}us'.format(x, y, z, e*1000)

class Sign():
    def __init__(self) -> None:
        self.old_x = 0
        self.old_y = 0
        self.old_z = 0

    def changed_sign(self, x, y, z):
        if(x*self.old_x < 0 or y*self.old_y < 0 or z*self.old_z < 0):
            self.old_x = x
            self.old_y = y
            self.old_z = z

            return True
        
        self.old_x = x
        self.old_y = y
        self.old_z = z
        
        return False
    
    def is_rising(self, x, y, z):
        if(self.old_x <= x and self.old_y <= y and self.old_z <= z):
            self.old_x = x
            self.old_y = y
            self.old_z = z

            return True
        
        self.old_x = x
        self.old_y = y
        self.old_z = z
        
        return False

def get_mask(radius, factor):
    a = np.arange(2*radius+1) - radius
    x, y = np.meshgrid(a, a)
    r = np.sqrt(x**2 + y**2)
    mask = r <= radius*factor

    return mask

def detect_roi(img, factor=2/3):
    center, radius = get_radius(img)
    
    ROI = [center[0]-radius, center[1]-radius, 2*radius+1, 2*radius+1]
    print(ROI)

    mask = get_mask(radius, factor)

    img2 = np.copy(img)

    cv2.circle(img2, center, int(radius*factor), (255,0,0), 2)
    cv2.imshow("Image", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ROI, mask, center, radius

def setDC(mag):
    mag.setDC('x')
    mag.setDC('y')
    mag.setDC('z')

def setDCAmplitude(mag, x, y, z):
    mag.setDCAmplitude('x', x)
    mag.setDCAmplitude('y', y)
    mag.setDCAmplitude('z', z)

def calculateAllFields(x, y, z):
    Bx = calculateField(x, "x") * 1000
    By = calculateField(y, "y") * 1000
    Bz = calculateField(z, "z") * 1000

    return Bx, By, Bz

def disable(mag):
    mag.disable('x')
    mag.disable('y')
    mag.disable('z')

def read_img(path, fps):
    i = 1

    try:
        img = imageio.imread(path)
    except FileNotFoundError:
        if i > 3:
            raise

        print("File not found... retrying {}".format(i))
        i += 1
        sleep(SLEEP_TIME(fps))

    return img

def read_and_mask(path, ROIsignal, mask, fps):
    img = read_img(path, fps)

    imCrop = img[int(ROIsignal[1]):int(ROIsignal[1]+ROIsignal[3]),
            int(ROIsignal[0]):int(ROIsignal[0]+ROIsignal[2])]
    imCrop[~mask] = 0

    return imCrop

def measure(Ix, Iy, Iz, IMG_PATH, IP, FPS, ROIsignal, mask):
    sign = Sign()
    mag = Magnetic(IP)
    opt = Camera()

    setDC(mag)

    max_exposure = 1000 / FPS - 1
    exposure = max_exposure

    for x, y, z in zip(Ix, Iy, Iz):
        setDCAmplitude(mag, x, y, z)
        mag.synchronize()
        
        sleep(RELAXATION_SLEEP)

        if(sign.changed_sign(x, y, z)):
            sleep(20)
            exposure = max_exposure
        
        Bx, By, Bz = calculateAllFields(x, y, z)

        opt.setFolder(IMG_PATH)  

        while True:
            fname = gen_fname(Bx, By, Bz, exposure)
            fpath = IMG_PATH + fname + ".jpg"

            print(opt.setExposure(round(exposure, 3)))

            sleep(SLEEP_TIME(FPS))
            opt.takeImage(fname)
            sleep(SLEEP_TIME(FPS))

            imCrop = read_and_mask(fpath, ROIsignal, mask, FPS)

            if np.amax(imCrop) >= 255:
                exposure *= 0.8
                os.remove(fpath)
            else:
                print("Saving image to file {}".format(fpath))
                break

    disable(mag)
    mag.synchronize()

def measure_hysteresis(Ix, Iy, Iz, IMG_PATH, IP, FPS, ROIsignal, mask):
    sign = Sign()
    mag = Magnetic(IP)
    opt = Camera()

    setDC(mag)

    max_exposure = 1000 / FPS - 1
    exposure = max_exposure

    for x, y, z in zip(Ix, Iy, Iz):
        setDCAmplitude(mag, x, y, z)
        mag.synchronize()

        sleep(RELAXATION_SLEEP)

        Bx, By, Bz = calculateAllFields(x, y, z)

        rising = sign.is_rising(x, y, z)

        opt.setFolder(IMG_PATH) 

        while True:
            fname = gen_fname(Bx, By, Bz, exposure)

            if not rising:
                fname += "_falling"

            fpath = IMG_PATH + fname + ".jpg"

            if exposure >= max_exposure:
                print(opt.setExposure(round(max_exposure, 3)))
            else:
                print(opt.setExposure(round(exposure, 3)))

            sleep(SLEEP_TIME(FPS))
            opt.takeImage(fname)
            sleep(SLEEP_TIME(FPS))

            imCrop = read_and_mask(fpath, ROIsignal, mask, FPS)

            if rising:
                if np.amax(imCrop) >= 255:
                    os.remove(fpath)
                    exposure *= 0.8
                else:
                    print("Saving image to file {}".format(fpath))
                    break
            else:
                os.remove(fpath)

                if exposure >= max_exposure or np.amax(imCrop) >= 255:
                    if exposure >= max_exposure and np.amax(imCrop) < 255:
                        exposure = max_exposure
                    else:
                        exposure *= 0.8

                    fname = gen_fname(Bx, By, Bz, exposure) + "_falling"
                    fpath = IMG_PATH + fname + ".jpg"

                    print(opt.setExposure(round(exposure, 3)))

                    sleep(SLEEP_TIME)
                    opt.takeImage(fname)
                    sleep(SLEEP_TIME)

                    print("Saving image to file {}".format(fpath))
                    
                    break

                else:
                    exposure *= 1/0.8
                    
    disable(mag)
    mag.synchronize()

def prepare(FPS, properties):
    TMP_PATH = "E:\\Martin\\Kapljice\\meritve\\"

    opt = Camera()

    fname = "uncrossed"
    fpath = TMP_PATH + fname + '.jpg'
    opt.setFolder(TMP_PATH)

    print("Saving image to file {}".format(fpath))
    
    opt.takeImage(fname)
    sleep(2/FPS)

    img = read_img(fpath, FPS)
    os.remove(fpath)

    factor = 2/3
    ROIsignal, mask, center, radius = detect_roi(img, factor)

    properties["ROIsignal"] = ROIsignal
    properties["factor"] = factor
    properties["center"] = center
    properties["radius"] = radius

    properties = json.dumps(properties)

    return img, ROIsignal, mask, properties

def run_multiple(paths, properties, Ixs, Iys, Izs, IP, FPS, types):
    img, ROIsignal, mask, properties = prepare(FPS, properties)

    messagebox.showinfo("Cross polarizers", "Cross polarizers")

    for path, Ix, Iy, Iz, meas_type in zip(paths, Ixs, Iys, Izs, types):
        with open(path + "properties.json", "w") as f:
            f.write(properties)

        imageio.imwrite(path + "uncrossed.jpg", img)

        if meas_type == "normal":
            measure(Ix, Iy, Iz, path, IP, FPS, ROIsignal, mask)
        elif meas_type == "hysteresis":
            measure_hysteresis(Ix, Iy, Iz, path, IP, FPS, ROIsignal, mask)
        else:
            raise Exception
        
        sleep(10)
