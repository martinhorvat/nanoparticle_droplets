import os
import json
import cv2
import imageio
import numpy as np
from measurement import get_mask
from tkinter import messagebox

class Slika:
    def __init__(self, path, exposure):
        self.path = path
        self.exposure = exposure

        self.original_image = imageio.imread(path)

        try:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = self.original_image

        if gray.dtype == 'uint16':
            gray = (gray/256).astype('uint8')
        
        self.original_image = gray

        self.reset()

    def clear_image(self):
        self.show_image = np.copy(self.gray)

    def reset(self):
        self.gray = np.copy(self.original_image)
        self.gaussian_image = np.copy(self.gray)
        _, self.thresh = cv2.threshold(self.gaussian_image, 0,
                                  255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.clear_image()

        self.roi = [0, 0, 
                    self.original_image.shape[1], 
                    self.original_image.shape[0]]

    def apply_gaussian(self, size=None):
        if size == None:
            size=(5,5)

        self.gaussian_image = cv2.GaussianBlur(self.gaussian_image, size, 0)
        _, self.thresh = cv2.threshold(self.gaussian_image, 0,
                                  255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # self.thresh = 255 - self.thresh

        self.show_image = np.copy(self.gaussian_image)

        return self.gaussian_image
    
    def open_and_close(self, kernel=None):
        if kernel == None:
            kernel=np.ones((7, 7), np.uint8)

        self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel)
        self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel)

        self.show_image = np.copy(self.thresh)

        return self.thresh

    def fit_radius(self):
        contours, _ = cv2.findContours(self.thresh, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(c) for c in contours]
        sorted_areas = np.sort(areas)
        cnt = contours[areas.index(sorted_areas[-1])] #the biggest contour

        (x_axis,y_axis), radius = cv2.minEnclosingCircle(cnt)
        
        self.radius = int(radius)
        self.center = [int(x_axis),int(y_axis)]

        self.show_image = np.copy(self.gray)
        cv2.circle(self.show_image, self.center, self.radius, (255,0,0), 2)

        newROI = [self.center[0]-self.radius, self.center[1]-self.radius, 
                  2*self.radius+1, 2*self.radius+1]
        self.apply_roi(newROI)

        self.center[0], self.center[1] = self.radius, self.radius

        return self.radius, self.center

    def average_intensity(self, ROIbackground, factor=0.9):
        mask = get_mask(self.radius, factor)

        imCrop = np.copy(self.gray)
        imBackground = self.original_image[int(ROIbackground[1]):int(ROIbackground[1]+ROIbackground[3]), int(ROIbackground[0]):int(ROIbackground[0]+ROIbackground[2])]
        imCrop = imCrop[mask]

        avg_brightness_signal = np.mean(imCrop)
        avg_brightness_background = np.mean(imBackground)

        self.avg_I = avg_brightness_signal - avg_brightness_background

        return self.avg_I / self.exposure * 1000, np.std(imCrop)

    
    def show(self):
        cv2.imshow("Image", self.show_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_roi(self, ROI):
        self.roi[0] = self.roi[0] + ROI[0]
        self.roi[1] = self.roi[1] + ROI[1]

        self.roi[2], self.roi[3] = ROI[2], ROI[3]

        x1, x2 = ROI[0], ROI[0] + ROI[2] 
        y1, y2 = ROI[1], ROI[1] + ROI[3] 

        self.gaussian_image = self.gaussian_image[y1:y2, x1:x2]
        self.show_image = self.show_image[y1:y2, x1:x2]
        self.thresh = self.thresh[y1:y2, x1:x2]
        self.gray = self.gray[y1:y2, x1:x2]

    def distance(self, wire_x):
        center_x = self.center[0]

        distance = wire_x - (center_x + self.roi[0])
        distance = np.abs(distance)

        return distance

    def get_avg_background(self):
        cv2.namedWindow("select background ROI", cv2.WINDOW_NORMAL)
        ROIbackground = cv2.selectROI("select background ROI", 
                                      self.original_image, False)
        cv2.destroyWindow("select background ROI")

        imBackground = self.original_image[int(ROIbackground[1]):int(ROIbackground[1]+ROIbackground[3]), 
                       int(ROIbackground[0]):int(ROIbackground[0]+ROIbackground[2])]

        avg = np.average(imBackground)

        return avg
        

class Slika_multi_exposure(Slika):
    def __init__(self, paths):
        self.im_names = [path.split("/")[-1].split(".")[0] for path in paths]
        grays = []

        for path in paths:
            img = imageio.imread(path)

            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except cv2.error:
                gray = img

            if gray.dtype == 'uint16':
                gray = (gray/256).astype('uint8')

            grays.append(gray)
        
        self.original_images = grays
        self.original_image = self.get_max_exposure_img()

        self.reset()

    def distance(self, wire_x):
        center_x = self.center[0]

        distance = wire_x - center_x
        distance = np.abs(distance)

        return distance

    def get_max_exposure_img(self):
        self.exposures = [im_name.split("_")[-1][0:-2] for im_name in self.im_names]

        inds = np.flip(np.argsort(self.exposures))
        self.exposures = [self.exposures[i] for i in inds]
        self.original_images = [self.original_images[i] for i in inds]

        return self.original_images[0]

    def get_optimal_image(self, ROI):
        x1, x2 = ROI[0], ROI[0] + ROI[2] 
        y1, y2 = ROI[1], ROI[1] + ROI[3] 

        for i, original_image in enumerate(self.original_images):
            original_image = original_image[y1:y2, x1:x2]

            if np.amax(original_image) < 255:
                break

        return self.original_images[i], int(self.exposures[i])

    def apply_roi(self, ROI):
        self.original_image, self.exposure = self.get_optimal_image(ROI)
        self.ROI = ROI

        super().apply_roi(ROI)

    def average_intensities(self, ROIbackground, factor=0.9):
        intensities = []
        for exposure, img in zip(self.exposures, self.original_images):
            self.reset()
            self.original_image = img
            self.apply_roi(self.ROI)

            mask = get_mask(self.radius, factor)

            imCrop = np.copy(self.gray)
            imBackground = self.original_image[int(ROIbackground[1]):int(ROIbackground[1]+ROIbackground[3]), int(ROIbackground[0]):int(ROIbackground[0]+ROIbackground[2])]
            imCrop = imCrop[mask]

            avg_brightness_signal = np.mean(imCrop)
            avg_brightness_background = np.mean(imBackground)

            print(exposure)

            intensities.append((avg_brightness_signal - avg_brightness_background) / exposure * 1000)

        return intensities

class Zica:
    def __init__(self, path):
        self.path = path

        with open(path + "/properties.json", "r") as f:
            self.properties = json.loads(f.read())

        if os.path.exists(path + "/roi.json"):
            with open(path + "/roi.json", "r") as f:
                self.roi = json.loads(f.read())

        else:
            self.roi = {}

    def list_images(self):
        images = []

        for f in os.listdir(self.path):
            if (f.endswith(".jpg") or f.endswith(".tif")) and "wire" not in f:
                images.append(f.split(".")[0])

        return images

    def set_roi(self):
        for im_name, img_container in self.images.items():
            if not im_name in self.roi:
                img = img_container.original_image

                cv2.namedWindow("select signal ROI", cv2.WINDOW_NORMAL)
    
                ROIsignal = cv2.selectROI("select signal ROI", 
                                          img, False)
                cv2.destroyWindow("select signal ROI")

                ROIbackground = cv2.selectROI("select background ROI", 
                                              img, False)
                cv2.destroyWindow("select background ROI")

                tmp = {"ROIsignal" : ROIsignal, 
                       "ROIbackground" : ROIbackground}
                
                self.roi[im_name] = tmp

        with open(self.path + "/roi.json", "w") as f:
            f.write(json.dumps(self.roi))

    def load_images(self):
        exposure = int(self.properties["exposure"])
        im_names = self.list_images()
        self.images = {}

        for im_name in im_names:
            if im_name == "wire":
                print("pass")
                continue

            try:
                img = Slika(self.path + "/" + im_name + ".jpg", exposure)
            except FileNotFoundError:
                img = Slika(self.path + "/" + im_name + ".tif", exposure)
                
            self.images[im_name] = img

    def load_images_multiple_exposures(self):
        im_names = self.list_images()
        tmp = [im_name.split("_") for im_name in im_names if im_name not in "uncrossed,wire"]
        im_names = {ele[0] : [] for ele in tmp}

        print(tmp)

        for ele in tmp:
            if ele[0] == "wire" or ele[0] == "uncrossed":
                pass

            im_names[ele[0]].append(ele[1])

        self.images = {}

        for key, values in im_names.items():
            try:
                paths = ["{}/{}_{}.tif".format(self.path, key, value) for value in values if value != "loc"]
                img = Slika_multi_exposure(paths)
            except FileNotFoundError:
                paths = ["{}/{}_{}.jpg".format(self.path, key, value) for value in values if value != "loc"]
                img = Slika_multi_exposure(paths)
                
            self.images[key] = img

    def show_images(self, im_names=None):
        if type(im_names) == list:
            for im_name in im_names:
                self.images[str(im_name)].show()
        elif im_names == None:
            for imname, image in self.images.items():
                print(imname)
                image.show()
        else:
            self.images[im_names].show()

    def reset_images(self):
        for image in self.images.values():
            image.reset()

    def apply_rois(self):
        for key, image in self.images.items():
            image.apply_roi(self.roi[key]["ROIsignal"])

    def process_images(self, kernel=None, size=None, N=1):
        for image in self.images.values():

            for _ in range(N):
                image.apply_gaussian(size)

            image.open_and_close(kernel)
            image.fit_radius()

            # image.show()

    def get_intensities(self):
        self.intensities = []
        
        for key, image in self.images.items():
            tmp = image.average_intensity(self.roi[key]["ROIbackground"])[0]
            tmp = [tmp] + [image.distance(self.wire_center[0])] #+ [key]

            self.intensities.append(tmp)

        return np.array(self.intensities).astype(float)
    
    def get_intensities_all(self):
        self.intensities = []
        
        for key, image in self.images.items():
            tmp = image.average_intensities(self.roi[key]["ROIbackground"])
            tmp = tmp + [image.distance(self.wire_center[0])] #+ [key]

            self.intensities.append(tmp)

        return np.array(self.intensities).astype(float)

    def get_wire_location(self):
        try:
            img = imageio.imread(self.path + "/wire.jpg")
        except FileNotFoundError:
            img = imageio.imread(self.path + "/wire.tif")

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            pass

        if img.dtype == 'uint16':
            img = (img/256).astype('uint8')

        # try:
        #     gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow("select wire ROI", cv2.WINDOW_NORMAL)
        ROIsignal = cv2.selectROI("select wire ROI", 
                                    img, False)
        cv2.destroyWindow("select wire ROI")

        imgCrop = img[ROIsignal[1]:ROIsignal[1]+ROIsignal[3],
                      ROIsignal[0]:ROIsignal[0]+ROIsignal[2]]

        _, thresh = cv2.threshold(imgCrop, 0, 255, 
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imshow("Image", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(c) for c in contours]
        sorted_areas = np.sort(areas)
        cnt = contours[areas.index(sorted_areas[-1])] #the biggest contour

        (x_axis,y_axis), radius = cv2.minEnclosingCircle(cnt)
        
        self.wire_radius = int(radius)
        self.wire_center = [int(x_axis),int(y_axis)]

        cv2.circle(imgCrop, self.wire_center, self.wire_radius, (127,127,127), 2)

        cv2.imshow("Image", imgCrop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.wire_center[0], self.wire_center[1] = (self.wire_center[0] + ROIsignal[0], 
                                                    self.wire_center[1] + ROIsignal[1])


class Zica_2(Zica):
    factor = 0.5

    def set_roi(self, center, radius, roisignal, roibackground):
        self.roi["center"] = center
        self.roi["radius"] = radius
        self.roi["ROIsignal"] = roisignal
        self.roi["ROIbackground"] = roibackground

        with open(self.path + "/roi.json", "w") as f:
            f.write(json.dumps(self.roi))

    def set_roi_old(self):
        if "ROIsignal" and "ROIbackground" not in self.roi.keys():
            try:
                img = imageio.imread(self.path + "/uncrossed.jpg")
            except FileNotFoundError:
                img = imageio.imread(self.path + "/uncrossed.tif")

            while True:
                cv2.namedWindow("select signal ROI", cv2.WINDOW_NORMAL)
                
                ROI = cv2.selectROI("select signal ROI", img, False)
                cv2.destroyWindow("select signal ROI")
                print("Signal ROI selected:")

                radius = min((ROI[2]/2, ROI[3]/2))
                print(ROI, radius)
                radius = int(radius)
                center = (int(ROI[2]/2), int(ROI[3]/2))

                img2 = np.copy(img)

                cv2.circle(img2, (center[0]+ROI[0], center[1]+ROI[1]), int(radius*self.factor), (255,0,0), 2)
                cv2.imshow("Image", img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                msg_box = messagebox.askquestion('Region check', 'Do you like detected region?',
                                                icon='warning')

                ROI = [center[0]+ROI[0]-radius, center[1]+ROI[1]-radius, 2*radius+1, 2*radius+1]

                if msg_box == 'yes':
                    break


            ROIbackground = cv2.selectROI("select background ROI", 
                                            img, False)
            cv2.destroyWindow("select background ROI")

            self.roi = {"ROIsignal" : ROI, 
                    "ROIbackground" : ROIbackground}
            self.radius = radius
            self.center = center

            with open(self.path + "/roi.json", "w") as f:
                f.write(json.dumps(self.roi))

    def apply_rois(self):
        for image in self.images.values():
            image.apply_roi(self.roi["ROIsignal"])
            image.radius = self.roi["radius"]
            image.center = self.roi["center"]

    def process_images(self, kernel=None, size=None, N=1):
        pass

    def get_intensities(self):
        self.intensities = []
        
        for key, image in self.images.items():
            avg, std = image.average_intensity(self.roi["ROIbackground"], self.factor)
            tmp = [avg] + [image.distance(self.roi[key][0])] + [std] #+ [key]

            self.intensities.append(tmp)

        return np.array(self.intensities).astype(float)
    
    def get_intensities_all(self):
        self.intensities = []
        
        for key, image in self.images.items():
            tmp = image.average_intensities(self.roi["ROIbackground"])
            tmp = tmp + [image.distance(self.roi[key][0])] #+ [key]

            self.intensities.append(tmp)

        return np.array(self.intensities).astype(float)
    
    def get_wire_location(self):
        # for key, img in self.images.items():
        for key in self.images.keys():
            if key not in self.roi.keys():
                img = imageio.imread("{}/{}_loc.tif".format(self.path, key))
                # cv2.imshow('image', img.original_image*20)
                cv2.imshow("image", img)
                cv2.setMouseCallback('image', self.click_event)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                self.roi[key] = (self.x, self.y)
                print(self.roi)

        with open(self.path + "/roi.json", "w") as f:
            f.write(json.dumps(self.roi))

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x = x
            self.y = y

    def draw_lines(self):
        for key, img in self.images.items():
            img2 = np.copy(img.original_image) * 10
            img2 = cv2.line(img2, img.center,
                             self.roi[key], (255, 255, 255), 3)

            cv2.imshow('image', img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()            