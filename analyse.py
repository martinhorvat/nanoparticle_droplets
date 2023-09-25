# -*- coding: utf-8 -*-
"""
Created on May 6 2022



@author: natan
"""

import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
from pathlib import Path
import imageio

def track(path, threshold = None, testrun = False, ROIsignal = None, ROIbackgroud= None):
    #import doctest
    #doctest.testmod()
    
    #fname = "mocnapAST_3.twv"




    files = Path(path).glob('*.png')
    imagefiles = []

    #plt.clf()
    #plt.imshow(file0,  cmap = cm.gray)
    
    # print(os.path.dirname(path))

    frez = open(path+ "//results.dat", 'w')     
    
    print("Results in file %s" % frez)

    brightness_evol = []
    background_evol = []

    for idx, file in enumerate(files):
        #imagefiles.append(file)

        fileinfo = (int) (Path(file).stem)
        img = imageio.imread(file)
        print(img.shape)
        if (idx==0):
             #plt.subplot(221).clear()
             plt.imshow(img,  cmap = cm.hot)
             #plt.imshow(img)

    
    
#    if (testrun):
         #num_frames = 20
       

        if (ROIsignal==None):
            cv2.namedWindow("select signal ROI", cv2.WINDOW_NORMAL)
            
            ROIsignal = cv2.selectROI("select signal ROI", img, False)
            cv2.destroyWindow("select signal ROI")
            print("Signal ROI selected:")
            print(ROIsignal)

        if (ROIbackgroud==None):
            ROIbackgroud = cv2.selectROI("select background ROI", img, False)
            cv2.destroyWindow("select background ROI")
            print("Background ROI selected:")
            print(ROIbackgroud)

        #f = open(path+"\\roi.txt", "w")
        #f.write("%d %d %d %d\n" % ROIsignal)
        #f.write("%d %d %d %d" % ROIbackgroud)
        #f.close()

        #minX=img.shape[1] //2 -5
        #maxX=img.shape[1] //2 +5 
        #part=img[8:20,8:20] # image is transposed as an array
        imCrop = img[int(ROIsignal[1]):int(ROIsignal[1]+ROIsignal[3]), int(ROIsignal[0]):int(ROIsignal[0]+ROIsignal[2])]
        imBackground = img[int(ROIbackgroud[1]):int(ROIbackgroud[1]+ROIbackgroud[3]), int(ROIbackgroud[0]):int(ROIbackgroud[0]+ROIbackgroud[2])]
        # if (idx==0):
        #     plt.imshow(imCrop,  cmap = cm.hot)
        #     plt.show()

        avg_brightness_signal=np.mean(imCrop)
        avg_brightness_background=np.mean(imBackground)
        print("frame no: %d    avg. brightness close to %d is %.2f, background is %.2f " % (idx, ROIsignal[0],avg_brightness_signal, avg_brightness_background))



        brightness_evol.append((fileinfo,avg_brightness_signal))
        background_evol.append((fileinfo,avg_brightness_background))

        #frez.write('%.4f ' % (frame_time))
        frez.write('%d ' % (fileinfo))
        frez.write('%.2f ' % (avg_brightness_signal))
        frez.write('%.2f\n' % (avg_brightness_background))
      

    frez.close()
    
    polarizationAngleOffset=0

    if len (brightness_evol)>0:
        t, signal = np.array(brightness_evol).T
        t, background = np.array(background_evol).T
        max_idx = np.argmax(signal)
        min_idx = np.argmin(signal)
        print("maximum value of signal is %.2f at angle of %.2f" % (signal[max_idx], t[max_idx], ))
        print("minimum value of signal is %.2f at angle of %.2f" % (signal[min_idx], t[min_idx], ))
        # plt.subplot(222)
        # plt.axis('auto')
        # plt.scatter(t, signal, marker=".", facecolors="None", color="black", s=2, linewidths=1, label="Circles")
        # plt.scatter(t, background, marker=".", facecolors="None", color="Gray", s=1, linewidths=1, label="Circles")
        # #plt.plot(t,bright)
        # plt.xlabel('Incoming polarization')
        # plt.ylabel('Signal, background [a.u.]')
        
        
        #fig = plt.figure()
        #ax1 = plt.subplot(121)
        #ax2 = plt.subplot(122, projection='polar')

        #ylabel='SHG Intensity (w.o. BG)'
        #ylabel='SHG-bgrnd vs. polarization, vertical E=10V/um, analyser horizontal'
       # ylabel='SHG-bgrnd vs. polarization, poled vertical, analyser vertical'
        #ylabel='SHG vs. polarization - analyser vertical'
        #ylabel='SHG vs. polarization - analyser horizontal'
        #ylabel='Another region,SHG-bgrnd vs. polarization, poled vertical, analyser Horizontal'
        #ylabel='Fundamental IR beam  - analyser horizontal'
        ylabel='SHG left droplet part - analyser horizontal'
        #ylabel='RM374 mali delcek - SHG signal - NO analyser - mode7 200ms/no gain '
        plt.clf()
        #plt.subplot(121)
        
        #ax1.axis('auto')
        plt.scatter(t-polarizationAngleOffset, signal-background, marker=".", facecolors="None", color="black", s=2, linewidths=1, label="Circles")
        #plt.plot(t,bright)
        plt.xlabel('Incoming polarization')
        plt.ylabel(ylabel)
        plt.title(path,fontsize = 6)

        #plt.subplot(122)
        plt.savefig(os.path.join(path,'results.jpg'))

        
        r = np.arange(0, 2, 0.01)
        theta = 2 * np.pi * r

        #ax2.scatter(theta, r)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot((t-polarizationAngleOffset)/180.0*np.pi, signal-background)
        #ax.set_rmax(2)
        #ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        plt.title(ylabel)
        ax.grid(True)
        

        plt.savefig(os.path.join(path,'polar-results.jpg'))
        plt.show()
     


if __name__ == "__main__":
    #track(fname = "d:\\FMF\\Lab\\scripts\\python\\CryoMicroscope\\AA.twv", testrun=False)
    # track(fname = "y:\\temp\\shg_int\\aoi_from_-25to25.twv", testrun=False)
    #track(fname = "y:\\temp\\shg_int\\aoi_from_25to-25-delayChanged.twv", testrun=False)
    #track(fname = "c:\\temp\\increasing_symmetric_0_450_step10.twv", testrun=False)
    
    #track(fname = r"y:/Lab/Meritve/SHGMicroscope/shg_natan/GillesPZT/2022-04-07/holding450.twv", testrun=False, selectROI=True)
    #track(fname = r"y:/Lab/Meritve/SHGMicroscope/shg_natan/GillesPZT/2022-04-07/increasing_symmetric_0_450_step10V_20s_recorded.twv", testrun=False, selectROI=True)
    #track(fname = r"y:/Lab/Meritve/SHGMicroscope/shg_natan/GillesPZT/2022-04-12/rotation_of_incoming_polarization_startHorizontal.twv", testrun=False, selectROI=False)
  #  track(fname = r"y:/Lab/Meritve/SHGMicroscope/shg_natan/GillesPZT/2022-04-12/rot_polariz_startHorizontal-outsideRegion.twv", testrun=False, ROIsignal=(94, 92, 9, 42), ROIbackgroud=(157, 84, 24, 64))
     #track(fname = r"y:/Lab/Meritve/SHGMicroscope/shg_natan/GillesPZT/2022-04-21/increasing_symmetric_0_450_step10V_20s_good.twv", testrun=False, ROIsignal=(81, 112, 10, 12), ROIbackgroud=(130, 110, 11, 12))
     #track(fname = r"y:/Lab/Meritve/SHGMicroscope/shg_natan/GillesPZT/2022-04-21/increasing_symmetric_450_610_step10V_20s_good.twv", testrun=False, ROIsignal=(81, 112, 10, 12), ROIbackgroud=(130, 110, 11, 12))
     #track(fname = r"y:/Lab/Meritve/SHGMicroscope/shg_natan/GillesPZT/2022-04-21/holding450_ok.twv", testrun=False, ROIsignal=(81, 112, 10, 12), ROIbackgroud=(130, 110, 11, 12))

    root = Tk()
    root.update() 
    
    path = askdirectory(title='Select Folder') # shows dialog box and return the path
    root.destroy()
    #path="Y://Lab//Meritve//SHGMicroscope//shg_natan//GillesPZT//2022-05-05//polarizationDependence//pureLaserBeam-AnalyserVertical"

   # roi=(761, 779, 272, 127)    

    roiFile = Path(path+"//roi.txt")

    if roiFile.is_file():
    #     print("Roi file exists")
        f = open(roiFile, "r")
        roiSignal= f.readline()
        roiSignal = tuple(map(int, roiSignal.split(' ')))
        roiBackground= f.readline()
        roiBackground = tuple(map(int, roiBackground.split(' ')))
        f.close()
        track(path, ROIsignal=roiSignal, ROIbackgroud=roiBackground)
    else:
        track(path)
    # 
    # if roiFile.is_file():
    #     print("Roi file exists")

    #print(path)
    #track(path)
    
    
 #   track(path, ROIsignal=roiSignal, ROIbackgroud=roiBackground)
    
    # track(path, ROIsignal=(61, 126, 21, 21), ROIbackgroud=(309, 120, 31, 28))
    #track(path, ROIsignal=(195, 446, 133, 88),ROIbackgroud= (853, 373, 129, 74))
    # iterate over files in
# that directory
    