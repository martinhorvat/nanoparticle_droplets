import cv2
import numpy as np

def get_radius(img, plot=True):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray = img

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # ret, thresh = cv2.threshold(gray, 127, 255, 0)
    
    mask = thresh == 0
    thresh[:, :] = 0
    thresh[mask] = 255

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)
    
    print(len(contours))

    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    # mask = sorted_areas < (np.pi * np.amin(img.shape)**2 / 4)
    # sorted_areas = sorted_areas[mask]

    #bounding box (red)
    cnt = contours[areas.index(sorted_areas[-1])] #the biggest contour

    # count = contours[i_max]
    (x_axis,y_axis), radius = cv2.minEnclosingCircle(cnt)
    
    radius = int(radius)
    center = (int(x_axis),int(y_axis))

    print(center, radius)

    if plot:
        img2 = np.copy(img)

        cv2.circle(img2, center, radius, (255,0,0), 2)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Image", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return center, radius