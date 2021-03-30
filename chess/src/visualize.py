import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import streamlit as st


def colorChannels(image): #View Color Spaces

    choice = st.sidebar.selectbox('Color Channels', ["HSV","GRAY"])
    choice = "GRAY"
    if(choice == "HSV"):
        image = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    if(choice == "GRAY"):
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    return image

def hough(image, edges):

    #dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    #lines: A vector that will store the parameters (r,θ) of the detected lines
    #rho : The resolution of the parameter r in pixels. We use 1 pixel.
    #theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    #threshold: The minimum number of intersections to "*detect*" a line
    #srn and stn: Default parameters to zero. Check OpenCV reference for more info.


    lines = cv.HoughLines(edges,1,np.pi/180,150, None, 0, 0)

    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        image = cv.line(image, pt1, pt2, (0,0,0), 2, cv.LINE_AA)

    return image

def threshType(thresh):

    thresh = 0

    if(thresh == "Binary"):
        thresh = cv.THRESH_BINARY

    if(thresh == "BinaryINV"):
        thresh = cv.THRESH_BINARY_INV

    if(thresh == "Trunc"):
        thresh = cv.THRESH_TRUNC

    if(thresh == "TOZERO"):
        thresh = cv.THRESH_TOZERO

    if(thresh == "TOZEROINV"):
        thresh = cv.THRESH_TOZERO_INV

    return thresh


def thresholds(image): #Selects type of threshold: Binary being 0s and 1s and BinaryINV

    thresh = st.sidebar.selectbox('Select', ["None","Binary","BinaryINV", "Trunc","TOZERO","TOZEROINV", "Adaptive"])

    if(thresh == "None"):
        return image

    if(thresh == "Adaptive"):

        maxVal = st.sidebar.slider("Max Value",min_value=1,value = 255, max_value=255)
        method = st.sidebar.selectbox('Select', ["Mean","Gaussian"])

        if(method == "Mean"):
            method = cv.ADAPTIVE_THRESH_MEAN_C
        if(method == "Gaussian"):
            method = cv.ADAPTIVE_THRESH_GAUSSIAN_C

        thresh = st.sidebar.selectbox('Treshold', ["None","Binary","BinaryINV", "Trunc","TOZERO","TOZEROINV"])
        thresh = threshType(thresh)
        block_size = st.sidebar.slider("Block Size",min_value=1,value = 11, max_value=255)
        constant = st.sidebar.slider("Constant",min_value=1,value = 5, max_value=100)
        image = cv.adaptiveThreshold(image,maxVal,method,thresh,block_size,constant)
        return image


    min = st.sidebar.slider("Minimum Pixel Intensity",min_value=1,value = 100, max_value=255)
    max = st.sidebar.slider("Maximum Pixel Intensity",min_value=1,value = 255, max_value=255)

    type = threshType(thresh)
    ret, image = cv.threshold(image,min,max,type)

    return image

def blurring(image):

    #type = st.sidebar.selectbox("Blur Type",["None","Boxfilter","Gaussian","Median","Bilateral"])
    type = 'Gaussian'

    if(type == "None"):
        return image

    if(type == "Boxfilter"): #takes the average of of the pixels under the kernel

        w = st.sidebar.slider("Kernel Width",min_value=1,value = 5, max_value=255)
        h = st.sidebar.slider("Kernel Height",min_value=1,value = 5, max_value=255)

        blur_img = cv.blur(image,(w,h))

    if(type == "Gaussian"): # uses a gaussian kernel

        w = st.sidebar.slider("Kernel Width",min_value=1,value = 5, max_value=100)
        h = st.sidebar.slider("Kernel Height",min_value=1, value = 5, max_value=100)
        sigmaX = st.sidebar.slider("SigmaX",min_value=0, max_value=255)

        blur_img  = cv.GaussianBlur(image,(w,h),sigmaX)

    if(type == "Median"): #takes the median of all the pixels under the kernel area and central element is replaced with this median value
        size = st.sidebar.slider("Filter Size",min_value=1,value = 5, max_value=255)
        blur_img = cv.medianBlur(image,size)

    if(type == "Bilateral"):

        d = st.sidebar.slider("Pixel Diameter",min_value=1, max_value=255)
        SigmaColor = st.sidebar.slider("SigmaColor",min_value=1,value = 75, max_value=255)
        SigmaSpace = st.sidebar.slider("SigmaSpace",min_value=1,value = 75, max_value=255)
        blur_img = cv.bilateralFilter(image,d,sigmaColor,sigmaSpace)

    return blur_img

        #arguments - d ( diameter of each pixel in neighborhood)
        #sigmaColor value of sigma in color space. Greater value means colors farther aways to each other will start to get mixed
        #Sigmacolr-   Value of \sigma in the coordinate space. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.


def corners(image):

    choice = st.sidebar.selectbox('Edge Detection', ["None","Canny","Shi-Tomashi"])
    choice = "Canny"

    if(choice == "None"):
        return image

    if(choice == "Shi-Tomashi"):

        edges = cv.goodFeaturesToTrack(image,100,0.01,100)
        edges = np.int0(edges)

        radius = st.sidebar.slider("Radius",min_value=1,value = 10, max_value=255)
        color =  st.sidebar.slider("Color",min_value=1, max_value=255)

        for i in edges:

           x,y = i.ravel()
           image = cv.circle(image,(x,y),radius,(color,0,0))

        return image

    if(choice == "Canny"):

        minval = st.sidebar.slider("Minval",min_value=1,value = 50, max_value=255)
        maxval =  st.sidebar.slider("Maxval",min_value=1,value = 100, max_value=200)
        aperature = st.sidebar.slider("aperature",min_value=1,value = 3, max_value=50)
        edges = cv.Canny(image,minval,maxval,aperature)
        image = hough(image,edges)


        return image


def contour(image):

    choice = st.sidebar.selectbox("Contour",["None","Contour"])
    choice = "Contour"
    count = 0
    newcontours = {}

    if(choice == "None") :
        return image

    contours,hierarchy = cv.findContours(image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)

    val = st.sidebar.slider("Perimeter Limit",min_value=1, max_value=10000)
    pixelSum = st.sidebar.slider("Sum",min_value=0,max_value=1000000)
    square = st.sidebar.slider("Square",min_value=0,max_value=63)

    for contour in contours:

        perimeter = cv.contourArea(contour, False)

        if(perimeter>val):

            newcontours[count] = contour
            count = count + 1
            #mask = np.zeros_like(image)
            #image = cv.drawContours(mask,[contour],0,(255,0,0),1)
            #ret,image = cv.threshold(image,min,max,0)
            #out = np.zeros_like(image) # Extract out the object and place into output image
            #out[mask == 255] = image[mask == 255]

            #if(np.sum(out) == 527850):
               # count = count + 1

            #print(contour.shape)

    if(square!=0):
        image = cv.drawContours(image,[newcontours[square]],0,(255,0,0),1)

    print(count)

    return image


if __name__ == "__main__":

    image = cv.imread("image2.png")

    image = colorChannels(image)
    image = blurring(image)
    image = thresholds(image)
    image = corners(image)
    image = contour(image)

    col =  st.beta_columns(1)
    col[0].image(image,use_column_width=True)
