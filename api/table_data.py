from google.cloud import vision
from os import environ
import urllib.request
from numpy import zeros, uint8, float32, array, argsort, vstack
from cv2 import imread, GaussianBlur, cvtColor, COLOR_BGR2GRAY, getStructuringElement, MORPH_ELLIPSE
from cv2 import morphologyEx, MORPH_CLOSE, normalize, NORM_MINMAX, COLOR_GRAY2BGR, adaptiveThreshold
from cv2 import findContours, RETR_TREE, CHAIN_APPROX_SIMPLE, contourArea, drawContours, bitwise_and
from cv2 import Sobel, convertScaleAbs, threshold, THRESH_BINARY, THRESH_OTSU, MORPH_DILATE, CV_16S
from cv2 import boundingRect, RETR_LIST, moments, imwrite, getPerspectiveTransform, warpPerspective
from cv2 import MORPH_RECT, RETR_EXTERNAL

#Reading the image and converting to Grayscale
def detect_document(img_data):
    environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"api/service-account.json"
    client = vision.ImageAnnotatorClient()
    imwrite("api/vision_data.jpg", img_data)
    with open("api/vision_data.jpg", "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    text = response.full_text_annotation.text
    return text

def imagePreProcessing(img_url):
    urllib.request.urlretrieve (img_url, "api/image.jpg")
    img = imread('api/image.jpg')
    img = GaussianBlur(img,(5,5),0)
    gray = cvtColor(img,COLOR_BGR2GRAY)
    kernel1 = getStructuringElement(MORPH_ELLIPSE,(11,11))

    close = morphologyEx(gray,MORPH_CLOSE,kernel1)
    div = float32(gray)/(close)
    # freeResources()
    return uint8(normalize(div,div,0,255,NORM_MINMAX))

def detectTableArea(res) :
    #Detecting Table Area
    mask = zeros((res.shape),uint8)
    thresh = adaptiveThreshold(res,255,0,1,19,2)
    contour,hier = findContours(thresh,RETR_TREE,CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for cnt in contour:
        area = contourArea(cnt)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    drawContours(mask,[best_cnt],0,255,-1)
    drawContours(mask,[best_cnt],0,0,2)

    return bitwise_and(res,mask)

def detectVerticalLines(res):
    #Detecting Vertical Lines

    kernelx = getStructuringElement(MORPH_RECT,(2,10))

    dx = Sobel(res,CV_16S,1,0)
    dx = convertScaleAbs(dx)
    normalize(dx,dx,0,255,NORM_MINMAX)
    ret,close = threshold(dx,0,255,THRESH_BINARY+THRESH_OTSU)
    close = morphologyEx(close,MORPH_DILATE,kernelx,iterations = 1)

    contour, hier = findContours(close,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = boundingRect(cnt)
        if h/w > 8:
            drawContours(close,[cnt],0,255,-1)
        else:
            drawContours(close,[cnt],0,0,-1)
    return morphologyEx(close,MORPH_CLOSE,None,iterations = 2)

def detectHorizontalLines(res):
    #Detecting Horizontal Lines

    kernely = getStructuringElement(MORPH_RECT,(10,2))
    dy = Sobel(res,CV_16S,0,2)
    dy = convertScaleAbs(dy)
    normalize(dy,dy,0,255,NORM_MINMAX)
    ret,close = threshold(dy,0,255,THRESH_BINARY+THRESH_OTSU)
    close = morphologyEx(close,MORPH_DILATE,kernely)

    contour, hier = findContours(close,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = boundingRect(cnt)
        if w/h > 8:
            drawContours(close,[cnt],0,255,-1)
        else:
            drawContours(close,[cnt],0,0,-1)

    return morphologyEx(close,MORPH_DILATE,None,iterations = 2)

def freeResources():
    from os import path, remove
    if path.exists("api/image.jpg"):
        remove("api/image.jpg")
        print("success_image")
    if path.exists("api/vision_data.jpg"):
        remove("api/vision_data.jpg")
        print("success_vision")


def tableToData(img_url):
    res = imagePreProcessing(img_url) #Image PreProcessing
    res2 = cvtColor(res,COLOR_GRAY2BGR)

    res = detectTableArea(res) #Detecting Table Area
    closex = detectVerticalLines(res) #Detecting Vertical Lines
    closey = detectHorizontalLines(res) #Detecting Horizontal Lines

    res = bitwise_and(closex,closey) #Detecting Intersection Points of Lines

    #Detecting Centroids of Intersection Points
    contour, hier = findContours(res,RETR_LIST,CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contour:
        mom = moments(cnt)
        if mom['m00'] > 0:
            (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
            centroids.append([x,y])

    #Sorting the Intersection Points by row Increasing Order
    centroids = array(centroids, dtype = float32)
    c = centroids.reshape((36,2))
    c2 = c[argsort(c[:,1])]

    b = vstack([c2[i*6:(i+1)*6][argsort(c2[i*6:(i+1)*6,0])] for i in range(6)])
    bm = b.reshape((6,6,2))

    data = []
    for ri in range(len(bm)-1):
        row_data = []
        for ci in range(len(bm[0])-1):
            src = bm[ri:ri+2, ci:ci+2 , :].reshape((4,2))
            cell = res2[int(src[0][1]):int(src[3][1]) , int(src[0][0]):int(src[3][0])].copy()
            detected_text = detect_document(cell)
            row_data.append(detected_text.strip())
        data.append(row_data)
    freeResources()
    return data

