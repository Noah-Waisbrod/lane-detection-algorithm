#---------------------------------------------------------------------------------
# Project by: Noah Waisbrod
# Date: Mar 2022
# Description: An algorythm that uses open CV to proecess dashcam footage and draws lane line on
#---------------------------------------------------------------------------------
import cv2
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

#converts image to canny image
def linesImg(image):
    #converts image to grey
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #blurs image to reduce noise
    blur = cv2.GaussianBlur(grey, (5,5), 10)
    #grabs sharp changes in colour
    canny = cv2.Canny(blur, 50, 150)
    return canny

#cuts out the region of interest wanted
def roadInterest(image):
    #grabs image height
    height = image.shape[0]
    #buildins triangle with area of interest
    triangle = np.array([[(200, height), (1200, height), (650, 450)]])
    #makes mask of triangle
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    #merges Image and mask
    maskedImg = cv2.bitwise_and(image, mask)
    return maskedImg

#draws lines
def lineDisplay(image, lines):
    lineImg = np.zeros_like(image)
    lineImg = cv2.cvtColor(lineImg, cv2.COLOR_GRAY2BGR)
    BLUE = (255,0,0)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            try:
                cv2.line(lineImg, (x1, y1), (x2, y2), BLUE, 10)
            except:
                print("an error happend")
    return lineImg

#avrages lines
def avgSlopeInt(image, lines):
    #init left and right line arrays
    Lfit = []
    Rfit = []
    #loop through all lines and break them up into slope and intersept
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        param = np.polyfit((x1,x2), (y1, y2), 1)
        slope = param[0]
        intercept = param[1]
        if slope < 0:
            Lfit.append((slope,intercept))
        else:
            Rfit.append((slope,intercept))
    #avrage all lines to get in middle line
    LfitAvg = np.average(Lfit, axis=0)
    RfitAvg = np.average(Rfit, axis=0)
    Lline = coordinates(image, LfitAvg)
    Rline = coordinates(image, RfitAvg)
    return np.array([Lline, Rline])

#sub functions for avrage slope function
def coordinates(image, lineParam):
    slope, intercept = lineParam
    y1 = image.shape[0]
    y2 = int(y1*(6.5/10))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def slope(line, int):
    return ((line[int][3]-line[int][1])/(line[int][2]-line[int][0]))

#combines fuctions and computs final product
# def action(image):
#     cannyImg = linesImg(image)
#     roadImg = roadInterest(cannyImg)
#     lines = cv2.HoughLinesP(roadImg, 2, (np.pi)/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
#     avgLines = avgSlopeInt(image, lines)
#     lineImg = lineDisplay(roadImg, avgLines)
#     return cv2.addWeighted(image, 1, lineImg, 1, 1)


#image test
#-------------------------------------------------------------------------------------------------------------------------------------------

img = cv2.imread('test4.jpg')
Lane_img = np.copy(img)
cannyImg = linesImg(Lane_img)
roadImg = roadInterest(cannyImg)
lines = cv2.HoughLinesP(roadImg, 2, (np.pi)/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
avgLines = avgSlopeInt(Lane_img, lines)
lineImg = lineDisplay(roadImg, avgLines)
colourOverlayImg = cv2.addWeighted(Lane_img, 1, lineImg, 1, 1)

# # plt.imshow(colourOverlayImg)
# # plt.show()
cv2.imshow("image", avgLines)
cv2.waitKey(0)

#video Test
#--------------------------------------------------------------------------------------------------------------------------------------------

cap = cv2.VideoCapture('highway.mp4')
tempLines = image
Avgl = np.array([])
while(cap.isOpened()):
    _, frame = cap.read()
    try:
        cannyImg = linesImg(frame)
        roadImg = roadInterest(cannyImg)
        lines = cv2.HoughLinesP(roadImg, 2, (np.pi)/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
        avgLines = avgSlopeInt(frame, lines)
        if(slope(avgLines,0) < -0.5 and slope(avgLines,1) > 0.5):
            Avgl = avgLines
            lineImg = lineDisplay(roadImg, Avgl)
            colourOverlayImg = cv2.addWeighted(frame, 1, lineImg, 1, 1)
            tempLines = lineImg
            cv2.imshow("image", colourOverlayImg)
            if cv2.waitKey(100) == ord('q'):
                break
        else:   
            lineImg = lineDisplay(roadImg, Avgl)
            colourOverlayImg = cv2.addWeighted(frame, 1, lineImg, 1, 1)
            tempLines = lineImg
            cv2.imshow("image", colourOverlayImg)
            if cv2.waitKey(100) == ord('q'):
                break
    except:
        colourOverlayImg = cv2.addWeighted(frame, 1, tempLines, 1, 1)
        cv2.imshow("image", colourOverlayImg)
        if cv2.waitKey(100) == ord('q'):
            break

cap.release()


cv2.destroyAllWindows()