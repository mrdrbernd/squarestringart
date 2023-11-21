# A simple stringart-script for rectangular images
# based on: https://github.com/theveloped/ThreadTone
#
# script version
# 13.11.23 Bernd Heitkamp
#

#!/usr/bin/env python
import sys
import cv2
import numpy as np
from tkinter import filedialog

imgPath="./J.jpg"
#imgPath=""
# image file for threading

initPin = 0        # Initial pin to start threading from:
# 0 is top right

numPins_horiz = 53
numPins_vert = 71
numPins = 2*(numPins_horiz+numPins_vert)-4      # Number of pins on the rectangular loom
# all horizontal pins of one side/ all vertical pins on one side
#     pin 0:  top right

numLines = 2000    # Maximal number of lines

minLoop = 5         # Disallow loops of less than minLoop lines

lineWidth = 3     # The number of pixels that represents the width of a thread
lineWeight = 20     # The weight a single thread has in terms of "darkness"

# \/main processes
banner = """
  _____ _        _                        _    _____                            
 / ____| |      (_)                      | |  / ____|                           
| (___ | |_ _ __ _ _ __   __ _  __ _ _ __| |_| (___   __ _ _   _  __ _ _ __ ___ 
 \___ \| __| '__| | '_ \ / _` |/ _` | '__| __|\___ \ / _` | | | |/ _` | '__/ _ \
 ____) | |_| |  | | | | | (_| | (_| | |  | |_ ____) | (_| | |_| | (_| | | |  __/
|_____/ \__|_|  |_|_| |_|\__, |\__,_|_|   \__|_____/ \__, |\__,_|\__,_|_|  \___|
                          __/ |                         | |                     
                         |___/                          |_|                                      

Build a thread based halftone representation of an image
(Press: ctrl+c in this terminal window to kill the drawing)
"""
PINS=[]

# Apply polygon mask to the image
def maskImage(imgInverted, coords):
    mask=np.zeros(imgInverted.shape[:2],dtype="uint8")
    cv2.fillPoly(mask, pts=[coords], color=255)
    masked = cv2.bitwise_and(imgInverted, imgInverted, mask=mask)
    
    return masked, mask

def pinCoords_Rectangle(rx,ry, numPins_horiz=200, numPins_vert=200, offset=0, x0=None, y0=None):
    if (x0 == None) or (y0 == None):
        x0 = rx + 1
        y0 = ry + 1

    L_total=4*rx+4*ry
    #delta_L=L_total/(numPins+1)
    
    y_val=np.arange(0,2*ry,2*ry/(numPins_vert-1))
    x_val=np.arange(0,2*rx,2*rx/(numPins_horiz-1))
    
    #print("[+] vertical pins (corrected): " + str(len(y_val)) + ", horizontal pins:" + str(len(x_val))+".")


    coords = []
    x=2*rx
    for y in y_val:
        coords.append([int(x),int(y)])
    y=2*ry
    for x in x_val[::-1]:
        coords.append([int(x),int(y)])
    x=0
    for y in y_val[::-1]:
        coords.append([int(x),int(y)])
    y=0
    for x in x_val:
        coords.append([int(x),int(y)])
    return np.array(coords)


# Compute a line mask
def linePixels(pin0, pin1):
    length = int(np.hypot(pin1[0] - pin0[0], pin1[1] - pin0[1]))

    x = np.linspace(pin0[0], pin1[0], length)
    y = np.linspace(pin0[1], pin1[1], length)

    return (x.astype(int)-1, y.astype(int)-1)


def error(img1, img2):
    # calculate difference of two images of same type
    h, w = img1.shape 
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    msre = np.sqrt(mse)
    return mse, diff


def saveSVG(lines,width,height,filename):
    svg_output = open(filename,'wb')
    header="""<?xml version="1.0" standalone="no"?>
    <svg width="%i" height="%i" version="1.1" xmlns="http://www.w3.org/2000/svg">
    """ % (width, height)
    footer="</svg>"
    svg_output.write(header.encode('utf8'))
    pather = lambda d : '<path d="%s" stroke="black" stroke-width="0.5" fill="none" />\n' % d
    pathstrings=[]
    #pathstrings.append("M" + "%i %i" % coords[lines[0][0]] + " ")
    pathstrings.append(f"M {coords[lines[0][0]][0]:f} {coords[lines[0][0]][1]:f} ")
    for l in lines:
        #nn = list(coords[l[1]])
        pathstrings.append(f"L {coords[l[1]][0]:f} {coords[l[1]][1]:f} ")
    pathstrings.append("Z")
    d = "".join(pathstrings)
    svg_output.write(pather(d).encode('utf8'))
    svg_output.write(footer.encode('utf8'))
    svg_output.close() 
 
def saveTXT(PINS,filename):
    txt_output=open(filename,'wb')
    header=""
    footer=""
    txt_output.write(header.encode('utf8'))
    pathstrings=[]
    for pin in PINS:
        pathstrings.append(f"{pin}, ")
    d="".join(pathstrings)
    txt_output.write(d.encode('utf8'))
    txt_output.write(footer.encode('utf8'))
    txt_output.close() 
                     
        

print(banner)


# Load image; open file dialog if no image is given
if imgPath=="":
    imgPath=filedialog.askopenfilename()

image = cv2.imread(imgPath)
height, width = image.shape[0:2]

imgRx=(width-1)//2
imgRy=(height-1)//2

print("[+] loaded " + imgPath + " for threading..")

# Convert to grayscale
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./gray.png', imgGray)

# invert image
imgInverted = ~imgGray
cv2.imwrite('./inverted.png', imgInverted)

# Define pin coordinates and mask image
coords = pinCoords_Rectangle(imgRx, imgRy, numPins_horiz, numPins_vert)

imgMasked, mask = maskImage(imgInverted, coords)
cv2.imwrite('./masked.png', imgMasked)

height, width = imgMasked.shape[0:2]
# image result is rendered to
imgResult = 255 * np.ones((height, width))

# Initialize variables
i = 0
lines = []
previousPins = []
oldPin = initPin
PINS.append(initPin)
lineMask = np.zeros((height, width))

imgResult = 255 * np.ones((height, width))

# Loop over lines until stopping criteria is reached
for line in range(numLines):
    i += 1
    bestLine = 0
    oldCoord = coords[oldPin]

    # Loop over possible lines
    for index in range(1, numPins):
        pin = (oldPin + index) % numPins

        coord = coords[pin]

        xLine, yLine = linePixels(oldCoord, coord)

        # Fitness function
        lineSum = np.sum(imgMasked[yLine, xLine])

        if (lineSum > bestLine) and not(pin in previousPins):
            bestLine = lineSum
            bestPin = pin

    # Update previous pins
    if len(previousPins) >= minLoop:
        previousPins.pop(0)
    previousPins.append(bestPin)

    # Subtract new line from image
    lineMask = lineMask * 0
    cv2.line(lineMask, oldCoord, coords[bestPin], lineWeight, lineWidth)
    imgMasked = np.subtract(imgMasked, lineMask)

    # Save line to results
    lines.append((oldPin, bestPin))
    PINS.append(bestPin)

    # plot results
    xLine, yLine = linePixels(coords[bestPin], coords[oldPin])
    imgResult[yLine, xLine] = 0
    cv2.imshow('image', imgResult)
    #print(bestPin)
    cv2.waitKey(1)

    # Break if no lines possible
    if bestPin == oldPin:
        break
    

    # Prepare for next loop
    oldPin = bestPin

    # Print progress
    if True: #line%100==0:
        sys.stdout.write("\b\b")
        sys.stdout.write("\r")
        sys.stdout.write("[+] Computing line " + str(line + 1) + " of " + str(numLines) + " total")
        sys.stdout.flush()

print("\n[+] Image threaded")

# Wait for user and save before exit
#cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'./threaded.png', imgResult)
        
cv2.destroyAllWindows()
saveSVG(lines,2*imgRx+1,2*imgRy+1,'./threaded.svg')
saveTXT(PINS,'./pins.txt')