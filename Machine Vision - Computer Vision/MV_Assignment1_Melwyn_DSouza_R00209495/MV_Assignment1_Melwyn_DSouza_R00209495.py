# -*- coding: utf-8 -*-
"""
Name  - Melwyn D Souza
ID - R00209495
Course - MSc in AI
Module - Machine Vision
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os, math, copy

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

cwd = os.getcwd()
savepath =cwd+"\\results\\"

global scaleSpace, sigmaList, doglist

def main():        
    global scaleSpace, sigmaList, doglist
    input_image = cv.imread("Assignment_MV_1_image.png")
    display("Input Image", input_image)
    
    """   TASK 1
    Scale Space Images
    scaleSpace - DICT - holds sigma as keys and respective images as values
    sigmaList - LIST - holds 12 value of sigmas used accross the assignment"""
    
    greyscaled = cv.cvtColor(input_image.astype('float32'), cv.COLOR_RGB2GRAY)
    display("Grey Input", greyscaled)
    
    scaleSpace = {}
    sigmaList = []
    doglist = {}
    
    for i in range (12): 
        sigma = 2**(i/2)
        sigmaList.append(sigma)
        scaleSpace[sigma] = gauss_blur("Image", greyscaled, sigma)
    
    """   TASK 2
    DoGs and Key-Points
    Iterates through all sigma values and its scale images to get DoG images"""
    for i in range(len(sigmaList)-1):
        dog = scaleSpace[sigmaList[i]] -scaleSpace[sigmaList[i+1]]
        doglist[sigmaList[i]] = dog
        windowname = "DoG - Sigma "+str(sigmaList[i])+" - Sigma "+str(sigmaList[i+1])
        display(windowname, dog)
    #Theresold is set to 10 and non_maxima_3D - function - checks if the current keypoint is maximum in 3D space+scale
    threshold = 10
    keyPoints = non_maxima_3d(doglist, threshold)
    
    keyImg = copy.deepcopy(input_image)
    for point in keyPoints:
        radius = round(3 * point[2])
        img = cv.circle(keyImg, (point[1], point[0]), radius, (0,0,255))
        
    display("Result", img)
        
    """   TASK 3
    dx,dy are the filters
    gx,gy are gaussian derivative images - DICT - Key: sigma, Value: repective gx or gy image
    mqr, thetaqr, wqr - DICT - magnitude, theta value and weights of keypoint x,y - key: [x,y], value: magintude, theta,weights resp.
    do - DICT - direction of orientation - key: [x,y] of keypoint, value - radian value
    h - DICT - histogram with 36 bins, key - x,y of keypoint, value - 36 bins"""
    dx = np.array([[1, 0, -1]])
    dy = np.transpose(dx)
    gx, gy = {},{}
    
    """task 3.a"""
    for k,v in scaleSpace.items():
        gx[k] = cv.filter2D(v, -1, dx)
        windowname = "Derivative Image gx, Sigma = "+str(k)
        display(windowname,gx[k])
        gy[k] = cv.filter2D(v, -1, dy)
        windowname = "Derivative Image gy, Sigma = "+str(k)
        display(windowname,gy[k])
    
    
    mqr = {}       
    thetaqr = {}   
    wqr = {}
    do = {}
    h = {} 
    
    keyPointsdo=[]
    
    # for each keypoint, a 7x7 grid is made to calculate magnitude adn orientation of the keypoint
    for point in keyPoints:
        #7x7 qr grid
        qr = []
        sigma = point[2]
        x,y = point[0], point[1]
        mqr[x,y] = {}
        thetaqr[x,y] = {}   
        wqr[x,y] = {} 
        h[x,y] = np.zeros(36)
        
        for i in range(-3, 4):
            qr.append((3/2)*i*sigma)
        
        #magnitude and orientation calculation using 49 (7x7) grid-points wrt keypoint at x,y
        for q in qr:
            for r in qr:
                if (round(x+q) < 1080) and (round(y+r)<1920):
                    """task 3.b"""
                    mqr[x,y][q,r] = np.sqrt((gx[sigma][round(x+q),round(y+r)])**2 + (gy[sigma][round(x+q),round(y+r)])**2)
                    # thetaqr[x,y][q,r] = np.arctan2(gx[sigma][round(x+q),round(y+r)], gy[sigma][round(x+q),round(y+r)])
                    thetaqr[x,y][q,r] = np.arctan2(gy[sigma][round(x+q),round(y+r)], gx[sigma][round(x+q),round(y+r)])
    
                    wqr[x,y][q,r] = np.exp(-(q**2+r**2)/((9*sigma**2)/2))*(1/((9*np.pi*sigma**2)/2))
                    
                    """task 3.c - accumulate theta in 36 bins of histogram and select the max bin which is the direction of orientation"""
                    
                    b = (36*thetaqr[x,y][q,r]) / (2*np.pi)
                    hi = np.sum(mqr[x, y][q,r]*wqr[x, y][q, r])
                    
                    h[x,y][math.floor(b)] += hi
                
        do[x,y] = ((2*np.pi)/36)*(0.5 + np.argmax(h[x,y]))
        keyPointsdo.append((x,y,sigma,do[x,y]))      
        
    keyImg = copy.deepcopy(input_image)
    
    """task 3.d"""
    #draw a circle around the keypoint with radius 3*sigma and ann arrow pointing towards the direction of orientation
    for point in keyPointsdo:
        x,y,sigma,radians= point[0], point[1], point[2], point[3]
        radius = round(3*sigma)
        cords = (y,x)
        cv.circle(keyImg, cords, radius, (0,0,255))
        y1 = math.sin(radians)*radius
        x1 = -math.cos(radians)*radius
        end = (y + round(y1), x + round(x1))
        cv.line(keyImg, cords, end, (0,0,255))
    
    display("KeyPoint Direction Orientation",keyImg)
    
    """   TASK 4
    similar to task 3, we have 16X16 grid around the keypoints which is further devided into 4x4 grid
    weights, magnitude, theta, histogram(8 bins) are all accuumulated in dictionaries"""
    wst = {}
    mst = {}       
    thetast = {}   
    hij = {}
    d = {}
    print(len(keyPointsdo))
    
    #itrate through all the keypoints
    #st - DICT - keys: i,j co-ordinate, value- 4X4 s,t grid
    for point in keyPointsdo:
        x,y,sigma,radians= point[0], point[1], point[2], point[3]
        ij = []
        st = {}
        wst[x,y] = {}
        mst[x,y] = {}       
        thetast[x,y] = {}  
        hij[x,y] = {}
        d[x,y] = []
        
        #create a 4x4 s,t grid 
        def get_st(cord):
            st_tmp = []
            for i in range(4*cord[0],4*cord[0]+4):
                for j in range(4*cord[1], 4*cord[1]+4):
                    s_tmp = round((9/16) * (i+0.5) * sigma)
                    t_tmp = round((9/16) * (j+0.5) * sigma)
                    st_tmp.append((s_tmp,t_tmp))
            st[cord] = st_tmp
          
        #create a 4x4 i,j grid
        for i in range(-2,2):
            for j in range(-2,2):
                ij.append((i,j))
                
        #each i,j value in 4x4 grid contains s,t 4x4 grid
        for cord in ij:
            get_st(cord)
        
        #iterate through all 16 i,j grid-points and their 16 s,t grid points to calculate weight, magnitude and theta
        for ij,stvalues in st.items():
            
            wst[x,y][ij] = {}
            mst[x,y][ij] = {}       
            thetast[x,y][ij] = {}  
            hij[x,y][ij] = np.zeros(8)
            for value in stvalues:
                s,t = value[0], value[1]
                if (round(x+s) < 1080) and (round(y+t)<1920):
                    
                    """task 4.a"""
                    wst[x,y][ij][value] = np.exp(-(s**2+t**2)/((81*sigma**2)/2))*(1/((81*np.pi*sigma**2)/2))
                    mst[x,y][ij][value] = np.sqrt((gx[sigma][round(x+s),round(y+t)])**2 + (gy[sigma][round(x+s),round(y+t)])**2)
                    thetast[x,y][ij][value] = np.arctan2(gy[sigma][round(x+s),round(y+t)], gx[sigma][round(x+s),round(y+t)]) - (radians/(2*np.pi))
                    
                    """tast 4.b"""
                    #create 8 bins for 8 vectors on each point i,j in the grid
                    b = 8*((thetast[x,y][ij][value])/(2*np.pi))
                    hi = np.sum(mst[x,y][ij][value]*wst[x,y][ij][value])
                    hij[x,y][ij][math.floor(b)] += hi
            
            #concatenate 8 value vector of each i,j grid-points - 8*16 = 128 value vector
            for v in hij[x,y][ij]:
                d[x,y].append(v)
        
        """task 4.c"""
        #cap the final 128 d vector at 0.2
        d[x,y] = np.array(d[x,y])
        denominator = np.sqrt(np.dot(d[x,y],d[x,y].T))
        d[x,y] = np.clip((d[x,y]/denominator), None, 0.2)
        # print(len(d[x,y]),np.max(d[x,y]))
            
def non_maxima_3d(doglist, T):
    """Function which checks if the keypoints are a local maxima, in both scale and space, 3D"""
    kPoints = []
    for k,v in doglist.items():
        sigma = k
        index1, index2 = sigmaList.index(sigma)-1, sigmaList.index(sigma)+1
        if index1 == -1:
            index1 = 2
        if index2 == 11:
            index2 = 8
        img0, img1, img2 = doglist[sigmaList[index1]], doglist[sigma], doglist[sigmaList[index1]]
        
        for x in range(1,v.shape[0]-1):
            for y in range (1,v.shape[1]-1):
                if ((img1[x,y]>T) and
                    (img1[x,y]>img1[x-1,y-1]) and
                    (img1[x,y]>img1[x-1,y]) and
                    (img1[x,y]>img1[x-1,y+1]) and
                    (img1[x,y]>img1[x,y-1]) and
                    (img1[x,y]>img1[x,y+1]) and
                    (img1[x,y]>img1[x+1,y-1]) and
                    (img1[x,y]>img1[x+1,y]) and
                    (img1[x,y]>img1[x+1,y+1]) and
                    
                    (img1[x,y]>img0[x,y]) and
                    (img1[x,y]>img0[x-1,y-1]) and
                    (img1[x,y]>img0[x-1,y]) and
                    (img1[x,y]>img0[x-1,y+1]) and
                    (img1[x,y]>img0[x,y-1]) and
                    (img1[x,y]>img0[x,y+1]) and
                    (img1[x,y]>img0[x+1,y-1]) and
                    (img1[x,y]>img0[x+1,y]) and
                    (img1[x,y]>img0[x+1,y+1]) and
                    
                    (img1[x,y]>img2[x,y]) and
                    (img1[x,y]>img2[x-1,y-1]) and
                    (img1[x,y]>img2[x-1,y]) and
                    (img1[x,y]>img2[x-1,y+1]) and
                    (img1[x,y]>img2[x,y-1]) and
                    (img1[x,y]>img2[x,y+1]) and
                    (img1[x,y]>img2[x+1,y-1]) and
                    (img1[x,y]>img2[x+1,y]) and
                    (img1[x,y]>img2[x+1,y+1])):
                    
                    kPoints.append((x,y,sigma))
    return kPoints

def gauss_blur(windowname, image, sigma):
    """Gaussian Smoothing Kernel"""
    size = 3*sigma
    name =  windowname + " Gauss Smoothened Image - sigma " + str(sigma)
    x, y = np.meshgrid(np.arange(-size, size+1),np.arange(-size, size+1))  
    # x,y = np.meshgrid(np.arange(6*sigma),np.arange(6*sigma))
    smooth_kernel = np.exp(-((x-np.mean(x))**2 + (y-np.mean(y))**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))
    smooth_image = cv.filter2D(image, -1, smooth_kernel)
    plt.imshow(smooth_kernel)
    plt.title("Gaussian smoothing kernel with sigma {}".format(sigma))
    plt.show()
    display(name, smooth_image)
    return smooth_image

def display(windowName, image, save = True  ):
    """Display the image, if save is TRUE then save the image"""
    image = image/np.max(image)
    cv.imshow(windowName, image)
    cv.waitKey(0)
    if save:
        savename  = savepath+windowName+'.png'
        if np.max(image) == 1:
            image = image*255
        cv.imwrite(savename, image)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
