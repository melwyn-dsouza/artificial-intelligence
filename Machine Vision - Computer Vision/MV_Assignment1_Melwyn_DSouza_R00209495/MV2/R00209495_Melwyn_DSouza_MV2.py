# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 19:35:39 2022

@author: dsouzm3
"""
import cv2 
import os,time,copy,random, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

random.seed(195877)

def task1():
    img_pts = []
    obj_pts = []                                          
    cord_3d = np.zeros((5*7,3), np.float32)
    cord_3d[:,:2] = np.mgrid[0:7, 0:5].T.reshape(-1,2)
    
    """Task 1 A - Checkboard corners with subpixel accuracy"""
    print("TASK 1 A")        
    calib_images_loc = os.getcwd()+r'\Assignment_MV_02_calibration'
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for i in os.listdir(calib_images_loc):
        rgb_image = cv2.imread(calib_images_loc+'\\'+i)
        grey_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        

        ret, corners = cv2.findChessboardCorners(grey_image, (7,5), None)  
        
        if ret:
            image1 = copy.deepcopy(grey_image)
            rgb_draw = copy.deepcopy(rgb_image)
            corners2 = cv2.cornerSubPix(grey_image, corners, (11,11), (-1,-1), criteria)
            img_pts.append(corners2) 
            obj_pts.append(cord_3d)
        
            img = cv2.drawChessboardCorners(rgb_draw, (7,5), corners2, ret) 
            cv2.imshow("sub-pixel accuracy "+str(i), img)
            # cv2.imwrite("sub-pixel accuracy "+str(i)+".png", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """Task 1 B - Camera Calibraton and Parameters"""
    print("Task 1 B")
    ret,K,d,r,t = cv2.calibrateCamera(obj_pts, img_pts, grey_image.shape[::-1], None, None)
    
    principal_length = K[0][0] # c = f*mx
    aspect_ratio = K[1][1]/K[0][0] # alpha = my/mx
    principal_point = (K[0][2], K[1][2]) # x0,y0
    skew = K[0][1]

    print("Camera Calibration Matrix K:\n\n", K)
    print("\nPrincipal Length:", principal_length)
    print("Aspect Ratio: ", aspect_ratio)
    print("Principal Point: ", principal_point) 
    print("Image Skew (Modern cameras skew == 0): ", skew)
    
    """Task 1 C - Feature tracking""" 
    video = cv2.VideoCapture('Assignment_MV_02_video.mp4')
    
    # initialise features to track the feature points in the first frame - Done
    ret, frame0 =video.read()
    if ret:
        grey_img = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(grey_img, 200, 0.3, 7)                           
        print("Total features in the first frame: {}".format(len(p0)))
        subpixel_p0 = cv2.cornerSubPix(grey_img, p0, (11, 11), (-1, -1), criteria=criteria)

        #display frame 0
        for i in range(len(subpixel_p0)):
            cv2.circle(frame0, (subpixel_p0[i,0,0],subpixel_p0[i,0,1]), 2, (0,0,255), 2)    
        cv2.imshow("Task 1 C Frame 0 feature points", frame0)            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        # initialise tracks
        index = np.arange(len(subpixel_p0))
        tracks = {}

        for i in range(len(subpixel_p0)):
            tracks[index[i]] = {0:subpixel_p0[i]}
            
        print("tracks in frame 0", len(tracks))
        
        """Task 1 D"""
        frame = 0
        while ret:
            ret, img = video.read()

            if not ret:
                break

            frame += 1
            old_img = grey_img
            grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_img, grey_img, p0, None)

                # visualise points
                for i in range(len(st)):
                    if st[i]:
                        cv2.circle(img, (p1[i, 0, 0], p1[i, 0, 1]), 2, (0, 0, 255), 2)
                        cv2.line(img, (p0[i, 0, 0], p0[i, 0, 1]), (int(p0[i, 0, 0] + (p1[i][0, 0] - p0[i, 0, 0]) * 5), int(p0[i, 0, 1] + (p1[i][0, 1] - p0[i, 0, 1]) * 5)), (0, 0, 255), 2)
                        
                p0 = p1[st == 1].reshape(-1, 1, 2)
                index = index[st.flatten() == 1]

            if len(p0) < 100:
                features = cv2.goodFeaturesToTrack(grey_img, 200 - len(p0), 0.3, 7)
                new_p0 = cv2.cornerSubPix(grey_img, features, (11, 11), (-1, -1), criteria=criteria)
                for i in range(len(new_p0)):
                    if np.min(np.linalg.norm((p0 - new_p0[i]).reshape(len(p0), 2), axis=1)) > 10:
                        p0 = np.append(p0, new_p0[i].reshape(-1, 1, 2), axis=0)
                        index = np.append(index, np.max(index) + 1)

            # update tracks
            for i in range(len(p0)):
                if index[i] in tracks:
                    tracks[index[i]][frame] = p0[i]
                else:
                    tracks[index[i]] = {frame: p0[i]}

            # visualise last 20 frames of active tracks
            for i in range(len(index)):
                for f in range(frame - 20, frame):
                    if (f in tracks[index[i]]) and (f + 1 in tracks[index[i]]):
                        cv2.line(img,
                                 (tracks[index[i]][f][0, 0], tracks[index[i]][f][0, 1]),
                                 (tracks[index[i]][f + 1][0, 0], tracks[index[i]][f + 1][0, 1]),
                                 (0, 255, 0), 1)

            k = cv2.waitKey(2)
            if k % 256 == 27:
                print("Escape hit, closing...")
                break

            cv2.imshow("Feature Track Task 1 D", img)
            # cv2.waitKey()
        
        time.sleep(5)
        cv2.destroyAllWindows()
        # video.release()
        print("\nTotal tracks: ", len(tracks))
        print("Total frames: ", frame)
    
    return p0, p1,index, tracks, K, frame



def extract_frames(filename, frames):
    """returns a dictionary of frames"""
    result = {}
    camera = cv2.VideoCapture(filename)
    last_frame = max(frames)
    frame = 0
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            break
        if frame in frames:
            result[frame] = img
        frame += 1
        if frame > last_frame:
            break

    return result


"""Task 2"""
def task2(tracks,index):
    
    """Task 2 A - Extract feature points in frame 0 and frame 30"""
    frame, frame0 = 0,0
    frame30 = 30
    images = extract_frames("Assignment_MV_02_video.mp4", [frame0, frame30])
    
    correspondences = [] #only common between frame 0 adn frame 30 
    for track in tracks:
        if (frame0 in tracks[track]) and (frame30 in tracks[track]):
            x1 = [tracks[track][frame0][0,1],tracks[track][frame0][0,0],1]
            x2 = [tracks[track][frame30][0,1],tracks[track][frame30][0,0],1]
            correspondences.append((np.array(x1), np.array(x2)))
            cv2.circle(images[frame0],(tracks[track][frame0][0,0],tracks[track][frame0][0,1]), 2, (0,0,255), 2)
            cv2.circle(images[frame30],(tracks[track][frame30][0,0],tracks[track][frame30][0,1]), 2, (0,0,255), 2)

    cv2.imshow("Features in both 0 and 30 frame (frame 0)", images[frame0])
    cv2.waitKey(0)         
    cv2.imshow("Features in both 0 and 30 frame (frame 30)", images[frame30])
    cv2.waitKey(0)         
     
    cv2.destroyAllWindows()
       
    print("The total feature points in frame 0 and frame 30 are:", len(correspondences))
    
    best_outliers = len(correspondences)+1
    best_error = 1e100
    
    
    """Task 2 B"""  
    #Mean, standard deviation and T

    x1_mean = np.mean(np.array(correspondences)[:, 0, :2], axis=0)
    x2_mean = np.mean(np.array(correspondences)[:, 1, :2], axis=0)

    x1_std = np.std(np.array(correspondences)[:, 0, :2], axis=0)
    x2_std = np.std(np.array(correspondences)[:, 1, :2], axis=0)


    row1 = [1/x1_std[1], 0, -x1_mean[1]/x1_std[1]]
    row2 = [0, 1/x1_std[0], -x1_mean[0]/x1_std[0]]
    row3 = [0,0,1]
    T1 = np.array([row1,row2,row3])

    row1 = [1/x2_std[1], 0, -x2_mean[1]/x2_std[1]]
    row2 = [0, 1/x2_std[0], -x2_mean[0]/x2_std[0]]
    row3 = [0,0,1]
    T2 = np.array([row1,row2,row3])
    
    print("x1 Mean\n:",x1_mean)
    print("x2_Mean\n:",x2_mean)
    print("x1_std\n:",x1_std)
    print("x2_std\n:",x2_std)
    print("T1\n\n", T1)
    print("T2\n\n", T2)
    

    y1 = (np.matmul(T1,np.array(correspondences)[:, 0, :].T)).T
    y2 = (np.matmul(T2,np.array(correspondences)[:, 1, :].T)).T
    ind = [i for i in range(len(correspondences))]

    """Task 2 Fundamental Matrix """
    for i in range(10000):  #change to 10000      

        inliers, inlier_index = [], []
        count_outliers, count_inliers  = 0,0
        accumulate_error = 0

        """Task 2 C"""

        samples_in = random.sample(ind, 8)
        samples_out = list(set(ind) - set(samples_in))
        A = np.zeros((0,9)) # for 8 pt DLT

        for y11, y22 in zip(y1[samples_in, :], y2[samples_in, :]):
            ai = np.kron(y11.T, y22.T)
            A = np.append(A, [ai], axis=0)
                     
        """Task 2 D"""

        U,S,V = np.linalg.svd(A)    
        Fcap = V[8,:].reshape(3,3).T 
        
        U,S,V = np.linalg.svd(Fcap)
        
        #converting F to singular by dividing it with s[2]
        Fcap = np.matmul(U,np.matmul(np.diag([S[0],S[1],0]),V)) 
        F = np.matmul(T2.T, np.matmul(Fcap, T1))        
          
        
        # print("The first fundamental matrix:\n", F)
        
        """Task 2 E"""
        cxx = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]])
        
        for i in samples_out:                           #For remainder of points find model equation + variance
            x1, x2 = correspondences[i]            
            # gi = np.matmul(x2.T, np.matmul(F, x1))
            gi = x2.T @ F @ x1
            varianceSigma = np.matmul(x2.T, np.matmul(F, np.matmul(cxx, np.matmul(F.T, x2)))) + np.matmul(x1.T, np.matmul(F.T, np.matmul(cxx, np.matmul(F, x1))))
            
            """Task 2_F"""
            Ti = gi**2/varianceSigma
            
            if Ti > 6.635: #outliers
                count_outliers += 1
            else: #inliers 
                count_inliers += 1
                inliers.append((x1,x2))
                inlier_index.append(i)
                accumulate_error += Ti
                
        """Task 2 G"""
        if count_outliers<best_outliers:
            best_error = accumulate_error
            best_outliers_count = count_outliers
            best_inliers_count = count_inliers
            best_inliers_cors = inliers
            best_inliers_index = inlier_index
            best_F = F
        elif count_outliers==best_outliers:
            if accumulate_error<best_error:
                best_error = accumulate_error
                best_outliers_count = count_outliers
                best_inliers_cors = inliers
                best_inliers_index = inlier_index
                best_inliers_count = count_inliers
                best_F = F
    
    print("Fundamental Matrix from best 8 samples: \n\n", best_F)
    print("Sum of test statistics (accumulated_error): ", best_error)
    print("Total number of inliers: ", best_inliers_count)
    print("Total number of outliers: ", best_outliers_count)
     
    """Task 2 H"""
    """Visualize the inliers and outliers from frame 0 to frame 30 points"""
    frame0, frame30 = 0, 30
    video = cv2.VideoCapture('Assignment_MV_02_video.mp4')
    
    frame = 0
    while video.isOpened():
        ret,img= video.read()        
        if frame == 0:
            f0_img = img
        if frame == 30:
            f30_img = img
        if not ret:                                       
            break   
        frame += 1
        
        #visualize points in frame 0 to frame 30
        for i in range(len(index)):
            for f in range(0,frame):   
                if (f in tracks[index[i]]) and (f+1 in tracks[index[i]]):
                    if i in best_inliers_index:
                        cv2.line(img,(tracks[index[i]][f][0,0],tracks[index[i]][f][0,1]),(tracks[index[i]][f+1][0,0],tracks[index[i]][f+1][0,1]),(0,255,0), 1)
                    else:
                        cv2.line(img,(tracks[index[i]][f][0,0],tracks[index[i]][f][0,1]),(tracks[index[i]][f+1][0,0],tracks[index[i]][f+1][0,1]),(0,0,255), 1)
        
        k = cv2.waitKey(1)
        time.sleep(0.05)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        cv2.imshow("Inliers and Outliers", img)  
        
    video.release() 
    
    images = extract_frames("Assignment_MV_02_video.mp4", [frame0, frame30])
    U,S,V = np.linalg.svd(F)    
    e1 = V[2,:]
    U,S,V = np.linalg.svd(F.T)    
    e2 = V[2,:]
    
    print("Epipoles:\n")
    print(e1)
    print(e2)
    print(e1/e1[2])    
    print(e2/e2[2])    
    
    cv2.circle(images[0], (int(e1[0]/e1[2]),int(e1[1]/e1[2])), 3, (0,0,255), 2)
    cv2.imshow('Epipoles F0', images[0])
    cv2.waitKey(0)

    cv2.circle(images[30], (int(e2[0]/e2[2]),int(e2[1]/e2[2])), 3, (0,0,255), 2)
    cv2.imshow('Epi poles F30', images[30])  
    cv2.waitKey(0)

    cv2.destroyAllWindows()   

    return best_F, correspondences, best_inliers_cors, best_inliers_index


def task3(F, K, correspondences, inlier_correspondces):
    """Task 3 A"""
    E = np.matmul(K.T, np.matmul(F,K)) #typically K matrices do not change so I assume K==K'
    print("Essential Matrix:\n\n",E)
    U, S, V = np.linalg.svd(E)

    print("Singular values before:\n", S)

    if np.linalg.det(U) < 0:
        U[:,2] *= -1
    if np.linalg.det(V) < 0:
        V[2,:] *= -1
    
    mean = (S[0]+S[1])/2
    
    E = np.matmul(U,np.matmul(np.diag([mean, mean, 0]),V.T))
    

    print("New Essential Matrix:\n\n",E)

    """Task 3 B"""

    Z = np.array([[0, 1, 0],
                 [-1, 0, 0],
                 [0, 0, 0]])
     
    W = np.array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 1]])
    
    car_speed = 50000 #50km/h
    hour = 60*60 #3600 seconds
    fps = 30 #30fps
    videolenght = 1 #video length
    beta = (car_speed/hour)*fps*videolenght
    
    #four possible solutions
    srtt1  = beta*np.matmul(U,np.matmul(Z,U.T))
    srtt2  = -(beta*np.matmul(U,np.matmul(Z,U.T)))
    rtt1 = np.array([srtt1[2, 1], srtt1[0, 2], srtt1[1, 0]])
    rtt2 = np.array([srtt2[2, 1], srtt2[0, 2], srtt2[1, 0]])
    rotation1 = np.matmul(U,np.matmul(W,V.T))
    rotation2 = np.matmul(U,np.matmul(W.T,V.T))
    translation1 = np.matmul(np.linalg.inv(rotation1), rtt1)
    translation2 = np.matmul(np.linalg.inv(rotation2), rtt2)
    
    print("Rotation matrx 1:\n", rotation1)
    print("Rotation matrx 2:\n", rotation2)
    print("Translation matrx 1:\n", translation1)
    print("Translation matrx 2:\n", translation2)
    
    solutions = [(translation1,rotation1), (translation1,rotation2), (translation2, rotation1), (translation2,rotation2)] #set of 4 solution points
    
    
    """Task 3 C"""
    solution_count =[]
    solution_corr_dict = {}
    i = 0
    for solution in solutions:
        t = solution[0]
        R = solution[1]
        count = 0
        pointsin3d = []
        for correspondence in inlier_correspondces: #inliers
            x1,x2 = correspondence[0], correspondence[1]

            m1 = np.matmul(np.linalg.inv(K), x1)
            m2 = np.matmul(np.linalg.inv(K), x2)
        
            m1 = np.array(m1)
            m2 = np.array(m2)
            
            m1Tm1 = np.matmul(m1.T, m1)
            m2Tm2 = np.matmul(m2.T, m2)
            m1TRm2 = np.matmul(m1.T, np.matmul(R,m2))
            tTm1 = np.matmul(t.T,m1)
            tTRm2 = np.matmul(t.T,np.matmul(R,m2))

            
            lambda_Mue = np.linalg.solve([[m1Tm1,-m1TRm2],[m1TRm2,-m2Tm2]], [tTm1,tTRm2])
            
            if (np.all(lambda_Mue>0)):
                count += 1
                xlamda = lambda_Mue[0] * m1
                xMue = t + np.multiply(lambda_Mue[1], np.matmul(R, m2))
                pointsin3d.append([xlamda,xMue])
        solution_corr_dict[i] = pointsin3d
        solution_count.append(count)
        i += 1
        
        
    bestSolIndex = np.argmax(solution_count)
    
    print("The best solution is:\n{}".format(solutions[bestSolIndex]))
    print("The total points infront of both frames from the best soution are: \n{}".format(solution_count[bestSolIndex]))
    
    """Task 3 D E"""
    ax = Axes3D(plt.figure())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    best3dcors = solution_corr_dict[bestSolIndex]
    x, y, z = np.array(best3dcors)[:, 0, 0], np.array(best3dcors)[:, 0, 1], np.array(best3dcors)[:, 0, 2]
    ax.scatter3D(x, y, z, marker='o', c='green')

    ax.plot([0.], [0.], [0.], marker='+', c='red')

    ax.plot(solutions[bestSolIndex][0][0], solutions[bestSolIndex][0][1], solutions[bestSolIndex][0][2], marker='o', c='blue')

    plt.show()

    
    
def main():
    p0, p1, index, tracks, K ,frames= task1()
    F, correspondences, inlier_cords, inilier_ind = task2(tracks, index) 
    task3(F, K, correspondences, inlier_cords)
    
    
    
if __name__ =="__main__":
    main()