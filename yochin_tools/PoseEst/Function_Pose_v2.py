import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import scipy.io as io    
import h5py
import yo_network_info

basePath = os.path.join(yo_network_info.PATH_BASE, 'yochin_tools/PoseEst/DBv1')
TH_CORRESPONDENCES = yo_network_info.POSE_EST_TH_CORRESPONDENCES
TH_INLIERS = yo_network_info.POSE_EST_TH_INLIERS

def ReadDB(classname) : 
    ftr = io.loadmat(os.path.join(basePath, str(classname).lower(), str(classname).lower() + '_DB1.mat'))
    FeatureDB=np.array(ftr['FeatureDB'])
    FeatureDB = FeatureDB.astype("float32")
    keypointDB=np.array(ftr['keypointDB'])
    keypointDB = keypointDB.astype("float32")
    CoorDB=np.array(ftr['CoorDB'])
    CoorDB = CoorDB.astype("float32")
    return FeatureDB,CoorDB, keypointDB

# added by yochin
# MaxIterRansac : to avoid infinite loop.
def PoseEstimate(img, FeatureDB, CoorDB, ret, init_coord, MaxIterRansac=1000):      #image, left_upper point of boundingBox, ReadDB-FeatureDB, ReadDB-CoorDB, CameraMatrix, CameraDistortion
    # init_coord [x, y, -]
    try:
        imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.float32)/25
        img = img.astype('float32')
        img = img-np.min(img)
        img = img/np.max(img)
        img = img*128/np.mean(img)
        img = img.astype('uint8')
        imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgg2 = cv2.filter2D(imgg,-1,kernel)
        # imgg = cv2.medianBlur(imgg, 3)
        # opencv3.~
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=4, nOctaveLayers=3, extended=False, upright=False)
        kp, descritors = surf.detectAndCompute(imgg,None)

        #for ptDisp in kp:
        #     cv2.circle(img, (int(ptDisp.pt[0]), int(ptDisp.pt[1])), 1, (255, 255, 255, 0), -1)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=300)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(descritors,FeatureDB,k=3) #k controls the numbers of matches per each descriptor
        matches = sorted(matches, key=lambda x:x[0].distance/(x[1].distance+0.5*x[2].distance))
        # matches=matches[0:max(int(round(len(matches)/2)),80)]
        good = matches
        #for ptDisp in good:
        #    cv2.circle(img, (int(kp[ptDisp.queryIdx].pt[0]), int(kp[ptDisp.queryIdx].pt[1])), 2,
        #               (int((CoorDB[ptDisp.trainIdx,0]+100)/200.*255.),
        #                int((CoorDB[ptDisp.trainIdx,1]+50)/100.*255.),
        #                int((CoorDB[ptDisp.trainIdx,2]+30)/60.*255.), 0), -1)


        print('\tnum corrs : %d' % (len(good)))

        if len(good) > TH_CORRESPONDENCES:
            objpoints = np.zeros((len(matches),3),np.float32)
            imgpoints=np.zeros((len(matches),2),np.float32)
            tempCoord = np.zeros((1,3),np.float32)
            tempkey = np.zeros((1,2),np.float32)
            for m in xrange(len(matches)):
                tempCoord[:,0] = CoorDB[matches[m][0].trainIdx,0]
                tempCoord[:,1] = CoorDB[matches[m][0].trainIdx,1]
                tempCoord[:,2] = CoorDB[matches[m][0].trainIdx,2]
                objpoints[m,:] = tempCoord
                tempkey[:,0] = kp[matches[m][0].queryIdx].pt[1]
                tempkey[:,1] = kp[matches[m][0].queryIdx].pt[0]
                imgpoints[m,:] = tempkey
            objpoints = objpoints.astype('float32')
            imgpoints = imgpoints.astype('float32')
            # imgpoints[:,0] = imgpoints[:,0] + init_coord[1]
            # imgpoints[:,1] = imgpoints[:,1] + init_coord[0]
            objpoints = np.expand_dims(objpoints, axis=0)
            imgpoints = np.expand_dims(imgpoints, axis=0)
            imgpoints[:,:,0] = imgpoints[:,:,0]+init_coord[1]
            imgpoints[:,:,1] = imgpoints[:,:,1]+init_coord[0]
            # ret = np.zeros((3,3))
            # ret[0,0] = 1158.03
            # ret[1,1] = 1158.03
            # ret[0,2] = 540.
            # ret[1,2] = 960.
            # ret[2,2] = 1

            rvec = np.expand_dims([ 0.04110314, -2.37024752, 0.37337192], axis = 1)
            tvec = np.expand_dims([  0., 0., -9999], axis = 1)
            dist = np.array([[0.0, 0.0, 0.0, 0.0]])
            (_, rvec,tvec,inliers)= cv2.solvePnPRansac(objpoints, imgpoints, ret, dist, rvec = rvec, tvec = tvec, reprojectionError=3.0) # flags = cv2.SOLVEPNP_P3P, rvec=rvec, tvec=tvec, useExtrinsicGuess=True
            # (rvec,tvec,inliers)= cv2.solvePnPRansac(objpoints, imgpoints, ret, dist,reprojectionError=3.0)
            ErrorThreshold = 3.0
            IterRansac = 0

            while len(inliers)<int(round(len(matches)/4)) or tvec[2][0]<0:
                ErrorThreshold = ErrorThreshold + 3
                (_, rvec,tvec,inliers)= cv2.solvePnPRansac(objpoints, imgpoints, ret, dist,reprojectionError=ErrorThreshold)

                # print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
                # print(rvec)
                # print(tvec[2][0])
                # print('<<<<<<<<<<<<<<<<<<<<<<<<<<')

                # added by yochin
                # MaxIterRansac : to avoid infinite loop.
                IterRansac = IterRansac + 1
                if IterRansac > MaxIterRansac:
                    print('IterRansac > MaxIterRansac')
                    inliers = None
                    break

            print('IterRansac: %d'%IterRansac)
                # print(tvec[2][0])
			#(_, rvec, tvec) = cv2.solvePnP(objpoints, imgpoints, ret, dist)  # flags = cv2.SOLVEPNP_P3P, iterationsCount=1000, rvec=rvec, tvec=tvec, useExtrinsicGuess=True

            # # for debugging
            # print(rvec)
            # print(tvec)
            # for iii in inliers:
            #     cv2.circle(img, (int(imgpoints[0,iii,1]-init_coord[0]),int(imgpoints[0,iii,0]-+init_coord[1])), 2,
            #                (int((objpoints[0,iii, 0] + 100) / 200. * 255.),
            #                 int((objpoints[0,iii, 1] + 50) / 100. * 255.),
            #                 int((objpoints[0,iii, 2] + 30) / 60. * 255.), 0), -1)


            if inliers is not None:
                print('\tnum inliers : %d' % (len(inliers)))
                if len(inliers) > TH_INLIERS:
                    # tvec[0] = 0
                    # tvec[2] = -tvec[2]+745
                    # tvec[0] = -tvec[2]
                    rmat = cv2.Rodrigues(rvec)[0]
                    # rmat2 = rmat[[2,1,0],:]
                    # rmat2 = rmat2[:,[2,1,0]]
                    # rmat2[[0,2],:] = -rmat2[[0,2],:]
                else:
                    rmat = np.zeros((3, 3))
                    tvec = np.zeros((3, 1))
            else:
                print('\tnum inliers : 0')
                rmat = np.zeros((3, 3))
                tvec = np.zeros((3, 1))
        else:
            rmat = np.zeros((3, 3))
            tvec = np.zeros((3, 1))
    except:
        rmat = np.zeros((3, 3))
        tvec = np.zeros((3, 1))

    # cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
    # cv2.imshow('cropped', img)

    return rmat, tvec


# def PoseEstimateBF(img, FeatureDB, CoorDB,ret,init_coord):      #image, left_upper point of boundingBox, ReadDB-FeatureDB, ReadDB-CoorDB, CameraMatrix, CameraDistortion
#     imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # opencv2.~
#     # surf = cv2.SURF()
#     # kp, descritors = surf.detectAndCompute(imgg, None)
#
#     # opencv3.~
#     surf = cv2.xfeatures2d.SURF_create()
#     kp, descritors = surf.detectAndCompute(imgg, None)
#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#     search_params = dict(checks=150)
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
#     matches = bf.match(descritors,FeatureDB)
#     # matches = bf.knnMatch(descritors, FeatureDB, k=2)
#     threshold = 0.8
#     threshold_mini_up = 0.14
#     threshold_mini_down = 0.07
#     candidate1 = np.zeros((len(matches),1))
#     candidate2 = np.zeros((len(matches),1))
#     kp2 = np.zeros((len(kp),2))
#     cnt = 0
#     for jj in xrange((len(matches))):
#         if matches[jj].distance < threshold_mini_up and matches[jj].distance > threshold_mini_down:
#             candidate1[cnt,0] = matches[jj].queryIdx
#             candidate2[cnt,0] = matches[jj].distance
#             kp2[cnt,0] = kp[matches[jj].queryIdx].pt[0]
#             kp2[cnt,1] = kp[matches[jj].queryIdx].pt[1]
#             cnt = cnt+1
#     candidate1 = candidate1[0:cnt-1,:]
#     candidate1 = candidate1.astype("int")
#     candidate2 = candidate2[0:cnt-1,:]
#     kp2 = kp2[0:cnt-1,:]
#     descritors = descritors[candidate1,:]
#     matches2 = np.zeros((len(descritors),2))
#     cnt = 0
#     for jj in xrange((len(descritors))):
#         dist = np.sum((FeatureDB-descritors[jj,:])**2,axis=1)
#         mini = np.min(dist)
#         mini_ind = np.argmin(dist)
#         dist[mini_ind] = 9999
#         second_mini = np.min(dist)
#         second_mini_ind = np.argmin(dist)
#         dist[mini_ind] = mini
#         if np.sqrt(mini)/np.sqrt(second_mini)  < threshold:
#             matches2[cnt,0] = jj
#             matches2[cnt,1] = mini_ind
#             cnt = cnt+1
#
#     print('\tnum corrs : %d' % (cnt))
#
#     if cnt > TH_CORRESPONDENCES :
#         matches = matches2[0:cnt-1,:]
#         matches=matches.astype("int")
#         objpoints = np.zeros((len(matches),3),np.float32)
#         imgpoints=np.zeros((len(matches),2),np.float32)
#         tempCoord = np.zeros((1,3),np.float32)
#         tempkey = np.zeros((1,2),np.float32)
#         for m in xrange(len(matches)):
#             tempCoord[:,0] = CoorDB[matches[m,1],0]
#             tempCoord[:,1] = CoorDB[matches[m,1],1]
#             tempCoord[:,2] = CoorDB[matches[m,1],2]
#             objpoints[m,:] = tempCoord
#             tempkey[:,0] = kp2[matches[m,0],1]
#             tempkey[:,1] = kp2[matches[m,0],0]
#             imgpoints[m,:] = tempkey
#         objpoints = objpoints.astype('float32')
#         imgpoints = imgpoints.astype('float32')
#         # imgpoints[:,0] = imgpoints[:,0] + init_coord[1]
#         # imgpoints[:,1] = imgpoints[:,1] + init_coord[0]
#         objpoints = np.expand_dims(objpoints, axis=0)
#         imgpoints = np.expand_dims(imgpoints, axis=0)
#         imgpoints[:,:,0] = imgpoints[:,:,0]+init_coord[1]
#         imgpoints[:,:,1] = imgpoints[:,:,1]+init_coord[0]
#         # ret = np.zeros((3,3))
#         # ret[0,0] = 1158.03
#         # ret[1,1] = 1158.03
#         # ret[0,2] = 540.
#         # ret[1,2] = 960.
#         # ret[2,2] = 1
#         dist = np.array([[0.0, 0.0, 0.0, 0.0]])
#
#         (_,rvec,tvec,inliers)= cv2.solvePnPRansac(objpoints, imgpoints, ret, dist)
#
#         if inliers is not None:
#             print('\tnum inliers : %d'%(len(inliers)))
#             if len(inliers) > TH_INLIERS:
#                 # tvec[0] = 0
#                 # tvec[2] = -tvec[2]+745
#                 # tvec[0] = -tvec[2]
#                 rmat = cv2.Rodrigues(rvec)[0]
#                 # rmat2 = rmat[[2,1,0],:]
#                 # rmat2 = rmat2[:,[2,1,0]]
#                 # rmat2[[0,2],:] = -rmat2[[0,2],:]
#             else:
#                 rmat = np.zeros((3, 3))
#                 tvec = np.zeros((3, 1))
#         else:
#             print('\tnum inliers : 0')
#             rmat = np.zeros((3, 3))
#             tvec = np.zeros((3, 1))
#     else:
#         rmat = np.zeros((3, 3))
#         tvec = np.zeros((3, 1))
#     return rmat, tvec


    
def computeTransfrom(point,rmat,tvec,ret,init_coord):
    A = point
    rs = np.dot(ret,np.dot(rmat,np.transpose(A,[1,0]))+tvec)
    rs = rs/rs[2]
    rs = rs[[1,0,2],:]
    rs[0,:] = rs[0,:] -init_coord[0]
    rs[1,:] = rs[1,:] -init_coord[1]
    return rs

def cornerpointsTransform(img,rmat,tvec,ret,init_coord):
    # filelist2=os.listdir(os.path.join(basePath, str(classname) , 'geom2'))
    # filelist2 = np.sort(filelist2)
    # strTRSet=os.path.join(os.path.join(basePath, str(classname) , 'geom2',filelist2[0] ))
    # ftr=h5py.File(strTRSet, 'r')
    # img=np.array(ftr['img'])
    # img = np.transpose(img,[2,1,0])
    # above commented part is moved into the initialization part to improve processing time.
    ch1 = img[:,:,2]
    ch2 = img[:,:,1]
    ch3 = img[:,:,0]
    max1 = np.max((np.max(ch1),np.min(ch1)))
    max2 = np.max((np.max(ch2),np.min(ch2)))
    max3 = np.max((np.max(ch3),np.min(ch3)))
    min1 = -max1
    min2 = -max2
    min3 = -max3
    Dots = np.zeros((8,3))
    Dots[0,:] = np.array((max1,max2,max3))
    Dots[1,:] = np.array((min1,max2,max3))
    Dots[2,:] = np.array((max1,min2,max3))
    Dots[3,:] = np.array((max1,max2,min3))
    Dots[4,:] = np.array((min1,min2,max3))
    Dots[5,:] = np.array((min1,max2,min3))
    Dots[6,:] = np.array((max1,min2,min3))
    Dots[7,:] = np.array((min1,min2,min3))
    point = np.zeros((1,3))
    Result=np.zeros((8,3))
    for ii in xrange(0,8):
        point[:,0] = Dots[ii,0]
        point[:,1] = Dots[ii,1]
        point[:,2] = Dots[ii,2]
        rs = computeTransfrom(point,rmat,tvec,ret,init_coord)
        Result[ii,0] = rs[0,:]
        Result[ii,1] = rs[1,:]
        Result[ii,2] = rs[2,:]
    return Result

def cornerpointsTransform2(img,rmat,tvec,ret,init_coord):
    # filelist2=os.listdir(os.path.join(r'I:\PoseEstimation\function' , str(classname) , 'geom2'))
    # filelist2 = np.sort(filelist2)
    # strTRSet=os.path.join(os.path.join(r'I:\PoseEstimation\function' , str(classname) , 'geom2',filelist2[0] ))
    # ftr=h5py.File(strTRSet, 'r')
    # img=np.array(ftr['img'])
    # img = np.transpose(img,[2,1,0])
    # above commented part is moved into the initialization part to improve processing time.
    ch1 = img[:,:,2]
    ch2 = img[:,:,1]
    ch3 = img[:,:,0]
    max1 = np.max((np.max(ch1),np.min(ch1)))
    max2 = np.max((np.max(ch2),np.min(ch2)))
    max3 = np.max((np.max(ch3),np.min(ch3)))
    Dots = np.zeros((4,3))
    Dots[0,:] = np.array((0,0,0))
    Dots[1,:] = np.array((max1,0,0))
    Dots[2,:] = np.array((0,max2,0))
    Dots[3,:] = np.array((0,0,max3))
    point = np.zeros((1,3))
    Result=np.zeros((4,3))
    for ii in xrange(0,4):
        point[:,0] = Dots[ii,0]
        point[:,1] = Dots[ii,1]
        point[:,2] = Dots[ii,2]
        rs = computeTransfrom(point,rmat,tvec,ret,init_coord)
        Result[ii,0] = rs[0,:]
        Result[ii,1] = rs[1,:]
        Result[ii,2] = rs[2,:]
    return Result
    
    
    


##################################
if __name__ == '__main__':
    FeatureDB,CoorDB, keypointDB = ReadDB('chococo')
    filelist=os.listdir(r'I:\PoseEstimation\function\chococo\test2')
    fileind = 0
    img =cv2.imread(os.path.join(r'I:\PoseEstimation\function\chococo\test2',filelist[fileind]))
    ret = np.zeros((3,3))
    # MS C920
    ret[0,0] = 1158.03
    ret[1,1] = 1158.03
    ret[0,2] = 540.
    ret[1,2] = 960.
    ret[2,2] = 1
    dist = np.array([[0.0, 0.0, 0.0, 0.0]])
    gap =1
    FeatureDB2 = FeatureDB[::gap,:]
    CoorDB2 = CoorDB[::gap,:]
    init_coord = np.array([278,461])
    rmat, tvec = PoseEstimate(img,FeatureDB2,CoorDB2,ret,init_coord)
    rmat2 = rmat[[2,1,0],:]
    rmat2 = rmat2[:,[2,1,0]]
    rmat2[[0,2],:] = -rmat2[[0,2],:]
    Result = cornerpointsTransform('chococo',rmat,tvec,ret,init_coord)

    # # to display points
    # img2 = img
    # Result = Result.astype("int")
    # for ii in xrange(0,8):
    #     cv2.circle(img2,(Result[ii,0],Result[ii,1]), 5, (255,0,0), -1)
    #
    # cv2.imshow('crop2',img2)
    # cv2.waitKey(0)


    # Result = cornerpointsTransform2('chococo',rmat,tvec,ret,init_coord)

    # img2 = img
    # Result = Result.astype("int")
    # for ii in xrange(0,4):
        # cv2.circle(img2,(Result[ii,0],Result[ii,1]), 5, (255,0,0), -1)

    # cv2.imshow('crop2',img2)
    # cv2.waitKey(0)