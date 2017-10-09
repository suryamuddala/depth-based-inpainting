#!/usr/bin/python

import matplotlib.pyplot as plt
import sys, os
import cv2
import numpy as np
import time
np.seterr(divide='ignore', invalid='ignore')
from skimage.filters import threshold_otsu

class DisoccllusionFill():
    textureImaged= None
    depthImaged= None
    sourceRegion = None
    fillRegion = None
    confidenceImage = None
    dataImage = None
    curvatureData = None
    boundaryImage = None
    halfPatchSize = None
    numMissingPixels = None
    kernel = None
    kernelsel = None
    depthThreshold = None
    depthThresholdsR = None

    inpaintedImage = None
    inpaintedDepth = None
    kurv = None
    fillBoundaylist = []
    prioritiesList = []

    targetPatchCenter = None
    targetPatch_tl = None
    targetPatch_br = None
    targetPatch_sRmsk = None
    targetPatch_fRmsk = None
    targetPatch_tex = None
    targetPatch_dep = None

    sourcePatchCenter = None
    sourcePatch_tl = None
    sourcePatch_br = None
    sourcePatch_tex = None
    sourcePatch_dep = None
    bestErr = 1000000000.0
    searchRegion = 90
    patchErr = None

    def __init__(self, textureImage, depthImage, fillColor=[0, 0, 0], halfPatchSize=5, kernelsel=0, depthThreshold=120):
        self.textureImaged = np.float32(textureImage.copy())
        self.depthImaged = np.float32(depthImage.copy())
        self.fillColor  = fillColor
        self.halfPatchSize = halfPatchSize
        self.img_shape = textureImage.shape
        self.height, self.width = self.img_shape[:2]
        self.kernelsel = kernelsel
        self.depthThreshold = depthThreshold
        self.inpaintedImage = np.zeros(self.img_shape, dtype = textureImage.dtype)

    def iniatializeInpainting(self):

        self.depthThresholdsR = threshold_otsu(self.depthImaged)
        #print("globalThreshold", self.depthThresholdsR)
        self.extractDisocclusionregion()
        if self.kernelsel == 0:
            self.kernel = np.ones((3, 3), dtype = np.float32)
            self.kernel[1, 1] = -8
        elif self.kernelsel == 1: #left
            self.kernel = np.zeros((3, 3), dtype = np.float32)
            self.kernel[1, 0] = 1
            self.kernel[1, 1] = -1
        else:
            self.kernel = np.zeros((3, 3), dtype = np.float32)
            self.kernel[1, 1] = -1
            self.kernel[1, 2] = 1
        self.cofidenceImage = np.copy(self.fillRegion)/255
        self.dataImage = np.zeros((self.height,self.width), dtype=np.float32)
        self.computeStructure()
        self.numMissingPixels = np.count_nonzero(self.fillRegion)
        print("Number of pixels to be inpainted {}".format(self.numMissingPixels))

    def inpaintDisocclusion(self):
        self.iniatializeInpainting()
        iter =0
        while self.numMissingPixels>0:
            iter += 1
            #print("iter:{}".format(iter))
            #extractboundary
            self.computeFillboundary()
            #computeconfidence along boundary
            self.computeConfidenceTerm()
            #compute data term
            self.computeDataTerm()
            #compute priority
            self.computePriority()
            #select target patch
            self.getTargetpatch()
            #find sourcepatch
            self.findSourcePatch()
            #Update dataterm, fillregion, image, depth
            self.fillandUpdate()

            ###---visualization
            """
            tempdata = np.uint8(np.copy(self.curvatureData))
            temp = np.uint8(np.copy(self.textureImaged))
            temp[self.targetPatch_tl[0]:self.targetPatch_br[0], self.targetPatch_tl[1]:self.targetPatch_br[1],:] = (255, 0 , 0)
            temp[self.sourcePatch_tl[0]:self.sourcePatch_br[0], self.sourcePatch_tl[1]:self.sourcePatch_br[1],:] = (0, 0 , 255)
            tempd = np.uint8(np.copy(self.depthImaged))
            tempd[self.targetPatch_tl[0]:self.targetPatch_br[0], self.targetPatch_tl[1]:self.targetPatch_br[1]] = (255)
            tempd[self.sourcePatch_tl[0]:self.sourcePatch_br[0], self.sourcePatch_tl[1]:self.sourcePatch_br[1]] = (125)
            #plot
            plt.imshow(tempdata)
            plt.show()
            plt.imshow(temp)
            plt.show()
            #plt.imshow(tempd)
            #plt.show()
            ### ----
            """
            #check if the holes still left!
            self.extractDisocclusionregion()
            self.numMissingPixels = np.count_nonzero(self.fillRegion)
            #print("Number of pixels to be inpainted {}".format(self.numMissingPixels))

        self.inpaintedImage = np.uint8(np.copy(self.textureImaged))
        self.inpaintedDepth = np.uint8(np.copy(self.depthImaged))

        ## disocclusion Mask
    def extractDisocclusionregion(self):
        #create a mask of disocclusion and its inverse mask
        #print("(self.height, self.width)", (self.height, self.width))
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        if len(self.img_shape)>2:
            mask[np.where((self.textureImaged == self.fillColor).all(axis = 2))] = 255
        else:
            mask[np.where(self.textureImaged == self.fillColor)] = 255
        _,self.fillRegion = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        _,self.sourceRegion = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY_INV)

    def computeGradients(self, mat):
        Ix = np.zeros((self.height, self.width), dtype = np.float32)
        Iy = np.zeros((self.height, self.width), dtype = np.float32)
        #Inside boundary of Image
        for row in range(1, self.height-1):
            for col in range(1, self.width-1):
                if self.fillRegion[row][col]==0:
                    #For Ix
                    if (self.fillRegion[row][col-1]==0) and (self.fillRegion[row][col+1]==0):
                        Ix[row][col]= max(abs(mat[row][col]-mat[row][col-1]), abs(mat[row][col+1]-mat[row][col]))

                    if (self.fillRegion[row][col-1]==0) and (self.fillRegion[row][col+1]!=0):
                        Ix[row][col]= abs(mat[row][col]-mat[row][col-1])

                    if (self.fillRegion[row][col-1]!=0) and (self.fillRegion[row][col+1]==0):
                        Ix[row][col]= abs(mat[row][col+1]-mat[row][col])

                    #For Iy
                    if (self.fillRegion[row-1][col]==0) and (self.fillRegion[row+1][col]==0):
                        Iy[row][col]= max(abs(mat[row][col]-mat[row-1][col]), abs(mat[row+1][col]-mat[row][col]))

                    if (self.fillRegion[row-1][col]==0) and (self.fillRegion[row+1][col]!=0):
                        Iy[row][col]= abs(mat[row][col]-mat[row-1][col])

                    if (self.fillRegion[row-1][col]!=0) and (self.fillRegion[row][col+1]==0):
                        Iy[row][col]= abs(mat[row+1][col]-mat[row][col])
        #Boundary of Image
        #Left and Right Ix:
        for row in range(self.height):
            for col in range(0, self.width, self.width-1):
                if (col==0) and (self.fillRegion[row][col]==0) and (self.fillRegion[row][col+1]==0):
                    Ix[row][col] = abs(mat[row][col+1]-mat[row][col])
                if (col==self.width-1) and (self.fillRegion[row][col]==0) and (self.fillRegion[row][col-1]==0):
                    Ix[row][col] = abs(mat[row][col]-mat[row][col-1])
        #Top and bottom Iy:
        for row in range(0, self.height, self.height-1):
            for col in range(0, self.width):
                if (row==0) and (self.fillRegion[row][col]==0) and (self.fillRegion[row+1][col]==0):
                    Ix[row][col] = abs(mat[row+1][col]-mat[row][col])
                if (row==self.height-1) and (self.fillRegion[row][col]==0) and (self.fillRegion[row-1][col]==0):
                    Ix[row][col] = abs(mat[row][col]-mat[row-1][col])
        return Ix, Iy

    def getexemPatch(self, pixel, pSize):
        row_center, col_center = pixel
        min_row = max(row_center - pSize, 0)
        max_row = min(row_center + pSize+1, self.height)
        min_col = max(col_center - pSize, 0)
        max_col = min(col_center + pSize+1, self.width)
        topLeft = (min_row, min_col)
        bottomRight = (max_row, max_col)
        return topLeft, bottomRight

    def computeFillboundary(self):
        fillRegionD = self.fillRegion/255
        boundaryImage = cv2.filter2D(fillRegionD, -1, self.kernel)
        idx = np.where(boundaryImage >0)
        ridx, cidx = idx
        del self.fillBoundaylist[:]
        for i in range(len(ridx)):
            if self.depthImaged[ridx[i], cidx[i]] < self.depthThreshold :
                self.fillBoundaylist.append((ridx[i], cidx[i]))
        if self.numMissingPixels>0 and len(self.fillBoundaylist)==0:
            self.kernelsel = 2
            boundaryImage = cv2.filter2D(fillRegionD, -1, self.kernel)
            idx = np.where(boundaryImage >0)
            ridx, cidx = idx
            for i in range(len(ridx)):
                self.fillBoundaylist.append((ridx[i], cidx[i]))

    def computeConfidenceTerm(self):
        for pixel in self.fillBoundaylist:
            Hq_tl, Hq_br = self.getexemPatch(pixel, self.halfPatchSize)
            Hq_fR = self.sourceRegion[Hq_tl[0]:Hq_br[0], Hq_tl[1]:Hq_br[1]]
            nZ_Hq = np.float(np.count_nonzero(Hq_fR))
            pinHq = np.float(Hq_fR.size)
            #print("Hq_fR", Hq_fR)
            #print("NumofNonZero {} out of {}, {}".format(nZ_Hq, pinHq, nZ_Hq/pinHq))
            if(Hq_fR.size ==0):
                #print(pixel, Hq_fR)
                print("patchErr")
            self.cofidenceImage[pixel[0],pixel[1]] = float(nZ_Hq/pinHq)

    def computeStructure(self):
        textureGray = cv2.cvtColor(self.textureImaged, cv2.COLOR_RGB2GRAY)
        texturegd = np.float32(textureGray)
        Itx, Ity = self.computeGradients(texturegd)
        Idx, Idy = self.computeGradients(self.depthImaged)
        Ix = (Itx+Idx)/(2*255)
        Iy = (Ity+Idy)/(2*255)
        #INr = np.linalg.norm(Ix+Iy)
        INr = np.sqrt(Ix**2 + Iy**2)
        INx = Ix/INr
        INy = Iy/INr
        INx[np.isnan(INx)] = 0
        INy[np.isnan(INy)] = 0
        INx_x,_=self.computeGradients(INx)
        _,INy_y=self.computeGradients(INy)
        Kurv = np.divide(INx_x+INy_y,INr)
        nanLoc = np.isnan(Kurv)
        infLoc = np.isinf(Kurv)
        Kurv[nanLoc] = 0
        Kurv[infLoc] = 0
        #print("np.max(kurv)",np.max(Kurv))
        self.curvatureData = Kurv
        self.curvatureData[nanLoc] = 0
        self.curvatureData[infLoc] = 0
        #self.curvatureData = INr
        #self.curvatureData = (INx_x+INy_y)/INr
        #print("np.max(self.curvatureData)",np.max(self.curvatureData))

    def computeDataTerm(self):
        maxalongfillBoundary = 0
        for pixel in self.fillBoundaylist:
            if maxalongfillBoundary<self.curvatureData[pixel[0],pixel[1]]:
                maxalongfillBoundary = self.curvatureData[pixel[0],pixel[1]]
        #print("maxalongfillBoundary", maxalongfillBoundary)
        for pixel in self.fillBoundaylist:
            self.dataImage[pixel[0],pixel[1]] = abs(1-(self.curvatureData[pixel[0],pixel[1]]/maxalongfillBoundary))
            #print("datavalues, K, D, maxB:", self.curvatureData[pixel[0],pixel[1]], self.dataImage[pixel[0],pixel[1]], maxalongfillBoundary, self.curvatureData[pixel[0],pixel[1]]/maxalongfillBoundary)

    def computePriority(self):
        del self.prioritiesList[:]
        for pixel in self.fillBoundaylist:
            #print("C and D:",self.cofidenceImage[pixel[0],pixel[1]], self.dataImage[pixel[0],pixel[1]])
            Pr = self.cofidenceImage[pixel[0],pixel[1]]*self.dataImage[pixel[0],pixel[1]]
            self.prioritiesList.append(Pr)
        #print("lenbounday",len(self.fillBoundaylist))
        #print("self.prioritiesList",self.prioritiesList)
        #print("max",np.max(self.prioritiesList))
        #print("location",self.prioritiesList.index(np.max(self.prioritiesList)))
        self.targetPatchCenter = self.fillBoundaylist[self.prioritiesList.index(np.max(self.prioritiesList))]

    def getTargetpatch(self):
        self.targetPatch_tl, self.targetPatch_br = self.getexemPatch(self.targetPatchCenter, self.halfPatchSize)
        self.targetPatch_fR = self.fillRegion[self.targetPatch_tl[0]:self.targetPatch_br[0], self.targetPatch_tl[1]:self.targetPatch_br[1]]
        self.targetPatch_sR = self.sourceRegion[self.targetPatch_tl[0]:self.targetPatch_br[0], self.targetPatch_tl[1]:self.targetPatch_br[1]]
        self.targetPatch_tex = self.textureImaged[self.targetPatch_tl[0]:self.targetPatch_br[0], self.targetPatch_tl[1]:self.targetPatch_br[1], :]
        self.targetPatch_dep = self.depthImaged[self.targetPatch_tl[0]:self.targetPatch_br[0], self.targetPatch_tl[1]:self.targetPatch_br[1]]
        self.targetPatch_sRmsk = np.where(self.targetPatch_sR>0)
        self.targetPatch_fRmsk = np.where(self.targetPatch_fR>0)

    def findSourcePatch(self):
        patch_shape = self.targetPatch_tex.shape
        bound = int(round(self.searchRegion/2))
        #print(bound)
        searchReg_tl, searchReg_br = self.getexemPatch(self.targetPatchCenter, bound)
        searchReg_depth = self.depthImaged[searchReg_tl[0]:searchReg_br[0], searchReg_tl[1]:searchReg_br[1]]
        depthbased_sourceRegFg = self.depthImaged > self.depthThresholdsR + self.fillRegion
        self.sourcePatch_tl = None
        self.sourcePatch_br = None
        self.bestErr = 1000000000.0
        for row in range(searchReg_tl[0], searchReg_br[0]-patch_shape[0]):
            for col in range(searchReg_tl[1], searchReg_br[1]-patch_shape[1]):
                #print(row, col)
                Hq_fR = self.fillRegion[row:row+patch_shape[0], col:col+patch_shape[1]]
                Hq_nz = np.count_nonzero(Hq_fR)
                #depth based classification of source region
                Hq_FG = depthbased_sourceRegFg[row:row+patch_shape[0], col:col+patch_shape[1]]
                Hq_FGnz = np.count_nonzero(Hq_FG)
                if Hq_nz == 0 and Hq_FGnz == 0:
                    Hq_tex = self.textureImaged[row:row+patch_shape[0], col:col+patch_shape[1], :]
                    Hq_dep = self.depthImaged[row:row+patch_shape[0], col:col+patch_shape[1]]
                    err = np.sum(self.targetPatch_tex[self.targetPatch_sRmsk]-Hq_tex[self.targetPatch_sRmsk]) + \
                    2*np.sum(self.targetPatch_dep[self.targetPatch_sRmsk]-Hq_dep[self.targetPatch_sRmsk])
                    self.patchErr = err*err
                    if self.patchErr < self.bestErr:
                        self.bestErr = self.patchErr
                        self.sourcePatch_tl = (row, col)
                        self.sourcePatch_br = (row+patch_shape[0], col+patch_shape[1])
                        #print(patchErr)
        #print("sourcePatch:", self.sourcePatch_tl, self.sourcePatch_br, self.patchErr)

    def copyData(self,vImg,debug):
        vsourcePatch = vImg[self.sourcePatch_tl[0]:self.sourcePatch_br[0], self.sourcePatch_tl[1]:self.sourcePatch_br[1]]
        vtargetPatch = vImg[self.targetPatch_tl[0]:self.targetPatch_br[0], self.targetPatch_tl[1]:self.targetPatch_br[1]]
        vtargetPatch[self.targetPatch_fRmsk] = vsourcePatch[self.targetPatch_fRmsk]
        vImg[self.targetPatch_tl[0]:self.targetPatch_br[0], self.targetPatch_tl[1]:self.targetPatch_br[1]] = vtargetPatch
        if debug==1:
            print("vtargetPatch", np.uint8(vtargetPatch))
            print("targetPatch_fR", self.targetPatch_fR)
            print("vsourcePatch", np.uint8(vsourcePatch))
            #print("fillMask",self.targetPatch_fRmsk)
        return vImg

    def fillandUpdate(self):
        ## Filltexture
        for i in range(3):
            self.textureImaged[:,:,i] = self.copyData(self.textureImaged[:,:,i],0)
        ## Filldepth
        self.depthImaged = self.copyData(self.depthImaged,0)
        ## Fill curvatureData
        self.curvatureData = self.copyData(self.curvatureData,0)
        # UpdateConfidence
        self.cofidenceImage[self.targetPatch_tl[0]:self.targetPatch_br[0], self.targetPatch_tl[1]:self.targetPatch_br[1]] = 1
        #self.cofidenceImage = self.copyData(self.cofidenceImage,0)
