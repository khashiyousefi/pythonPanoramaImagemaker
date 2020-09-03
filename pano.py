import imutils

import cv2
import numpy as np
from matplotlib import pyplot as plt


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=.75, Thresh=4.0,
               showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images

        # imageA_ = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        # imageB_ = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, Thresh)
        # keypoints to create a panorama
        # otherwise, apply a perspective warp to stitch the images together
        (matches, H, status) = M

        # check to see if the keypoint matches should be visualized
        resultWrong = cv2.warpPerspective(imageA, H,
                                          (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        # res = cv2.fisheye.undistortPoints(imageA, pts)

        if showMatches:
            (copy, vis, ptA, ptB) = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                                     status)
        # getting concat image size

        copyH, copyW = copy.shape[:2]
        copyMid = int(copyW * .5)

        imageAx = copyMid - ptA[0]
        imageAxis = copyMid + imageAx

        imageBx = copyMid - ptB[0]
        imageBxis = copyMid + imageBx

        cutA = copy[0:copyH, 0:imageBxis]
        cutB = copy[0:copyH, imageAxis:copyW]

        final = np.concatenate(
            (cutA, cutB), axis=1)

        return (final, resultWrong, M, vis, ptA, ptB)
        # return vis Image, attchedImage, findHomography in M and one paired Matched point(x,y)

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        copy = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        copy[0:hA, 0:wA] = imageA
        copy[0:hB, wA:] = imageB
        # loop over the matches
        TotalptA = []
        TotalptB = []
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                TotalptA.append(ptA)
                TotalptB.append(ptB)
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return copy, vis, ptA, ptB

    def detectAndDescribe(self, image):

        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, Thresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

                # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, Thresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        # otherwise, no homograpy could be computed
        return None
