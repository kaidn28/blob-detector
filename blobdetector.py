# Standard imports
import cv2
import numpy as np

class BlobDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()

        #thresholds:
        params.minThreshold = 50
        params.maxThreshold = 200
        params.thresholdStep = 10
        
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.maxCircularity = 0.7

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0
        params.maxInertiaRatio = 1
        self.detector = cv2.SimpleBlobDetector_create(params)
    
    def detect(self, image):
        keypoints = self.detector.detect(image)
        for i in keypoints:
            print("x= {0}, y = {1}".format(i.pt[0], i.pt[1]))
            print("diameter={0}".format(i.size))
            print("angle={0}".format(i.angle))
        return keypoints
        




# Read image
im = cv2.imread("blob.jpg", 0)
print(im.shape)
cv2.imshow("///", im)
cv2.waitKey(0)

bd = BlobDetector()
keypoints = bd.detect(im)
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()