import numpy as np
import cv2
import sys

def cluster(inputImage, outputImage, clusters):
   img = cv2.imread(inputImage)
   # cv2.imshow('Input', img)
   Z = img.reshape((-1,3))

   # convert to np.float32
   Z = np.float32(Z)

   # define criteria, number of clusters(K) and apply kmeans()
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
   K = clusters
   ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

   # Now convert back into uint8, and make original image
   center = np.uint8(center)
   res = center[label.flatten()]
   res2 = res.reshape((img.shape))

   # cv2.imshow('Output',res2)
   cv2.imwrite(outputImage,res2)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

if __name__ == "__main__":
    if (len(sys.argv) == 4):
        inputImage = sys.argv[1]
        outputImage = sys.argv[2]
        clusters = int(sys.argv[3])
        cluster(inputImage,outputImage,clusters)
    else:
        print "Invalid Input arguments"
        print "python.exe ImageSegmentation inputImagePath outputImagePath numberOfClusters"