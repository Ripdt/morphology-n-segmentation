import numpy as np
import cv2
import sys
from random import randint as randi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bgr2hex(bgr):
    return "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def ScatterPlot(img, centroids, clusterLabels, plotNameOut="scatterPlot.png"):
    fig = plt.figure()
    ax = Axes3D(fig)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            ax.scatter(img[x, y, 2], img[x, y, 1], img[x, y, 0], color=bgr2hex(centroids[clusterLabels[x, y]]))
    plt.savefig(plotNameOut)
    plt.show()

def ShowCluster(img, centroids, clusterLabels, imgNameOut="kmeans.png"):
    result = np.zeros((img.shape), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bgr = centroids[clusterLabels[i, j]]
            result[i, j, 0] = np.uint8(bgr[0])
            result[i, j, 1] = np.uint8(bgr[1])
            result[i, j, 2] = np.uint8(bgr[2])
    cv2.imwrite(imgNameOut, result)
    ScatterPlot(img, centroids, clusterLabels, plotNameOut="scatterPlot.png")
    cv2.imshow("K-Mean Cluster", result)
    cv2.waitKey(0)

def GetEuclideanDistance(Cbgr, Ibgr):
    return np.sqrt(np.sum((Cbgr - Ibgr) ** 2))

def KMeans3D(img, k=2, max_iterations=100, tolerance=1.0, imgNameOut="kmeans.png"):
    Clusters = k
    centroids = np.zeros((k, 3), dtype=np.float64)
    
    # Initialize centroids
    for i in range(Clusters):
        x = randi(0, img.shape[0]-1)
        y = randi(0, img.shape[1]-1)
        centroids[i] = img[x, y]

    ClusterLabels = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    prev_centroids = np.copy(centroids)

    for i in range(max_iterations):
        # Assign each pixel to the nearest centroid
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                MinDist = sys.float_info.max
                for c in range(Clusters):
                    dist = GetEuclideanDistance(centroids[c], img[x, y])
                    if dist < MinDist:
                        MinDist = dist
                        ClusterLabels[x, y] = c

        # Update centroids
        for c in range(Clusters):
            points = img[ClusterLabels == c]
            if len(points) > 0:
                centroids[c] = np.mean(points, axis=0)

        # Check for convergence
        diff = np.sum(np.sqrt(np.sum((centroids - prev_centroids) ** 2, axis=1)))
        if diff <= tolerance:
            print(f"Converged after {i+1} iterations.")
            break
        prev_centroids = np.copy(centroids)

    ShowCluster(img, centroids, ClusterLabels, imgNameOut)

if __name__ == '__main__':
    Image = cv2.imread("res/unexistent-person-1.jpg") 

    Image = cv2.resize(Image, None, fx=0.25, fy=0.25)

    print("Image Size:", Image.shape)
    KMeans3D(Image, k=2, max_iterations=10, imgNameOut="out/img_kmeans.png")