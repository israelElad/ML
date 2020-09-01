# Elad Israel 313448888

import sys
import scipy.io.wavfile
import numpy as np


# compute distance=sqrt((x1-x2)^2 + (y1-y2)^2)
def distanceP1P2(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# find the closest centroid to a point(x and y) and returns its index
def findClosestCentroidToXY(x, y, centroidsArr):
    # initialize closest distance to be the first centroid
    closestDist = distanceP1P2(x, y, centroidsArr[0][0], centroidsArr[0][1])
    closestCentroidIndex = 0
    # search for a closer centroid
    for i in range(len(centroidsArr)):
        distFromCurrentCentroid = distanceP1P2(x, y, centroidsArr[i][0], centroidsArr[i][1])
        if (distFromCurrentCentroid < closestDist):
            closestDist = distFromCurrentCentroid
            closestCentroidIndex = i
    return closestCentroidIndex


# update each Centroid to the average location of all the points of the same color
def updateCentroids(xArr, yArr, centroids, colorsArray):
    for centroidIndex in range(len(centroids)):
        pointsCounter = 0
        pointsSumX = 0
        pointsSumY = 0
        # iterate all cells in the colors array(cell i represent point i's color)
        for pointIndex in range(len(colorsArray)):
            if (colorsArray[pointIndex] == centroidIndex):
                pointsSumX += xArr[pointIndex]
                pointsSumY += yArr[pointIndex]
                pointsCounter += 1
        # avoid devision by zero if there are no points
        if (pointsCounter == 0):
            break
        # update the current centroid to the average
        centroids[centroidIndex][0] = pointsSumX / pointsCounter
        centroids[centroidIndex][1] = pointsSumY / pointsCounter


# kMeans algorithm
def kMeansAlgorithm(centroids, dataArrCopy):
    # separate 2D data list to two 1D lists
    xArr, yArr = zip(*dataArrCopy)

    # step 1: initialize centroids
    centroids = np.loadtxt(centroids)
    if len(centroids) == 0:
        exit(-1)
    # keep previous centroids for comparison of convergence
    prevCentroids = np.array(centroids.copy())
    f = open("output.txt", "w")
    count = 0

    # step 2: repeat until convergence:
    while (count < 30):
        colorsArray = []

        # step 2.1: assign each point to the closest centroid
        for i in range(len(dataArrCopy)):
            colorsArray.append(findClosestCentroidToXY(xArr[i], yArr[i], centroids))

        # step 2.2: update each centroid to be the average of the points in its cluster
        updateCentroids(xArr, yArr, centroids, colorsArray)
        centroids = centroids.round()

        # writing to file after each centroids update
        if np.array_equal(prevCentroids, centroids):
            f.write(f"[iter {count}]:{','.join([str(i) for i in centroids])}\n")
            break
        prevCentroids = np.array(centroids.copy())
        f.write(f"[iter {count}]:{','.join([str(i) for i in centroids])}\n")
        count += 1
    f.close()

    # update each point in the wav file to be its centroid
    for i in range(len(dataArrCopy)):
        dataArrCopy[i][0] = centroids[colorsArray[i]][0]
        dataArrCopy[i][1] = centroids[colorsArray[i]][1]


def main():
    # get sample file and centroid as arguments
    if len(sys.argv)<3:
        exit(-1)
    sample = sys.argv[1]
    centroids = sys.argv[2]
    fs, dataArr = scipy.io.wavfile.read(sample)
    dataArrCopy = np.array(dataArr.copy())

    kMeansAlgorithm(centroids, dataArrCopy)

    # save the compressed result to the wav file
    scipy.io.wavfile.write("compressed.wav", fs, np.array(dataArrCopy, dtype=np.int16))


main()
