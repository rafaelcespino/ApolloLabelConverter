import numpy as np 
import os 
import scipy.io as sio

inputDir = "./SAD/dev/"

for labelFile in os.listdir(inputDir):
    with open("./SAD/dev/{}".format(labelFile)) as f:
        lines = f.readlines()
        base = os.path.basename(labelFile)
        base = os.path.splitext(base)[0]
    print(base)

    lastLine = lines[-1]
    totalTime = lastLine.split()[3]
    arraySize = float(totalTime) * 16000
    arraySize = int(arraySize)
    matLabels = np.zeros(arraySize)

    for line in lines:
        label = line.split()
        if label[4] == "S":

            start = float(label[2]) * 16000
            start = int(start)
        
            end = float(label[3]) * 16000
            end = int(end)

    matLabels[start:end] = 1
    matLabels = matLabels.reshape((arraySize, 1))
    sio.savemat("./convertedLabels/{}.mat".format(base), {"y_label":matLabels})
        