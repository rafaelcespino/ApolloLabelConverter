import numpy as np 
import os 
import scipy.io as sio

labels = "./SAD/train/FS02_train_001.txt"
outpath = "./FS02_train_001.mat"


#read each line of file into list
with open(labels) as f:
    lines = f.readlines()

#initialize 2d numpy array with zeros 
lastLine = lines[-1]
totalTime = lastLine.split()[3]
arraySize = float(totalTime) * 16000
arraySize = int(arraySize)
matLabels = np.zeros(arraySize)



#obtain each line with speech data 
for line in lines:
    label = line.split()
    if label[4] == "S":

        start = float(label[2]) * 16000
        start = int(start)
        
        end = float(label[3]) * 16000
        end = int(end)

        matLabels[start:end] = 1

#reshape array into a 2d array and save to a mat file
matLabels = matLabels.reshape((arraySize, 1))
print(matLabels)
sio.savemat(outpath, {"y_label":matLabels})
