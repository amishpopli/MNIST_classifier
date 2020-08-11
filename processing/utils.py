from sklearn.datasets import fetch_openml
from scipy.ndimage import shift
import numpy as np

def getData():
    return fetch_openml('mnist_784', version=1)

def augment(data,target):
    aug_data = [_ for _ in data]
    aug_target = [_ for _ in target]

    for dx,dy in ((0,5),(0,-5),(5,0),(-5,0)):
        for x,y in zip(data,target):
            shifted_img = shift(x.reshape((28, 28)),[dx,dy],mode='constant', cval=0)
            aug_data.append(shifted_img.reshape(-1))
            aug_target.append(y)

    aug_data = np.array(aug_data)
    aug_target = np.array(aug_target)
    return aug_data, aug_target

def saveOutput(accuracy):
    file = open('output', 'w')

    for _ in accuracy.keys():
        file.write(f"{_}: {accuracy[_]}\n")

    file.close()








