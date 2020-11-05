import ML02 as fst
import matplotlib.pyplot as plt
import math
from operator import itemgetter

X1S = 1
X1F = 10

X2S = 1
X2F = 22

ROWS = 7
COLUMNS = 5

SIZE = 1000

def takeCloseElements(list, element):
    res = []
    it = iter(list)
    kek =  next(it)
    while(element >= kek):
        res.append(kek)
        kek = next(it)

    return res

def triangleKernel(r):
    r = abs(r)
    if r > 1 :
        return 0
    else:
        return 1 - r

def rectangleKernel(r):
    r = abs(r)
    return 0 if r > 1 else r

def quadraticKernel(r):
    r = abs(r)
    if r > 1:
        return 0
    else:
        return 1 - r**2

def biquadraticKernel(r):
    return (quadraticKernel(r))**2

def gaussianKernel(r):
    return math.pow(math.e , -2 * (r**2))


def parzenWindow(x_train, x_test , y_train, y_test , h , kernel):
    classes = {}
    result = []
    for testElement in x_test:
        for i , trainElement in enumerate(x_train):
            r = fst.distance(testElement,trainElement) / h
            k = kernel(r)
            if (y_train[i] in classes) :
                classes[y_train[i]] += k
            else:
                classes[y_train[i]] = k

        resClass = max(classes,key=classes.get)
        classes.clear()
        result.append(resClass)

    return result

def getMaxH( X1S,X1F,X2S,X2F ):
    a = X1F - X1S
    b = X2F - X1S
    return math.sqrt(a**2 + b**2) / 2


def parzen_training(x_train, x_test , y_train, y_test , X1S,X1F,X2S,X2F , stepK = 0.01 ):
    kernels = [rectangleKernel,triangleKernel,quadraticKernel,biquadraticKernel,gaussianKernel]
    kernelLabels = ['rectangle Kernel','triangle Kernel','quadraticKernel','biquadraticKernel','gaussianKernel']


    for i in range(len(kernels)):
        maxH = getMaxH( X1S,X1F,X2S,X2F )
        step = maxH * stepK
        h = step
        accuracy = []
        steps = []
        while(h < maxH ):
            parzenRes = parzenWindow(x_train, x_test , y_train, y_test,h ,kernels[i])
            accuracy.append(fst.calculateAccuracy(y_test , parzenRes))
            steps.append(h)
            h += step
        plt.figure(i)
        plt.title(f'Accuracy of {kernelLabels[i]}')
        plt.ylabel("Accuracy percent")
        plt.xlabel("width of window")
        plt.plot(steps, accuracy, 'green')
        opt = max(enumerate(accuracy), key=itemgetter(1))
        optimalH = steps[opt[0]]
        print(f'best accuracy of {kernelLabels[i]} = {opt[1]} in H = {optimalH}')
    return None




x,y = fst.generateData(ROWS,COLUMNS,X1S,X1F,X2S,X2F , SIZE)
x_train, x_test , y_train, y_test = fst.train_test_split(x,y,test_size=0.2)

optimalH = parzen_training(x_train, x_test , y_train, y_test , X1S,X1F,X2S,X2F , stepK = 0.01 )

fst.drawPlot(ROWS,COLUMNS,X1S,X1F,X2S,X2F , x_train, x_test)

