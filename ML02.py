import math
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from operator import itemgetter

X1S = 1
X1F = 10

X2S = 1
X2F = 10

ROWS = 6
COLUMNS = 6

SIZE = 1000


def drawLines(rows, columns, xStart , xFinish , yStart, yFinish):
  width = (xFinish - xStart ) / columns
  height = (yFinish - yStart ) / rows

  temp = xStart + width
  for i in range(1,columns):
    plt.plot([temp,temp],[yStart,yFinish],'black')
    temp += width

  temp = yStart + height
  for i in range(1,rows):
    plt.plot([xStart,xFinish],[temp,temp],'black')
    temp += height

def drawPlot(rows, columns, xStart , xFinish , yStart, yFinish, x_train, x_test):
  plt.figure()
  plt.title("Distirbution")
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.autoscale(tight=True)

  drawLines(rows, columns, xStart , xFinish , yStart, yFinish)

  for point in x_train:
    plt.scatter(point[0], point[1], c='blue')

  for point in x_test:
    plt.scatter(point[0], point[1], c='red')






def assignClasses(data, rows, columns, xStart , xFinish , yStart, yFinish ):
  width = (xFinish - xStart) / columns
  height = (yFinish - yStart) / rows

  a = [xStart + width * (i -1 )   for i in range(1, columns+2)  ]
  b = [yStart + height * (i -1 )   for i in range(1, rows+2)  ]

  res = []
  # (x , y)
  for element in data:
    x1 = element[0]
    xres = 0
    for index ,val in enumerate(a):
      if x1 < val:
        xres = index
        break;

    x2 = element[1]
    xres2 = 0
    for index, val in enumerate(b):
      if x2 < val:
        xres2 = index
        break;

    res.append(f'({xres} , {xres2})')
  return res

def generateData(rows,columns, xStart , xFinish , yStart, yFinish , n  ):
  x = [ (random.uniform(xStart,xFinish),random.uniform(yStart,yFinish)) for i in range(n)]
  y = assignClasses(x,rows,columns,xStart , xFinish , yStart, yFinish)
  return x, y


def distance(a,b):
  return math.sqrt( float((a[0] - b[0])**2 +  (a[1] - b[1])**2) )

def most_frequent(List):
  dict = {}
  count, itm = 0, ''
  for item in reversed(List):
    dict[item] = dict.get(item, 0) + 1
    if dict[item] >= count:
      count, itm = dict[item], item
  return (itm)



def calculateAccuracy(actualData , predictedData):
  counter = 0
  for i in range(len(actualData)):
    if actualData[i] == predictedData[i]:
      counter += 1

  return counter / len(actualData)


def getDistance(point, x_train):
  distances = []
  for index, trainElement in enumerate(x_train):
    distances.append((distance(point, trainElement), index))

  return sorted(distances)


def classifyKNN(x_train, x_test , y_train, y_test , k ):
  result = []
  for testElement in x_test:
    resultedList = getDistance(testElement,x_train)[:k]

    labeledList =  [y_train[i[1]]  for i in resultedList ]
    res = most_frequent(labeledList)
    result.append(res)
    #print( f'{testElement}   =   {res}' )

  return  result



def knn(x_train, x_test , y_train, y_test , minK =1, maxK = 50, color = None ):
  if minK < 1 or maxK > len(x_train):
    raise IndexError('WRONG VALUE OF INDEX')
    return
  accuracy = [calculateAccuracy(y_test,classifyKNN(x_train, x_test, y_train, y_test, i))    for i in range(minK, maxK) ]
  plt.figure(2)
  plt.title("Accuracy")
  plt.ylabel("Accuracy percent")
  plt.xlabel("number of K")
  plt.plot(range(minK,maxK), accuracy,  color)

  optimalK = max(enumerate(accuracy), key=itemgetter(1))[0]
  print(optimalK)
  return optimalK

def improveDatasetA(x_train, y_train, percent = 0.2, threshold = 10):
  newX_train = []
  newY_train = []
  for i, point in enumerate(x_train):
    arrCopy = x_train[:]
    arrCopy.remove(point)
    okil = getDistance(point,arrCopy)[:threshold]
    labeledList = [y_train[j[1]] for j in okil]

    ownLabelsPercentage = labeledList.count(y_train[i]) / len(labeledList)
    foreignLabelsPercentage = 1 - ownLabelsPercentage
    if foreignLabelsPercentage < percent:
      newX_train.append(point)
      newY_train.append(y_train[i])

  return newX_train, newY_train

def improveDatasetB(x_train, y_train, percent = 0.8):
  labelDict = {}
  for i in range(len(y_train)):
    if y_train[i] in labelDict.keys():
      labelDict[y_train[i]].append(x_train[i])
    else:
      labelDict[y_train[i]] = [x_train[i]]

  newX_train = []
  newY_train = []

  for label, arr in  zip(labelDict.keys(), labelDict.values()):
    distances = []
    dictDistances = {}
    for point in arr:
      copyArr = arr[:]
      copyArr.remove(point)
      sum = 0
      for bPoint in copyArr:
        sum += distance(point, bPoint)
      distances.append(sum)
      dictDistances[sum] = point

    distances.sort()
    slicedDistances = distances[:int(len(distances) * percent)]
    for el in slicedDistances:
      newX_train.append(dictDistances[el])
      newY_train.append(label)



  return newX_train, newY_train


if __name__ == "__main__":
  x,y = generateData(ROWS,COLUMNS,X1S,X1F,X2S,X2F , SIZE)
  x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2)

  a , b  = improveDatasetB(x_train, y_train,0.8)
  print(len(a))

  optimalK = knn(x_train,x_test,y_train,y_test, 1, 50, 'green')
  optimalK2 = knn(a, x_test, b, y_test, 1, 50, 'red')

  drawPlot(ROWS, COLUMNS, X1S, X1F, X2S, X2F, a, x_test)
  drawPlot(ROWS,COLUMNS,X1S,X1F,X2S,X2F , x_train, x_test)

  plt.show()
  print("sss")

# відсоток найеталонніших
# найеталонніші - беремо відстань від кожного елемнут кожного класу і сумуємо відстань до всіх решти
# елемнтів цього класу з навчальної вибірки.  І вибиаємо ті елементи, у яких суми будуть найменші