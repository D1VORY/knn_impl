# k-nearest neighbors and Parzen window

There are two implementations of the k-nearest neighbors algorithm. It also contains Parzen window algorithm.


## Structure

### knn.py
It contains a simple implementation of knn classifier made with NumPy. It uses distance matrix and argsorted matrix to speed up computing. As a data source was chosen the susy dataset that contains 100k train examples and 1000 test samples.

### ML02.py
This file contains a naive implementation of knn with bare python code. It creates its own dataset of (x1,x2) points and finds the best number of nearest neighbors. It also has two functions that eliminate points close to boundaries of class.


### ML02B.py
Here lies the implementation of Parzen window along with different kernels. Dataset and settings are the same as in knn.
