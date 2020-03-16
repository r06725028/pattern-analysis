                                                                          1,1           All
from sklearn.cluster import KMeans
import numpy as np
from keras.datasets import mnist



def accu(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y = np.concatenate((y_train, y_test))
print(y.shape)
n_clusters = len(np.unique(y))

encoder_feature = np.load('encoder_feature.npy').reshape(y.shape[0],-1)
#print(encoder_feature.shape)

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder_feature)
print(y_pred.shape)
acc = np.round(accu(y, y_pred), 5)
print(acc)
~
