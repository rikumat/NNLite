from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
import numpy as np
from NNLite.loss.BCE.BCE import BCE
from NNLite.activation.relu.relu import Relu
from NNLite.activation.logistic.logistic import Logistic
from NNLite.network.sequential.sequential import Sequential
from NNLite.layer.linear.linear import Linear
from NNLite.optimizer.SGD.SGD import SGD

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_mask = (y_train == 5) | (y_train == 6)
x_train_filtered = x_train[train_mask]
y_train_filtered = y_train[train_mask]

test_mask = (y_test == 5) | (y_test == 6)
x_test_filtered = x_test[test_mask]
y_test_filtered = y_test[test_mask]

x_train_flat = x_train_filtered.reshape(x_train_filtered.shape[0], -1)  # flatten to (num_samples, 784)
x_test_flat = x_test_filtered.reshape(x_test_filtered.shape[0], -1)

x_train_flat = x_train_flat.astype('float32') / 255.0
x_test_flat  = x_test_flat.astype('float32') / 255.0

batch_size = 100
num_samples = x_train_filtered.shape[0]

model = Sequential(
    Linear(784, 128),
    Relu(alpha=0.01),
    Linear(128, 64),
    Relu(alpha=0.01),
    Linear(64, 1),
    Logistic()
)
loss = BCE()

optim = SGD(model.params(), lr=0.0001)
NUM_EPOCH=100

for j in range(0, NUM_EPOCH):
    indices = np.random.permutation(num_samples)
    x_train_flat = x_train_flat[indices]
    y_train_filtered = y_train_filtered[indices]
    
    acc = 0
    for i in range(0, num_samples, batch_size):

        x_batch = x_train_flat[i:i+batch_size]
        y_batch = y_train_filtered[i:i+batch_size]

        y_train_binary = np.where(y_batch == 5, 1, 0).reshape(-1, 1)

        result = model.forward(x_batch)
        acc += accuracy_score(y_train_binary, (result>0.5).astype(int))

        loss.forward(result, y_train_binary)
        model.backward(loss)
        optim.step()

    print(acc/(num_samples/batch_size))























