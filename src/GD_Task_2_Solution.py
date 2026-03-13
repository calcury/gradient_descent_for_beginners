import numpy as np

# 训练数据
train_X = [[1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0],
           [1, 1, 0, 1, 0],
           [0, 0, 1, 0, 1],
           [1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1]]

train_y = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [1, 0, 0]]

np.random.seed(3)

lr = 0.3

data = np.array(train_X)
res = np.array(train_y)

input_layer = 5
hidden_layer_1 = 9
hidden_layer_2 = 7
output_layer = 3

w = [
    np.random.rand(input_layer, hidden_layer_1) - 0.5,
    np.random.rand(hidden_layer_1, hidden_layer_2) - 0.5,
    np.random.rand(hidden_layer_2, output_layer) - 0.5,
]

b = [
    np.random.rand(1, hidden_layer_1) - 0.5,
    np.random.rand(1, hidden_layer_2) - 0.5,
    np.random.rand(1, output_layer) - 0.5,
]


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def fp(data, w, b, activation):
    z = np.dot(data, w) + b
    if activation == 'tanh':
        a = tanh(z)
    elif activation == 'relu':
        a = relu(z)
    elif activation == 'softmax':
        a = softmax(z)
    else:
        a = z
    return z, a


def bp(w, b, z, a, res):
    l = len(data)
    Loss = -np.sum(res * np.log(a[2] + 1e-8)) / l
    d_2 = a[2] - res
    relu_d = np.where(z[1] > 0, 1, 0)
    d_1 = np.dot(d_2, w[2].T) * relu_d
    tanh_d = 1 - np.square(a[0])
    d_0 = np.dot(d_1, w[1].T) * tanh_d
    w[2] = w[2] - lr * np.dot(a[1].T, d_2) / l
    w[1] = w[1] - lr * np.dot(a[0].T, d_1) / l
    w[0] = w[0] - lr * np.dot(data.T, d_0) / l
    b[2] = b[2] - lr * np.sum(d_2, axis=0, keepdims=True) / l
    b[1] = b[1] - lr * np.sum(d_1, axis=0, keepdims=True) / l
    b[0] = b[0] - lr * np.sum(d_0, axis=0, keepdims=True) / l
    return Loss


for epoch in range(5000):
    z, a = [0]*3, [0]*3
    z[0], a[0] = fp(data, w[0], b[0], 'tanh')
    z[1], a[1] = fp(a[0], w[1], b[1], 'relu')
    z[2], a[2] = fp(a[1], w[2], b[2], 'softmax')

    Loss = bp(w, b, z, a, res)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {Loss:.4f}")
print("-"*30)
print(f"Final Loss: {Loss:.4f}")
