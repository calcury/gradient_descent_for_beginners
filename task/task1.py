import numpy as np

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

train_y = [[1, 1, 0, 0, 1, 1, 0, 0, 1, 0]]

np.random.seed(1)

lr = 0.3
data = np.array(train_X)
res = np.array(train_y).T

input_layer = 5
hidden_layer_1 = 9
hidden_layer_2 = 7
output_layer = 1

w = [
    np.random.rand(input_layer, hidden_layer_1)-0.5,
    np.random.rand(hidden_layer_1, hidden_layer_2)-0.5,
    np.random.rand(hidden_layer_2, output_layer)-0.5,
]

b = [
    np.random.rand(1, hidden_layer_1)-0.5,
    np.random.rand(1, hidden_layer_2)-0.5,
    np.random.rand(1, output_layer)-0.5,
]


def fp(data, w, b):
    z = np.dot(data, w) + b
    a = 1/(1+np.exp(-z))
    return z, a


def bp(w, b, z, a, res):
    l = len(data)
    Loss = sum(sum((a[2] - res)**2))
    Loss_ = a[2] - res
    a2_ = Loss_*a[2]*(1-a[2])
    a1_ = None  # 1. Calculate the error term of the second hidden layer using chain rule
    a0_ = np.dot(a1_, w[1].T)*a[0]*(1-a[0])
    w[2] = None  # 2. Update the weight between the second hidden layer and output layer
    w[1] = w[1] - lr*np.dot(a[0].T, a1_)/l
    w[0] = None  # 3. Update the weight between the input layer and first hidden layer
    b[0] = None  # 4. Update the bias_0 term of the first hidden layer
    b[1] = b[1] - lr*np.sum(a1_, axis=0, keepdims=True)/l
    b[2] = None  # 5. Update the bias_2 term of the output layer
    return Loss


for _ in range(5000):
    z, a = [0]*3, [0]*3
    z[0], a[0] = fp(data, w[0], b[0])
    z[1], a[1] = fp(a[0], w[1], b[1])
    z[2], a[2] = fp(a[1], w[2], b[2])

    Loss = bp(w, b, z, a, res)
    if _ % 1000 == 0:
        print(f"Loss: {Loss:.4f}")
print("-"*30)
print(f"Final Loss: {Loss:.4f}")
