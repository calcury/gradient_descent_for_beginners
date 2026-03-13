lr = 0.1
x = 100
for _ in range(100):
    Loss = (x ** 2) / 2  # Loss function
    Gradient = x  # Gradient
    x = x - lr * Gradient  # Update param
    if _ % 10//9:
        print(f'Epochs:{_+1}; x:{x:.2f}; Loss:{Loss:.2f}')
