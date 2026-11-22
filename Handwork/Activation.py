import math

def threshold(v):
    if v>=0:
        return 1
    else:
        return 0

def piecewise_linear(v):
    if v>=0.5:
        return 1
    elif v>-0.5:
        return v+0.5
    else:
        return 0

def sigmoid(v):
    return 1/(1+math.exp(-v))

def signum(v):
    return math.tanh(v)

def relu(v):
    return max(0, v)


def main():

    n = int(input("Enter number of inputs: "))

    inputs = []
    weights = []

    if n <= 0:
        print("Error: Number of inputs must be positive.")
        return
    
    for i in range(n):
        value = float(input(f"Enter I{i+1}: "))
        inputs.append(value)

    for i in range(n):
        value = float(input(f"Enter W{i+1}: "))
        weights.append(value)

    if len(inputs)!=len(weights):
        print("Error: Every input must have a weight!")
        return

    v = 0
    for i in range(n):
        v += inputs[i] * weights[i]

    print("\nNet Input (v):", v)
    print("Threshold:", threshold(v))
    print("Piecewise Linear:", piecewise_linear(v))
    print("Sigmoid:", round(sigmoid(v), 4))
    print("Signum:", round(signum(v), 4))
    print("ReLU:", relu(v))

main()
