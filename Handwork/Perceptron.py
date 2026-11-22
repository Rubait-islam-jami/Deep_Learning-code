
import math

def sigmoid(v):
    return 1 / (1 + math.exp(-v))


def main():

    n = int(input("Enter number of inputs: "))
    rows = int(input("Enter number of training rows: "))
    lr = float(input("Enter learning rate: "))

    if n <= 0 or rows <= 0 or lr <= 0:
        print("Error: Inputs, rows and learning rate must be positive.")
        return

    weights = []

    for j in range(n):
        w = float(input(f"Enter initial weight w{j+1}: "))
        weights.append(w)

    bias = float(input("Enter initial bias: "))

    training_data = []

    print("\n--- Enter Training Data ---")

    for i in range(rows):
        inputs = []

        for j in range(n):
            value = float(input(f"Row {i+1} - I{j+1}: "))
            inputs.append(value)

        d = float(input(f"Row {i+1} - d: "))

        if d not in [0, 1]:
            print("Error: d must be 0 or 1.")
            return

        training_data.append((inputs, d))

    print("\n========== TRAINING ==========")

    for i in range(rows):

        inputs, d = training_data[i]

        print(f"\n========== ROW {i+1} ==========")

        print("Inputs:", inputs)
        print("d:", d)
        print("Old Weights:", [round(w, 4) for w in weights])
        print("Old Bias:", round(bias, 4))

        v = bias

        for j in range(n):
            v += inputs[j] * weights[j]

        f = sigmoid(v)

        error = d - f

        delta = error * f * (1 - f)

        print("\nNet Input (v):", round(v, 4))
        print("Sigmoid Output (f):", round(f, 4))
        print("Error:", round(error, 4))
        print("Delta:", round(delta, 4))

        for j in range(n):
            old_weight = weights[j]
            weights[j] += lr * delta * inputs[j]

            print(f"Updated w{j+1}: {old_weight:.4f} -> {weights[j]:.4f}")

        old_bias = bias
        bias += lr * delta

        print(f"Updated Bias: {old_bias:.4f} -> {bias:.4f}")

    print("\n========== TRAINING COMPLETED ==========")
    print("Final Weights:", [round(w, 4) for w in weights])
    print("Final Bias:", round(bias, 4))

    print("\n========== TESTING ==========")

    test_inputs = []

    for j in range(n):
        value = float(input(f"Enter test i{j+1}: "))
        test_inputs.append(value)

    v = bias

    for j in range(n):
        v += test_inputs[j] * weights[j]

    f = sigmoid(v)

    prediction = 1 if f >= 0.5 else 0

    print("\nTest Inputs:", test_inputs)
    print("Net Input (v):", round(v, 4))
    print("Sigmoid Output (f):", round(f, 4))
    print("Final Prediction:", prediction)

main()
