import math
import csv

def grnn(I, DATA, SIGMA, SKIP):
    DIST = []
    F = []
    for R in range(len(DATA)):
        D2 = 0
        for K in range(len(I)):
            D2 += (I[K] - DATA[R][K]) ** 2
        DIST.append(D2)
        if R == SKIP:
            F.append(0.0)
        else:
            F.append(math.exp(-D2 / (2 * SIGMA ** 2)))
    TOP = 0
    BOTTOM = 0
    for R in range(len(DATA)):
        TOP += F[R] * DATA[R][-1]
        BOTTOM += F[R]
    return DIST, F, TOP, BOTTOM, TOP / BOTTOM

def main():
    FILE = input("Enter CSV file path: ").strip().strip('"')
    HEADER = input("Does CSV have a header? (yes/no): ").lower()

    ROWS = list(csv.reader(open(FILE, encoding="utf-8-sig")))
    if HEADER == "yes":
        ROWS = ROWS[1:]

    DATA = []
    for ROW in ROWS:
        VALUES = []
        for X in ROW:
            VALUES.append(float(X))
        DATA.append(VALUES)

    N = len(DATA[0]) - 1
    print("Training entries:", len(DATA))
    print("Number of inputs:", N)

    SIGMA = float(input("Enter smoothing factor (sigma): "))

    print("\n===== TRAINING =====")
    print("No weights to train. Each of the", len(DATA), "rows is stored as one pattern neuron.")

    TOTAL = 0
    for R in range(len(DATA)):
        Y = grnn(DATA[R][:-1], DATA, SIGMA, R)[4]
        TOTAL += (DATA[R][-1] - Y) ** 2
    print("Final MSE (each row predicted without itself):", round(TOTAL / len(DATA), 4))

    print("\n===== TESTING =====")
    I = []
    for J in range(N):
        I.append(float(input(f"Enter I{J+1}: ")))

    DIST, F, TOP, BOTTOM, Y = grnn(I, DATA, SIGMA, -1)

    for R in range(len(DATA)):
        print(f"Pattern {R+1}: distance^2 =", round(DIST[R], 4), " F =", round(F[R], 4))

    print("\nSummation: top =", round(TOP, 4), " bottom =", round(BOTTOM, 4))
    print("Final Output:", round(Y, 4))
    if Y >= 0.5:
        print("Prediction: 1")
    else:
        print("Prediction: 0")

main()