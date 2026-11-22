import math
import csv

def sigmoid(V):
    return 1 / (1 + math.exp(-V))

def forward(I, W, B):
    F = [I]
    V = []
    # Layer loop
    for L in range(len(W)):
        VL = []
        FL = []
        # Neuron loop
        for J in range(len(W[L])):
            S = B[L][J]
            # Input loop
            for K in range(len(F[L])):
                S += F[L][K] * W[L][J][K]
            VL.append(S)
            FL.append(sigmoid(S))
        V.append(VL)
        F.append(FL)
    return F, V

def main():
    FILE = input("Enter CSV file path: ").strip().strip('"')
    HEADER = input("Does CSV have a header? (yes/no): ").lower()

    ROWS = list(csv.reader(open(FILE, encoding="utf-8-sig")))
    if HEADER == "yes":
        ROWS = ROWS[1:]
    # CSV data to float
    DATA = []
    for ROW in ROWS:
        VALUES = []
        for X in ROW:
            VALUES.append(float(X))
        DATA.append(VALUES)

    N = len(DATA[0]) - 1 #no. of inputs = total columns - output column

    print("Training entries:", len(DATA))
    print("Number of inputs:", N)

    H = int(input("Enter number of hidden layers: "))
    SIZES = [N]
    for L in range(H):
        SIZES.append(int(input(f"Enter neurons in H{L+1}: "))) # neurons in hidden layer
    SIZES.append(1) # neuron in output layer

    LR = float(input("Enter learning rate: "))
    ALPHA = float(input("Enter momentum coefficient: "))
    USE_BIAS = input("Use bias? (yes/no): ").lower() == "yes"

    W, DW, B, DB = [], [], [], []
    #goes through each layer and asks for temporary W, DW, B, DB
    for L in range(1, len(SIZES)):
        NAME = f"H{L}" if L <= H else "O"
        WL, DWL, BL, DBL = [], [], [], []
        # goes through each neuron in the layer and asks incoming W and DW
        for J in range(SIZES[L]):
            WJ, DWJ = [], []
            # goes through each neuron from the previous layer and asks for weights and deltas
            for K in range(SIZES[L - 1]):
                WJ.append(float(input(f"W {NAME} N{J+1} input {K+1}: ")))
                DWJ.append(float(input(f"DW {NAME} N{J+1} input {K+1}: ")))
            WL.append(WJ)
            DWL.append(DWJ)
            if USE_BIAS:
                BL.append(float(input(f"B {NAME} N{J+1}: ")))
                DBL.append(float(input(f"DB {NAME} N{J+1}: ")))
            else:
                BL.append(0.0)
                DBL.append(0.0)
        W.append(WL)
        DW.append(DWL)
        B.append(BL)
        DB.append(DBL)

    print("\n===== TRAINING =====")
    for R in range(len(DATA)):
        I = DATA[R][:-1]    # inputs
        D = DATA[R][-1]     # desired output
        F, V = forward(I, W, B)

        DELTA = []
        for L in range(len(W)):
            DELTA.append([])

        for J in range(SIZES[-1]):              # backpropagation of output layer
            DELTA[-1].append((D - F[-1][J]) * F[-1][J] * (1 - F[-1][J]))

        for L in range(len(W) - 2, -1, -1):     # backpropagation of hidden layers
            for J in range(SIZES[L + 1]):       # for each neuron in the current layer
                E = 0
                for K in range(SIZES[L + 2]):   # for each neuron in the next layer
                    E += DELTA[L + 1][K] * W[L + 1][K][J]
                DELTA[L].append(F[L + 1][J] * (1 - F[L + 1][J]) * E)  

        for L in range(len(W)):
            for J in range(SIZES[L + 1]):
                for K in range(SIZES[L]):
                    CHANGE = LR * DELTA[L][J] * F[L][K] + ALPHA * DW[L][J][K]
                    W[L][J][K] += CHANGE
                    DW[L][J][K] = CHANGE
                if USE_BIAS:
                    CHANGE = LR * DELTA[L][J] + ALPHA * DB[L][J]
                    B[L][J] += CHANGE
                    DB[L][J] = CHANGE

        print(f"Row {R+1}/{len(DATA)} done")

    TOTAL = 0
    for ROW in DATA:
        F, V = forward(ROW[:-1], W, B)
        TOTAL += (ROW[-1] - F[-1][0]) ** 2      # error = desired - output
    print("\nFinal MSE:", round(TOTAL / len(DATA), 4))

    print("\n===== FINAL WEIGHTS =====")
    for L in range(len(W)):
        NAME = f"H{L+1}" if L < H else "O"
        print(NAME, "W:", W[L])
        print(NAME, "B:", B[L])

    print("\n===== TESTING =====")
    I = []
    for J in range(N):
        I.append(float(input(f"Enter I{J+1}: ")))

    F, V = forward(I, W, B)
    for L in range(len(W)):
        NAME = f"H{L+1}" if L < H else "O"
        for J in range(SIZES[L + 1]):
            print(f"{NAME} N{J+1}: V =", round(V[L][J], 4), " F =", round(F[L + 1][J], 4))

    print("Final Output:", round(F[-1][0], 4))
    if F[-1][0] >= 0.5:
        print("Prediction: 1")
    else:
        print("Prediction: 0")

main()