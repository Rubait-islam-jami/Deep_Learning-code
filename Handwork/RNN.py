import math
import csv

def sigmoid(V):
    return 1 / (1 + math.exp(-V))

def forward(SEQ, W1, W1H, W2O, B1, B2):
    HN = len(W1)
    F = [[0.0] * HN]
    V = []
    for X in SEQ:
        VT = []
        FT = []
        for J in range(HN):
            S = B1[J]
            for K in range(len(X)):     # input- hidden calculation
                S += W1[J][K] * X[K]
            for K in range(HN):         # hidden- hidden calculation
                S += W1H[J][K] * F[-1][K]
            VT.append(S)
            FT.append(sigmoid(S))
        V.append(VT)
        F.append(FT)
    VO = B2                 # output calculation
    for K in range(HN):
        VO += W2O[K] * F[-1][K]
    return F, V, VO, sigmoid(VO)

def main():
    FILE = input("Enter CSV file path: ").strip().strip('"')
    HEADER = input("Does CSV have a header? (yes/no): ").lower()
    N = int(input("Enter inputs per time step: "))
    T = int(input("Enter number of time steps: "))

    ROWS = list(csv.reader(open(FILE, encoding="utf-8-sig")))
    if HEADER == "yes":
        ROWS = ROWS[1:]

    DATA = []
    for ROW in ROWS:
        VALUES = []
        for X in ROW:
            VALUES.append(float(X))
        SEQ = []
        for S in range(T):          # create sequence of inputs for each time step
            SEQ.append(VALUES[S * N:(S + 1) * N])
        DATA.append((SEQ, VALUES[-1]))

    print("Training entries:", len(DATA))

    HN = int(input("Enter number of hidden neurons: "))
    LR = float(input("Enter learning rate: "))
    ALPHA = float(input("Enter momentum coefficient: "))
    USE_BIAS = input("Use bias? (yes/no): ").lower() == "yes"

    W1, W1H, W2O = [], [], []
    for J in range(HN):
        ROW = []
        for K in range(N):
            ROW.append(float(input(f"W1 H{J+1} X{K+1}: ")))
        W1.append(ROW)
    for J in range(HN):
        ROW = []
        for K in range(HN):
            ROW.append(float(input(f"W1H H{J+1} H{K+1}: ")))
        W1H.append(ROW)
    for J in range(HN):
        W2O.append(float(input(f"W2O H{J+1}: ")))

    B1 = [0.0] * HN
    B2 = 0.0
    if USE_BIAS:
        for J in range(HN):
            B1[J] = float(input(f"B1 H{J+1}: "))
        B2 = float(input("B2: "))

    DW1 = []
    DW1H = []
    for J in range(HN):
        DW1.append([0.0] * N)
        DW1H.append([0.0] * HN)
    DW2O = [0.0] * HN
    DB1 = [0.0] * HN
    DB2 = 0.0

    print("\n===== TRAINING =====")
    for R in range(len(DATA)):
        SEQ, D = DATA[R]
        F, V, VO, Y = forward(SEQ, W1, W1H, W2O, B1, B2)

        DO = (D - Y) * Y * (1 - Y)
        DELTA = []
        for S in range(T):
            DELTA.append([0.0] * HN)
        for S in range(T - 1, -1, -1):
            for J in range(HN):
                if S == T - 1:          # backpropagation of last time step
                    E = W2O[J] * DO
                else:
                    E = 0
                    for K in range(HN):
                        E += W1H[K][J] * DELTA[S + 1][K]
                DELTA[S][J] = F[S + 1][J] * (1 - F[S + 1][J]) * E

        for J in range(HN):
            CHANGE = LR * DO * F[T][J] + ALPHA * DW2O[J]
            W2O[J] += CHANGE
            DW2O[J] = CHANGE

            for K in range(N):
                G = 0
                for S in range(T):
                    G += DELTA[S][J] * SEQ[S][K]
                CHANGE = LR * G + ALPHA * DW1[J][K]
                W1[J][K] += CHANGE
                DW1[J][K] = CHANGE

            for K in range(HN):
                G = 0
                for S in range(T):
                    G += DELTA[S][J] * F[S][K]
                CHANGE = LR * G + ALPHA * DW1H[J][K]
                W1H[J][K] += CHANGE
                DW1H[J][K] = CHANGE

            if USE_BIAS:
                G = 0
                for S in range(T):
                    G += DELTA[S][J]
                CHANGE = LR * G + ALPHA * DB1[J]
                B1[J] += CHANGE
                DB1[J] = CHANGE

        if USE_BIAS:
            CHANGE = LR * DO + ALPHA * DB2
            B2 += CHANGE
            DB2 = CHANGE

        print(f"Row {R+1}/{len(DATA)} done")

    TOTAL = 0
    for SEQ, D in DATA:
        F, V, VO, Y = forward(SEQ, W1, W1H, W2O, B1, B2)
        TOTAL += (D - Y) ** 2
    print("\nFinal MSE:", round(TOTAL / len(DATA), 4))

    print("\n===== FINAL WEIGHTS =====")
    print("W1:", W1)
    print("W1H:", W1H)
    print("W2O:", W2O)
    print("B1:", B1)
    print("B2:", B2)

    print("\n===== TESTING =====")
    SEQ = []
    for S in range(T):
        print(f"Time step {S+1}")
        X = []
        for K in range(N):
            X.append(float(input(f"Enter X{K+1}: ")))
        SEQ.append(X)

    F, V, VO, Y = forward(SEQ, W1, W1H, W2O, B1, B2)
    for S in range(T):
        print(f"\nTime step {S+1}")
        for J in range(HN):
            print(f"H{J+1}: V =", round(V[S][J], 4), " F =", round(F[S + 1][J], 4))

    print("\nOutput: V =", round(VO, 4), " F =", round(Y, 4))
    if Y >= 0.5:
        print("Prediction: 1")
    else:
        print("Prediction: 0")

main()