import math
import csv

def sigmoid(V):
    return 1 / (1 + math.exp(-V))

def forward(SEQ, W, U, B, WO, BO):
    HN = len(WO)
    C = [[0.0] * HN]        # cell state
    H = [[0.0] * HN]        # hidden state
    A = []                  # activation values for gates
    for X in SEQ:
        AT = [[], [], [], []]       # forget, input, candidate, output gates
        for G in range(4):
            # V = B[G] + W[G] * X + U[G] * H[-1]
            for J in range(HN):
                V = B[G][J]
                for K in range(len(X)):
                    V += W[G][J][K] * X[K]
                for K in range(HN):
                    V += U[G][J][K] * H[-1][K]
                if G == 2:
                    AT[G].append(math.tanh(V))  # candidate gate uses tanh
                else:
                    AT[G].append(sigmoid(V))    # other gates use sigmoid
        CT = []
        HT = []
        for J in range(HN):
            CT.append(AT[0][J] * C[-1][J] + AT[1][J] * AT[2][J])    # New cell = forget * old cell + input * candidate
            HT.append(AT[3][J] * math.tanh(CT[J]))                  # New hidden = output gate * tanh(new cell)
        A.append(AT)
        C.append(CT)
        H.append(HT)
    VO = BO
    for K in range(HN):
        VO += WO[K] * H[-1][K]          # Final output = bias + sum(weight * hidden state)
    return A, C, H, VO, sigmoid(VO)

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
        for S in range(T):
            SEQ.append(VALUES[S * N:(S + 1) * N])
        DATA.append((SEQ, VALUES[-1]))

    print("Training entries:", len(DATA))

    HN = int(input("Enter number of hidden neurons: "))
    LR = float(input("Enter learning rate: "))
    ALPHA = float(input("Enter momentum coefficient: "))
    USE_BIAS = input("Use bias? (yes/no): ").lower() == "yes"

    NAMES = ["Forget", "Input", "Candidate", "Output"]
    W, U, B = [], [], []
    for G in range(4):
        WG, UG, BG = [], [], []
        for J in range(HN):
            WJ = []
            for K in range(N):
                WJ.append(float(input(f"W {NAMES[G]} H{J+1} X{K+1}: ")))
            UJ = []
            for K in range(HN):
                UJ.append(float(input(f"U {NAMES[G]} H{J+1} H{K+1}: ")))
            WG.append(WJ)
            UG.append(UJ)
            if USE_BIAS:
                BG.append(float(input(f"B {NAMES[G]} H{J+1}: ")))
            else:
                BG.append(0.0)
        W.append(WG)
        U.append(UG)
        B.append(BG)

    WO = []
    for J in range(HN):
        WO.append(float(input(f"WO H{J+1}: ")))
    BO = 0.0
    if USE_BIAS:
        BO = float(input("BO: "))

    DW, DU, DB = [], [], []
    for G in range(4):
        DWG, DUG = [], []
        for J in range(HN):
            DWG.append([0.0] * N)
            DUG.append([0.0] * HN)
        DW.append(DWG)
        DU.append(DUG)
        DB.append([0.0] * HN)
    DWO = [0.0] * HN
    DBO = 0.0

    print("\n===== TRAINING =====")
    for R in range(len(DATA)):
        SEQ, D = DATA[R]
        A, C, H, VO, Y = forward(SEQ, W, U, B, WO, BO)

        DO = (D - Y) * Y * (1 - Y)
        DV = []
        DC = []
        for S in range(T):
            DV.append([[0.0] * HN, [0.0] * HN, [0.0] * HN, [0.0] * HN])
            DC.append([0.0] * HN)

        for S in range(T - 1, -1, -1):
            for J in range(HN):
                DH = 0
                if S == T - 1:
                    DH = WO[J] * DO
                else:
                    for G in range(4):
                        for K in range(HN):
                            DH += DV[S + 1][G][K] * U[G][K][J]
                TC = math.tanh(C[S + 1][J])
                DC[S][J] = DH * A[S][3][J] * (1 - TC ** 2)
                if S < T - 1:
                    DC[S][J] += DC[S + 1][J] * A[S + 1][0][J]
                DV[S][0][J] = DC[S][J] * C[S][J] * A[S][0][J] * (1 - A[S][0][J])
                DV[S][1][J] = DC[S][J] * A[S][2][J] * A[S][1][J] * (1 - A[S][1][J])
                DV[S][2][J] = DC[S][J] * A[S][1][J] * (1 - A[S][2][J] ** 2)
                DV[S][3][J] = DH * TC * A[S][3][J] * (1 - A[S][3][J])

        for G in range(4):
            for J in range(HN):
                for K in range(N):
                    GRAD = 0
                    for S in range(T):
                        GRAD += DV[S][G][J] * SEQ[S][K]
                    CHANGE = LR * GRAD + ALPHA * DW[G][J][K]
                    W[G][J][K] += CHANGE
                    DW[G][J][K] = CHANGE
                for K in range(HN):
                    GRAD = 0
                    for S in range(T):
                        GRAD += DV[S][G][J] * H[S][K]
                    CHANGE = LR * GRAD + ALPHA * DU[G][J][K]
                    U[G][J][K] += CHANGE
                    DU[G][J][K] = CHANGE
                if USE_BIAS:
                    GRAD = 0
                    for S in range(T):
                        GRAD += DV[S][G][J]
                    CHANGE = LR * GRAD + ALPHA * DB[G][J]
                    B[G][J] += CHANGE
                    DB[G][J] = CHANGE

        for J in range(HN):
            CHANGE = LR * DO * H[T][J] + ALPHA * DWO[J]
            WO[J] += CHANGE
            DWO[J] = CHANGE
        if USE_BIAS:
            CHANGE = LR * DO + ALPHA * DBO
            BO += CHANGE
            DBO = CHANGE

        print(f"Row {R+1}/{len(DATA)} done")

    TOTAL = 0
    for SEQ, D in DATA:
        A, C, H, VO, Y = forward(SEQ, W, U, B, WO, BO)
        TOTAL += (D - Y) ** 2
    print("\nFinal MSE:", round(TOTAL / len(DATA), 4))

    print("\n===== FINAL WEIGHTS =====")
    print("W:", W)
    print("U:", U)
    print("B:", B)
    print("WO:", WO)
    print("BO:", BO)

    print("\n===== TESTING =====")
    SEQ = []
    for S in range(T):
        print(f"Time step {S+1}")
        X = []
        for K in range(N):
            X.append(float(input(f"Enter X{K+1}: ")))
        SEQ.append(X)

    A, C, H, VO, Y = forward(SEQ, W, U, B, WO, BO)
    for S in range(T):
        print(f"\nTime step {S+1}")
        for J in range(HN):
            print(f"H{J+1}: C =", round(C[S + 1][J], 4), " H =", round(H[S + 1][J], 4))
            for G in range(4):
                print("   ", NAMES[G], "gate =", round(A[S][G][J], 4))

    print("\nOutput: V =", round(VO, 4), " F =", round(Y, 4))
    if Y >= 0.5:
        print("Prediction: 1")
    else:
        print("Prediction: 0")

main()