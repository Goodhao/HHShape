import numpy as np

def thin(I):
    # 要求 I 是像素值∈{0, 255}的二值化图像
    I = np.array(I).astype(np.uint8)
    H, W = I.shape
    I[I == 255] = 1
    cnt = 0
    while True:
        flag = True
        mask = np.zeros((H, W))
        for x in range(H):
            for y in range(W):
                if I[x, y] == 0:
                    continue
                P = np.zeros((3, 3))
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        P[1 + i, 1 + j] = I[x + i, y + j] if 0 <= x + i < H and 0 <= y + j < W else 0
                P = 1 - P
                B = np.sum(P) - P[1, 1]
                t = [P[0, 1], P[0, 2], P[1, 2], P[2, 2], P[2, 1], P[2, 0], P[1, 0], P[0, 0], P[0, 1]]
                A = 0
                for i in range(len(t) - 1):
                    if t[i] == 0 and t[i + 1] == 1:
                        A += 1
                if 2 <= B <= 6 and A == 1:
                    if cnt % 2 == 0:
                        if P[0, 1] + P[1, 2] + P[2, 1] >= 1 and P[1, 2] + P[2, 1] + P[1, 0] >= 1:
                            mask[x, y] = 1
                            flag = False
                    else:
                        if P[0, 1] + P[1, 2] + P[1, 0] >= 1 and P[0, 1] + P[2, 1] + P[1, 0] >= 1:
                            mask[x, y] = 1
                            flag = False
        I = np.logical_xor(I, mask).astype(np.uint8)
        cnt += 1
        if flag:
            break

    I[I == 1] = 255
    return I


def prune(I, is_endpoint, is_junction):
    update = False
    H, W = I.shape
    for x in range(H):
        for y in range(W):
            if I[x, y] == 0:
                continue
            alone = True
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not (i == 0 and j == 0):
                        if 0 <= x + i < H and 0 <= y + j < W and I[x + i, y + j] != 0:
                            alone = False
                            break
            if alone:
                I[x, y] = 0
                update = True
    for x in range(H):
        for y in range(W):
            if is_endpoint[x, y] and I[x, y] != 0:
                queue = [(x, y)]
                I[x, y] = 0
                update = True
                cur = 0
                while cur < len(queue):
                    x, y = queue[cur]
                    cur += 1
                    stop = False
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if not (i == 0 and j == 0):
                                if 0 <= x + i < H and 0 <= y + j < W and I[x + i, y + j] != 0 and is_junction[x + i, y + j]:
                                    stop = True
                    if stop:
                        continue
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if not (i == 0 and j == 0):
                                if 0 <= x + i < H and 0 <= y + j < W and I[x + i, y + j] != 0 and not is_junction[x + i, y + j]:
                                    I[x + i, y + j] = 0
                                    queue.append((x + i, y + j))
    return I, update
