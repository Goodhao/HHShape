import numpy as np

def rotate(A):
    # 把矩阵顺时针旋转90°
    A = A.tolist()
    return np.array(list(zip(*A[::-1])))

def eight_connected():
    hit_miss = ['' for _ in range(4)]
    hit_miss[0] = np.array([[-1, 1, -1], [-1, 1, 1], [0, -1, -1]])
    for i in range(1, 4):
        hit_miss[i] = rotate(hit_miss[i - 1])
    return hit_miss

def compute_juction():
    hit_miss = ['' for _ in range(4 * 3)]
    hit_miss[0] = np.array([[-1, 1, -1], [0, 1, 0], [1, 0, 1]])
    for i in range(1, 4):
        hit_miss[i] = rotate(hit_miss[i - 1])
    hit_miss[4] = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
    for i in range(5, 8):
        hit_miss[i] = rotate(hit_miss[i - 1])
    hit_miss[8] = np.array([[-1, 0, 1], [1, 1, 0], [-1, 1, -1]])
    for i in range(9, 12):
        hit_miss[i] = rotate(hit_miss[i - 1])
    return hit_miss


def compute_endpoint():
    hit_miss = ['' for _ in range(4 * 2)]
    hit_miss[0] = np.array([[-1, 1, -1], [0, 1, 0], [0, 0, 0]])
    for i in range(1, 4):
        hit_miss[i] = rotate(hit_miss[i - 1])
    hit_miss[4] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    for i in range(5, 8):
        hit_miss[i] = rotate(hit_miss[i - 1])
    return hit_miss


def process(I, P, E, in_place=False):
    res = set()
    H, W = I.shape[0], I.shape[1]
    for x, y in P:
        for k in range(len(E)):
            drop = True
            for i in range(-1, 2):
                for j in range(-1, 2):
                    val = I[x + i, y + j] if 0 <= x + i < H and 0 <= y + j < W else 0
                    if E[k][1 + i, 1 + j] == 1 and val == 0:
                        drop = False
                    if E[k][1 + i, 1 + j] == 0 and val != 0:
                        drop = False
            if in_place:
                if drop:
                    I[x, y] = 0
                    break
            else:
                if drop:
                    res.add((x, y))
                    break
    if in_place:
        return I
    else:
        return res
