import numpy as np

def gauss_kernel(kernel_size=3, sigma=0.8):
    gauss = np.zeros((kernel_size, kernel_size))
    half = kernel_size // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            gauss[half + i, half + j] = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (i ** 2 + j ** 2) / (2 * sigma ** 2))
    # 整数化
    c = gauss[0, 0]
    for i in range(kernel_size):
        for j in range(kernel_size):
            gauss[i, j] = int(gauss[i, j] / c)
    # 归一化
    gauss /= np.sum(gauss)
    return gauss

def gauss_filter(I, T):
    img = I
    half_h = T.shape[0] // 2
    half_w = T.shape[1] // 2
    for x in range(I.shape[0]):
        for y in range(I.shape[1]):
            val = 0
            for i in range(-half_h, half_h + 1):
                for j in range(-half_w, half_w + 1):
                    tmp = I[x + i][y + j] if 0 <= x + i < I.shape[0] and 0 <= y + j < I.shape[1] else 255
                    val += tmp * T[half_h + i][half_w + j]
            img[x, y] = val
    return img

def pearson(I, T, x, y):
    assert 0 <= x < I.shape[0]
    assert 0 <= y < I.shape[1]
    subI = []
    half_h = T.shape[0] // 2
    half_w = T.shape[1] // 2
    for i in range(-half_h, half_h + 1):
        row = []
        for j in range(-half_w, half_w + 1):
            if 0 <= x + i < I.shape[0] and 0 <= y + j < I.shape[1]:
                row.append(I[x + i][y + j])
            else:
                row.append(0)
        subI.append(row)
    uI = np.mean(np.array(subI))
    uT = np.mean(np.array(T))
    a, b, c = 0, 0, 0
    for i in range(-half_h, half_h + 1):
        for j in range(-half_w, half_w + 1):
            II = I[x + i][y + j] if 0 <= x + i < I.shape[0] and 0 <= y + j < I.shape[1] else 255
            a += (II - uI) * (T[half_h + i][half_w + j] - uT)
            b += (II - uI) ** 2
            c += (T[half_h + i][half_w + j] - uT) ** 2
    res = a / np.sqrt(b * c) if a != 0 else 0
    # if np.isnan(res):
    #     print(a, b, c)
    #     input()
    return res

def extraction(I, T):
    if not isinstance(T, list):
        T = [T]
    I = np.array(I).astype(float)
    H, W = I.shape
    pcc = np.zeros((H, W, len(T)))
    for i in range(H):
        for j in range(W):
            for k in range(len(T)):
                pcc[i, j, k] = pearson(I, T[k], i, j)
                np.clip(pcc[i, j, k], -1, 1)
    maxpcc = np.zeros((H, W))
    minpcc = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            maxpcc[i, j] = np.max(pcc[i, j, :])
            minpcc[i, j] = np.min(pcc[i, j, :])
    mpcc = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if abs(maxpcc[i, j]) > abs(minpcc[i, j]):
                mpcc[i, j] = maxpcc[i, j]
            else:
                mpcc[i, j] = minpcc[i, j]
    return mpcc