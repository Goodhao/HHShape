import numpy as np
import copy
import cv2

def load(img):

    from line_extraction import gauss_kernel, extraction
    from line_thining import thin, prune
    from morphology import process, eight_connected, compute_juction, compute_endpoint
    def next_odd(x):
        return x if x % 2 == 1 else x + 1
    def remove_background(img):
        queue = [(0, 0)]
        cur = 0
        H, W = img.shape[:2]
        while cur < len(queue):
            x, y = queue[cur]
            cur += 1
            for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                if 0 <= x + dx < H and 0 <= y + dy < W and img[x + dx, y + dy] == 255:
                    img[x + dx, y + dy] = 0
                    queue.append((x + dx, y + dy))
        return img

    H, W = img.shape[:2]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img[img == 255] = 254
    img[img == 0] = 255
    img[img == 254] = 0
    # w = (2, 4, 8, 16, 32)
    # T = [gauss_kernel(kernel_size=next_odd(int(7 * x / 3)), sigma=(x / 3)) for x in w]
    # img = extraction(img, T)
    # for x, y in np.ndindex((H, W)):
    #     img[x, y] = 255 if img[x, y] < 0.1 else 0
    # img = img.astype(np.uint8)
    # img = remove_background(img)
    # cv2.imwrite('before.png', img)

    img = thin(img)
    print('thining ok')

    h = [eight_connected(), compute_juction(), compute_endpoint()]
    img = process(img, h[0], in_place=True)
    is_junction = process(img, h[1])
    is_endpoint = process(img, h[2])
    update = True
    while update:
        img, update = prune(img, is_endpoint, is_junction)
        is_junction = process(img, h[1])
        is_endpoint = process(img, h[2])

    color_junction = [255, 0, 0] # blue in BGR mode
    color_stroke = [0, 255, 0] # green in BGR mode

    out = np.zeros((H, W, 3))
    for x, y in np.ndindex((H, W)):
        if is_junction[x, y]:
            out[x, y] = np.array(color_junction)
        elif img[x, y] != 0:
            out[x, y] = np.array(color_stroke)
        else:
            out[x, y] = np.array([0, 0, 0])
    for x, y in np.ndindex((H, W)):
        if is_junction[x, y]:
            for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (1, -1), (1, 1), (-1, 1)]:
                if 0 <= x + dx < H and 0 <= y + dy < W and (out[x + dx, y + dy] == color_stroke).all():
                    out[x + dx, y + dy] = color_junction
    return out

def build_sketch(img):

    color_junction = [255, 0, 0] # blue in BGR mode
    color_stroke = [0, 255, 0] # green in BGR mode

    H, W = img.shape[:2]

    sketch = []
    which_stroke = {}

    def pick_stroke(res, I, x, y, c):
        res.append([x, y])
        I[x, y] = [0, 0, 0]
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (1, -1), (1, 1), (-1, 1)]:
            if 0 <= x + dx < H and 0 <= y + dy < W and (I[x + dx, y + dy] == c).all():
                pick_stroke(res, I, x + dx, y + dy, c)
                break

    def pick_junction(res, I, x, y, c):
        res.append([x, y])
        I[x, y] = [0, 0, 0]
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (1, -1), (1, 1), (-1, 1)]:
            if 0 <= x + dx < H and 0 <= y + dy < W and (I[x + dx, y + dy] == c).all():
                pick_junction(res, I, x + dx, y + dy, c)

    picked = copy.deepcopy(img)
    for x, y in np.ndindex((H, W)):
        if (picked[x, y] == color_stroke).all():
            deg = 0
            for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (1, -1), (1, 1), (-1, 1)]:
                if 0 <= x + dx < H and 0 <= y + dy < W and (picked[x + dx, y + dy] == color_stroke).all():
                    deg += 1
            if deg == 1:
                res = []
                pick_stroke(res, picked, x, y, color_stroke)
                for p in res:
                    which_stroke[tuple(p)] = len(sketch)
                sketch.append(res)
    # 成环的stroke
    for x, y in np.ndindex((H, W)):
        if (picked[x, y] == color_stroke).all():
            res = []
            pick_stroke(res, picked, x, y, color_stroke)
            for p in res:
                which_stroke[tuple(p)] = len(sketch)
            sketch.append(res)

    return sketch, which_stroke

def build_region(img, which_stroke):

    color_junction = [255, 0, 0] # blue in BGR mode
    color_stroke = [0, 255, 0] # green in BGR mode

    H, W = img.shape[:2]

    region = np.zeros((H, W))
    for x, y in np.ndindex((H, W)):
        if (img[x, y] != [0, 0, 0]).any():
            region[x, y] = -1
    def flood(res, I, x, y, region_ID=1):
        H, W = I.shape[:2]
        queue = []
        I[x, y] = region_ID
        queue.append([x, y])
        while len(queue) > 0:
            [x, y] = queue[0]
            del queue[0]
            for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                if 0 <= x + dx < H and 0 <= y + dy < W and I[x + dx, y + dy] == 0:
                    I[x + dx, y + dy] = region_ID
                    queue.append([x + dx, y + dy])
        if I[0, 0] == region_ID:
            # 背景区域
            for x, y in np.ndindex((H, W)):
                if I[x, y] == region_ID:
                    for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                        if 0 <= x + dx < H and 0 <= y + dy < W:
                            if I[x + dx, y + dy] != region_ID and (img[x + dx, y + dy] == color_stroke).all():
                                res.add(which_stroke[(x + dx, y + dy)])
            return
        # 处理无法同胚到圆盘的区域（亏格>0）
        I2 = np.zeros((H, W))
        queue = []
        I2[0, 0] = 1
        queue.append([0, 0])
        while len(queue) > 0:
            [x, y] = queue[0]
            del queue[0]
            for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                if 0 <= x + dx < H and 0 <= y + dy < W:
                    if I[x + dx, y + dy] != region_ID and I2[x + dx, y + dy] == 0:
                        I2[x + dx, y + dy] = 1
                        queue.append([x + dx, y + dy])
        for x, y in np.ndindex((H, W)):
            if I2[x, y] == 0:
                for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    if 0 <= x + dx < H and 0 <= y + dy < W:
                        if I2[x + dx, y + dy] != 0 and (img[x + dx, y + dy] == color_stroke).all():
                            res.add(which_stroke[(x + dx, y + dy)])

    stroke_sets = []
    region_ID = 1
    for x, y in np.ndindex((H, W)):
        if region[x, y] == 0:
            stroke_sets.append(set())
            flood(stroke_sets[-1], region, x, y, region_ID)
            if len(stroke_sets[-1]) == 0:
                del stroke_sets[-1]
                region[region == region_ID] = 0
            else:
                region_ID += 1

    del stroke_sets[0] # 删除背景区域的边界
    print(stroke_sets)
    return stroke_sets, region
