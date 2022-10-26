import numpy as np
import cv2
import os
import time
import copy
import sys
import random
import tkinter
import threading
from bresenham import bresenham
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from shapely.geometry import Point, LineString, MultiPoint
from shapely.geometry.polygon import Polygon
from vectorization import fit_tangent, close_curve
from datetime import datetime

color_junction = [255, 0, 0] # blue in BGR mode
color_stroke = [0, 255, 0] # green in BGR mode

from loader import load, build_sketch, build_region

filename = sys.argv[1] if len(sys.argv) > 1 else 'bear.png'
path = os.path.join('img', filename)
original = cv2.imread(path)
sample = load(original)
path = os.path.join('img', f'out_{filename}')
cv2.imwrite(path, sample)
H, W = sample.shape[:2]

sketch, which_stroke = build_sketch(sample)
stroke_sets, region = build_region(sample, which_stroke)

def dist(p, q):
    p = np.array(p)
    q = np.array(q)
    return np.linalg.norm(p - q)

def dist_to_stroke(p, stroke):
    return min(dist(p, stroke[0]), dist(p, stroke[-1]))

for k, s in enumerate(stroke_sets):
    s = list(s)
    for i in range(len(s) - 1):
        p = sketch[s[i]]
        d = []
        for j in range(i + 1, len(s)):
            q = sketch[s[j]]
            val = np.min([dist(p[u], q[v]) for u in [0, -1] for v in [0, -1]])
            d.append(val)
        idx = np.argmin(d) + i + 1
        s[i + 1], s[idx] = s[idx], s[i + 1]
    stroke_sets[k] = s

belong = []
for idx in range(len(sketch)):
    belong.append([])
    for i, s in enumerate(stroke_sets):
        if idx in s:
            belong[-1].append(i)
print('belong', belong)

def get_boundary(s):
    points = []
    for i in range(len(s) - 1):
        p = sketch[s[i]]
        q = sketch[s[i + 1]]
        d = [dist_to_stroke(p[u], q) for u in [0, -1]]
        points += p if np.argmin(d) == 1 else list(reversed(p))
        if i == len(s) - 2:
            d = [dist_to_stroke(q[u], p) for u in [0, -1]]
            points += q if np.argmin(d) == 0 else list(reversed(q))
    if len(s) == 1:
        points = sketch[s[0]]
    return points

region_boundary = [get_boundary(s) for s in stroke_sets]

def area(s):
    polygon = Polygon(get_boundary(s)).buffer(0.01)
    return polygon.area

arrows = [0 for _ in range(len(sketch))]
for i in range(len(arrows)):
    if len(belong[i]) == 1:
        # 与背景区域相接的stroke，或者自环stroke，肯定不是遮挡造成的，不需要设置arrow
        continue
    else:
        [r1, r2] = belong[i]
        assert r1 < r2
        res = []
        for s in [stroke_sets[r1], stroke_sets[r2]]:
            old_area = area(s)
            concave_num = 0
            tmp = copy.deepcopy(sketch[i])
            sketch[i] = [sketch[i][0], sketch[i][-1]]
            new_area = area(s)
            res.append(new_area / old_area)
            sketch[i] = tmp
        arrows[i] = np.argmin(res)

def score(component, show=False):
    polys = dict()
    for k, points in component:
        polys[k] = Polygon(points).buffer(0.01)
    res = []
    tot_area = 0
    for i in range(len(stroke_sets)):
        poly_region = Polygon(region_boundary[i]).buffer(0.01)
        region_area = poly_region.area
        if i in polys.keys():
            cover_area = polys[i].intersection(poly_region).area
            hidden_area = polys[i].difference(poly_region).area
        else:
            cover_area, hidden_area = 0, 0
        tot_area += region_area
        res.append(0.7 * cover_area + 0.3 * hidden_area)
    if show:
        plt.clf()
        ax.set_aspect(1)
        plt.xlim(0, W)
        plt.ylim(0, H)
        def get_cmap(n, name='hsv'):
            return plt.cm.get_cmap(name, n)
        cmap = get_cmap(len(stroke_sets))
        cc = [i for i in range(len(stroke_sets))]
        random.shuffle(cc)
        for i in range(len(stroke_sets)):
            if i in polys.keys():
                xx, yy = polys[i].exterior.coords.xy
                X = np.array(xx.tolist())
                Y = np.array(yy.tolist())
                plt.plot(Y, H - X, color=cmap(cc[i]))
        plt.axis('off')
        canvas.draw()
        canvas.get_tk_widget().pack()
    return np.sum(res) / tot_area, res

def is_component_stroke(arrows, i, k):
    assert i in stroke_sets[k]
    if len(belong[i]) == 1:
        return True
    if arrows[i] == 2:
        return True
    if arrows[i] == 0 and belong[i][0] == k:
        return True
    if arrows[i] == 1 and belong[i][1] == k:
        return True
    return False

def component_completion(arrows, k):
    s = stroke_sets[k]
    candidate_strokes = []
    for i in range(len(s)):
        st = copy.deepcopy(sketch[s[i]])
        if len(candidate_strokes) == 1:
            d = []
            for j in [0, -1]:
                d.append(min(dist(st[0], candidate_strokes[-1][j]), dist(st[-1], candidate_strokes[-1][j])))
            if np.argmin(d) == 0:
                candidate_strokes[-1].reverse()
        if len(candidate_strokes) > 0 and np.argmin([dist(st[0], candidate_strokes[-1][-1]), dist(st[-1], candidate_strokes[-1][-1])]) == 1:
            st.reverse()
        candidate_strokes.append(st)
    new_strokes = []
    last = -1
    first = -1
    for i in range(len(s)):
        if is_component_stroke(arrows, s[i], k):
            if last != -1:
                if last != i - 1:
                    p1, p2 = candidate_strokes[last][-1], candidate_strokes[i][0]
                    t1, t2 = fit_tangent(candidate_strokes[last][-15:], p1), fit_tangent(candidate_strokes[i][:15], p2)
                    V = [np.array(p1), np.array(p1) + np.array(t1), np.array(p2) + np.array(t2), np.array(p2)]
                    curve, _, _ = close_curve(V)
                    new_strokes.append(curve)
            else:
                first = i
            new_strokes.append(candidate_strokes[i])
            last = i
    if not (first == 0 and last == len(s) - 1):
        p1, p2 = candidate_strokes[last][-1], candidate_strokes[first][0]
        t1, t2 = fit_tangent(candidate_strokes[last][-15:], p1), fit_tangent(candidate_strokes[first][:15], p2)
        V = [np.array(p1), np.array(p1) + np.array(t1), np.array(p2) + np.array(t2), np.array(p2)]
        curve, _, _ = close_curve(V)
        new_strokes.append(curve)
    return new_strokes

def part_show(k):
    global arrows
    s = stroke_sets[k]
    print(s)
    new_strokes = component_completion(arrows, k)
    plt.clf()
    ax.set_aspect(1)
    plt.xlim(0, W)
    plt.ylim(0, H)
    for idx, stroke in enumerate(sketch):
        [X, Y] = np.transpose(stroke)
        if idx not in s:
            plt.plot(Y, H - X, color='b', ls='--')
    for idx, stroke in enumerate(new_strokes):
        [X, Y] = np.transpose(stroke)
        plt.plot(Y, H - X, color='r')
    plt.axis('off')          
    canvas.draw()
    canvas.get_tk_widget().pack()

def fitness(arrows, show=False):

    component = []
    for k in range(len(stroke_sets)):
        strokes = component_completion(arrows, k)
        points = []
        for stroke in strokes:
            points += stroke
        component.append((k, points))
        if show:
            print(stroke_sets[k])
            part_show(k)
            time.sleep(1)

    return score(component, show=show)

def dfs(L, x):
    global best
    global opt_arrows
    if x < len(L):
        arrows[L[x]] = 0
        dfs(L, x + 1)
        arrows[L[x]] = 1
        dfs(L, x + 1)
    else:
        res = fitness(arrows)
        if res is None:
            return
        if res[0] > best:
            best = res[0]
            opt_arrows = copy.deepcopy(arrows)

def mcts(L):
    global best
    global arrows
    global opt_arrows
    begin = datetime.now()
    for x in L:
        arrows[x] = 2
    from monte_carlo_tree_search import MCTS, Node

    def reward(self):
        global best
        global opt_arrows
        val = fitness(list(self.data))[0]
        if val > best:
            best = val
            opt_arrows = list(self.data)
        return val
    Node.reward = reward
    tree = MCTS()
    node = Node(tuple(arrows))
    while not node.is_terminal():
        for i in range(10):
            tree.do_rollout(node)
        node = tree.choose(node)
    end = datetime.now()
    print('mcts time:', (end - begin).seconds)

root = tkinter.Tk()
root.geometry(f'{W}x{H}')
root.wm_title("demo")

fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)

arrows_loc = []
arrows_dir = []
for idx, stroke in enumerate(sketch):
    [X, Y] = np.transpose(stroke)
    plt.plot(Y, H - X, color='b', ls='--')
    [x, y] = stroke[len(stroke) // 2]
    arrows_loc.append([x, y])
    if len(belong[idx]) == 1:
        r = belong[idx][0]
        L = []
        for dx in range(-5, 5):
            for dy in range(-5, 5):
                if region[x + dx, y + dy] == r + 2:
                    L.append([x + dx, y + dy])
        p = np.mean(L, axis=0)
        v = np.array([p[0] - x, p[1] - y])
        v /= np.linalg.norm(v)
        v = -v
        arrows_dir.append(v)
    else:
        r1 = belong[idx][0]
        r2 = belong[idx][1]
        L1 = []
        L2 = []
        for dx in range(-5, 5):
            for dy in range(-5, 5):
                if region[x + dx, y + dy] == r1 + 2:
                    L1.append([x + dx, y + dy])
                if region[x + dx, y + dy] == r2 + 2:
                    L2.append([x + dx, y + dy])
        p1 = np.mean(L1, axis=0)
        p2 = np.mean(L2, axis=0)
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v /= np.linalg.norm(v)
        arrows_dir.append(v)

def show_arrows():
    plt.clf()
    ax.set_aspect(1)
    plt.xlim(0, W)
    plt.ylim(0, H)
    for idx, stroke in enumerate(sketch):
        [X, Y] = np.transpose(stroke)
        plt.plot(Y, H - X, color='b', ls='--')
        [x, y] = arrows_loc[idx]
        v = arrows_dir[idx]
        if arrows[idx] == 2:
            continue
        if arrows[idx] == 0 or len(belong[idx]) == 1:
            plt.arrow(y, H - x, v[1], -v[0], width = 1, color='r')
        elif arrows[idx] == 1:
            plt.arrow(y, H - x, -v[1], v[0], width = 1, color='r')
    plt.axis('off')
    canvas.draw()
    canvas.get_tk_widget().pack()

global_score, score_per_region = fitness(arrows)
best = global_score
opt_arrows = copy.deepcopy(arrows)
k = min(4, len(stroke_sets))
r = np.argpartition(score_per_region, -k)[-k:]
to_modify = []
for i in r:
    to_modify += stroke_sets[i]
to_modify = sorted(list(set(to_modify)))
mcts(to_modify)
arrows = opt_arrows
show_arrows()


yesno = tkinter.IntVar(value=0)
mouse_x = tkinter.IntVar(value=0)
mouse_y = tkinter.IntVar(value=0)
user_input = []

def on_button_press(event):
    yesno.set(1)
    mouse_x.set(event.xdata)
    mouse_y.set(event.ydata)
    global user_input
    x, y = event.xdata, event.ydata
    user_input = [[x, y]]
    k = np.argmin([(x - p[1]) ** 2 + (y - (H - p[0])) ** 2 for p in arrows_loc])
    arrows[k] += 1
    if arrows[k] > 2:
        arrows[k] = 0
    show_arrows()

def on_button_release(event):
    yesno.set(0)

def on_mouse_move(event):
    if yesno.get() == 0:
        return
    x, y = event.xdata, event.ydata
    user_input.append([H - y, x])
    plt.plot([mouse_x.get(), x], [mouse_y.get(), y], color='r')
    canvas.draw()
    canvas.get_tk_widget().pack()
    mouse_x.set(x)
    mouse_y.set(y)

def work():
    print(arrows)
    global_score, score_per_region = fitness(arrows)
    print(score_per_region)
    print(f'评价函数：{global_score}')
    fitness(arrows, show=True)

def select():
    global arrows
    IoU = []
    A = Polygon(user_input).buffer(0.01)
    for i in range(len(stroke_sets)):
        B = Polygon(region_boundary[i]).buffer(0.01)
        IoU.append(B.intersection(A).area / B.union(A).area)
    k = np.argmax(IoU)
    print(f'选定部件{k}')

    mx = -1e5
    record = []
    polygon0 = MultiPoint(user_input).convex_hull

    def adjust(L, x):
        nonlocal mx
        nonlocal record
        nonlocal polygon0
        global arrows
        if x < len(L):
            arrows[L[x]] = 0
            adjust(L, x + 1)
            arrows[L[x]] = 1
            adjust(L, x + 1)
        else:
            new_strokes = component_completion(arrows, k)
            points = []
            for stroke in new_strokes:
                points += stroke
            if not LineString(points).is_simple:
                return
            polygon1 = Polygon(points).buffer(0.01)
            intersect = polygon0.intersection(polygon1).area
            union = polygon0.union(polygon1).area
            iou = intersect / union
            val = iou
            if val > mx:
                mx = val
                record = copy.deepcopy(arrows)
    
    s = stroke_sets[k]
    if len(s) < 5:
        adjust(s, 0)
        arrows = record
    print(arrows)
    part_show(k)

def on_key_press(event):
    print('press ', event.key)
    sys.stdout.flush()
    if event.key == 'enter':
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
    elif event.key == 'escape':
        thread = threading.Thread(target=show_arrows, daemon=True)
        thread.start()
    elif event.key == 'a':
        thread = threading.Thread(target=select, daemon=True)
        thread.start()

canvas.mpl_connect("button_press_event", on_button_press)
canvas.mpl_connect("button_release_event", on_button_release)
canvas.mpl_connect('key_press_event', on_key_press)
canvas.mpl_connect('motion_notify_event', on_mouse_move)

tkinter.mainloop()