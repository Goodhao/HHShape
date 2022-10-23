import numpy as np
import math
import copy
import bezier
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.optimize import newton
import warnings


def atan2(y, x):
    val = math.atan2(y, x)
    if val < 0:
        val += 2 * math.pi
    return val

def angle_diff(angle1, angle2):
    res = abs(angle1 - angle2)
    res = min(res, 2 * math.pi - res)
    return res

def unit(v):
    return v / np.linalg.norm(v)

def bernstein(n, i, t):
    return math.comb(n, i) * math.pow(t, i) * math.pow(1 - t, n - i)

def fit_tangent(points, p, outward=False):
    # 返回曲线上p点的单位切向量
    points, p = np.array(points), np.array(p)
    n = len(points)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    A[0, 0] = 2 * np.sum([points[i, 0] ** 2 for i in range(n)])
    A[0, 1] = 2 * np.sum([points[i, 0] for i in range(n)])
    A[0, 2] = - p[0]
    A[1, 1] = 2 * n
    A[1, 2] = -1
    A[2, 2] = 0
    A[1, 0] = A[0, 1]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    b[0] = 2 * np.sum([points[i, 0] * points[i, 1] for i in range(n)])
    b[1] = 2 * np.sum([points[i, 1] for i in range(n)])
    b[2] = - p[1]
    # t = np.linalg.solve(A, b)
    t = np.linalg.lstsq(A, b, rcond=-1)[0]
    t = np.array([1, t[0]])
    t /= np.linalg.norm(t)
    mp = np.mean(points, axis=0)
    if not outward:
        if np.dot(t, mp - p) < 0:
            t = -t
    else:
        if np.dot(t, mp - p) > 0:
            t = -t
    return t

def fit_curve(points, eps, t0=None, t1=None):
    points = np.array(points)
    t = ['' for i in range(2)]
    m = min(len(points), 3)
    t[0] = fit_tangent(points[:m], points[0]) if t0 is None else t0
    t[1] = fit_tangent(points[len(points) - m:], points[len(points) - 1]) if t1 is None else t1
    n = len(points)
    u = np.zeros(n)
    s = 0
    for i in range(1, n):
        s += np.linalg.norm(points[i] - points[i-1])
        u[i] = s
    for i in range(1, n):
        u[i] /= s
    A = np.zeros((2, 2))
    b = np.zeros(2)
    V = ['' for i in range(4)]
    V[0] = points[0]
    V[3] = points[n - 1]

    A[0, 0] = np.sum([np.linalg.norm(t[0] * bernstein(3, 1, u[i])) ** 2 for i in range(n)])
    A[0, 1] = np.sum([np.dot(t[0] * bernstein(3, 1, u[i]), t[1] * bernstein(3, 2, u[i])) for i in range(n)])
    b[0] = np.sum([np.dot(points[i] - V[0] * bernstein(3, 0, u[i]) - V[0] * bernstein(3, 1, u[i]) - V[3] * bernstein(3, 2, u[i]) - V[3] * bernstein(3, 3, u[i]),
    t[0] * bernstein(3, 1, u[i])) for i in range(n)])
    A[1, 0] = np.sum([np.dot(t[0] * bernstein(3, 1, u[i]), t[1] * bernstein(3, 2, u[i])) for i in range(n)])
    A[1, 1] = np.sum([np.linalg.norm(t[1] * bernstein(3, 2, u[i])) ** 2 for i in range(n)])
    b[1] = np.sum([np.dot(points[i] - V[0] * bernstein(3, 0, u[i]) - V[0] * bernstein(3, 1, u[i]) - V[3] * bernstein(3, 2, u[i]) - V[3] * bernstein(3, 3, u[i]),
    t[1] * bernstein(3, 2, u[i])) for i in range(n)])

    alpha = np.linalg.solve(A, b)
    V[1] = V[0] + alpha[0] * t[0]
    V[2] = V[3] + alpha[1] * t[1]

    a = [V[i][0] for i in range(4)]
    b = [V[i][1] for i in range(4)]
    nodes = np.asfortranarray([a, b])
    _curve = bezier.Curve(nodes, degree=3)
    # u = reparameterize(_curve, points, u)
    ds = np.linspace(0.0, 1.0, 100)
    curve = np.transpose(_curve.evaluate_multi(ds)).tolist()
    curve_sample = np.transpose(_curve.evaluate_multi(u)).tolist()
    error = np.max([np.linalg.norm(curve_sample[i] - points[i]) for i in range(n)])
    if error < eps:
        return curve, V
    else:
        spilt = np.argmax([np.linalg.norm(curve_sample[i] - points[i]) for i in range(n)])
        if spilt + 1 < 4 or n - spilt < 4:
            # print('无法继续分裂')
            return curve, V
        # print('分裂')
        t = fit_tangent(points, points[spilt])
        curve1, V1 = fit_curve(points[:spilt + 1], eps, t0=t0, t1=t)
        curve2, V2 = fit_curve(points[spilt:], eps, t0=t, t1=t1)
        curve = curve1 + curve2
        V = V1 + V2
        return curve, V

def findroot(u, p, curve):
    eps = 0.01
    def f(u):
        q = curve.evaluate(u)
        d1 = (curve.evaluate(min(1, u + eps)) - curve.evaluate(u)) / eps
        q = np.squeeze(q)
        d1 = np.squeeze(d1)
        return np.dot(q - p, d1)
    def fprime(u):
        q = curve.evaluate(u)
        d1 = (curve.evaluate(min(1, u + eps)) - curve.evaluate(u)) / eps
        d2 = (curve.evaluate(min(1, u + eps)) + curve.evaluate(max(0, u - eps)) - 2 * curve.evaluate(u)) / (2 * eps)
        q = np.squeeze(q)
        d1 = np.squeeze(d1)
        d2 = np.squeeze(d2)
        return np.dot(d1, d1) + np.dot(q - p, d2)
    root = newton(f, u, fprime=fprime, tol=0.1)
    return root

def reparameterize(curve, points, u):
    n = len(u)
    for i in range(1, n - 1):
        u[i] = findroot(u[i], points[i], curve)
    return u


def objective(alpha, V):
    warnings.filterwarnings("error")
    t = ['' for i in range(2)]
    t[0] = unit(V[0] - V[1])
    t[1] = unit(V[len(V) - 1] - V[len(V) - 2])
    v0 = V[len(V) - 1]
    v1 = V[len(V) - 1] + alpha[1] * t[1]
    v2 = V[0] + alpha[0] * t[0]
    v3 = V[0]
    a = [v0[0], v1[0], v2[0], v3[0]]
    b = [v0[1], v1[1], v2[1], v3[1]]
    nodes = np.asfortranarray([a, b])
    curve = bezier.Curve(nodes, degree=3)
    u = np.linspace(0.0, 1.0, 100)
    curve = np.transpose(curve.evaluate_multi(u))
    dxdt = np.gradient(curve[:, 0])
    dydt = np.gradient(curve[:, 1])
    dx2dt = np.gradient(dxdt)
    dy2dt = np.gradient(dydt)
    curvature = np.abs(dx2dt * dydt - dxdt * dy2dt) / (dxdt ** 2 + dydt ** 2) ** 1.5
    res = 0
    for i in range(len(curve) - 1):
        ds = np.linalg.norm(curve[i] - curve[i + 1])
        res += 1 / ds * curvature[i] ** 2
    return res
    # n = len(curve)
    # res = 0
    # ds = []
    # angles = []
    # k = []
    # dk = []
    # for i in range(n - 1):
    #     ds.append(np.linalg.norm(curve[i] - curve[i + 1]))

    # for i in range(n):
    #     if i == 0:
    #         u1 = unit(curve[i + 1] - curve[i])
    #         angle = atan2(t[0][1], t[0][0]) - atan2(u1[1], u1[0])
    #         angles.append(angle)
    #         k.append(1 / ds[i] * angle)
    #     elif i == n - 1:
    #         u0 = unit(curve[i] - curve[i - 1])
    #         t[1] = -t[1]
    #         angle = atan2(u0[1], u0[0]) - atan2(t[1][1], t[1][0])
    #         angles.append(angle)
    #         k.append(1 / ds[i - 1] * angle)
    #     else:
    #         u0 = unit(curve[i] - curve[i - 1])
    #         u1 = unit(curve[i + 1] - curve[i])
    #         angle = atan2(u0[1], u0[0]) - atan2(u1[1], u1[0])
    #         angles.append(angle)
    #         # k.append(2 / (ds[i - 1] + ds[i]) * angle)
    #         k.append(1 / ds[i] * angle)
    # for i in range(n - 1):
    #     dk.append((k[i + 1] - k[i]) / ds[i])
    # # for i in range(n):
    # #     try:
    # #         if i == 0:
    # #             res += k[i] ** 2 * ds[i]
    # #         elif i == n - 1:
    # #             res += k[i] ** 2 * ds[i - 1]
    # #         else:
    # #             res += k[i] ** 2 * (ds[i - 1] + ds[i]) / 2
    # #     except RuntimeWarning:
    # #         input('QAQ')
    # for i in range(n - 1):
    #     try:
    #         res += (dk[i] / ds[i]) ** 2 * ds[i]
    #     except RuntimeWarning:
    #         input('QAQ')
    # return res * np.sum(ds)
    # # return res * (np.sum(ds) ** 3)

def dedupe(curve, V):
    _curve = []
    for i in range(len(curve)):
        if i > 1 and curve[i] == curve[i-1]:
            continue
        _curve.append(curve[i])
    _V = []
    for i in range(len(V)):
        if i > 1 and (V[i] == V[i-1]).all():
            continue
        _V.append(V[i])
    return _curve, _V


def close_curve(V):
    t = [unit(V[0] - V[1]), unit(V[-1] - V[-2])]
    bound = 0.5 * np.linalg.norm(V[0] - V[-1])
    bounds = [(0, bound), (0, bound)]
    res = minimize(objective, [0.9 * bound, 0.9 * bound], args=V, method='powell', bounds=bounds)
    alpha = np.squeeze(res.x)
    # print(alpha)
    
    v0 = V[0]
    v1 = V[0] + alpha[0] * t[0]
    v2 = V[len(V) - 1] + alpha[1] * t[1]
    v3 = V[len(V) - 1]
    a = [v0[0], v1[0], v2[0], v3[0]]
    b = [v0[1], v1[1], v2[1], v3[1]]
    nodes = np.asfortranarray([a, b])
    new_curve = bezier.Curve(nodes, degree=3)
    u = np.linspace(0.0, 1.0, 20)
    new_curve = np.transpose(new_curve.evaluate_multi(u)).tolist()
    return new_curve, [v0, v1, v2, v3], res.fun