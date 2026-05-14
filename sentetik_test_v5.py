# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:38:56 2026

@author: gokhanpoyraz
"""

"""
sentetik oluşturup farklı kombinasyonlarda denemeler yaparark sonuçlar elde ediliyor. Max_combination = none yaparsan tüm versiyonları dener
bu da 82.522 adet yapar
sigma = 0:0.1:1
elipsellik = 0:1:10
cluster outlier ratio = 0:0.001:0.03
cluster outlier removal = True/false
near ellipse outlier = 0:0.001:0.01
11x11x31x2x11 = 82522 adet deneme

bunların hepsinin ortalamasına bakılıyor.

sentetik_test_v4:
Excel çıktısı özet karşılaştırma formatındadır.
Satırlarda outlier removal algoritmaları, sütunlarda fitting methodları vardır.
1. tablo/sayfa: Mean Absolute Error
2. tablo/sayfa: Absolute Error standart sapması

"""


import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
import random
import time
import matplotlib



def plot_circle_fitting(x, y, a, b, R, title="Circle Fitting Result"):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, 'bx', label='Edge Points')

    theta = np.linspace(0, 2 * np.pi, 500)
    circle_x = a + R * np.cos(theta)
    circle_y = b + R * np.sin(theta)
    ax.plot(circle_x, circle_y, 'r-', linewidth=2, label='Fitted Circle')

    ax.plot(a, b, 'ko', label='Center')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    title="Circle Fitting Result"
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
# --- Fitting Yöntemleri ---

def fit_geometric_ls(x, y):
    def residuals(c, x, y):
        Ri = np.sqrt((x - c[0])**2 + (y - c[1])**2)
        return Ri - Ri.mean()
    x_m = np.mean(x)
    y_m = np.mean(y)
    result = least_squares(residuals, x0=[x_m, y_m], args=(x, y))
    x0, y0 = result.x
    r = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2))
    return x0, y0, r

def fit_pratt(x, y):
    x = np.array(x)
    y = np.array(y)
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suu = np.sum(u**2)
    Suv = np.sum(u*v)
    Svv = np.sum(v**2)
    Suuu = np.sum(u**3)
    Suvv = np.sum(u*v**2)
    Svvv = np.sum(v**3)
    Svuu = np.sum(v*u**2)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = np.array([0.5 * (Suuu + Suvv), 0.5 * (Svvv + Svuu)])
    uc, vc = np.linalg.solve(A, b)
    x0 = x_m + uc
    y0 = y_m + vc
    r = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2))
    return x0, y0, r

def fit_taubin(x, y):
    x = np.array(x)
    y = np.array(y)
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suu = np.sum(u**2)
    Suv = np.sum(u*v)
    Svv = np.sum(v**2)
    Suuu = np.sum(u**3)
    Suvv = np.sum(u*v**2)
    Svvv = np.sum(v**3)
    Svuu = np.sum(v*u**2)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Svuu]) / 2
    uc, vc = np.linalg.solve(A, B)
    x0 = x_m + uc
    y0 = y_m + vc
    r = np.sqrt(uc**2 + vc**2 + (Suu + Svv) / len(x))
    return x0, y0, r

def fit_ransac(x, y, iterations=100, threshold=2.0):
    best_inliers = []
    best_circle = (0, 0, 0)
    x = np.array(x)
    y = np.array(y)
    points = np.stack([x, y], axis=1)

    for _ in range(iterations):
        samples = points[random.sample(range(len(points)), 3)]
        try:
            A = np.c_[2*samples[:,0], 2*samples[:,1], np.ones(3)]
            b = samples[:,0]**2 + samples[:,1]**2
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            xc, yc = c[0], c[1]
            r = np.sqrt(c[2] + xc**2 + yc**2)
            d = np.sqrt((x - xc)**2 + (y - yc)**2)
            inliers = d[np.abs(d - r) < threshold]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_circle = (xc, yc, r)
        except:
            continue
    return best_circle

def fit_irls(x, y, iterations=10):
    x = np.array(x)
    y = np.array(y)
    weights = np.ones_like(x)
    for _ in range(iterations):
        A = np.c_[2*x, 2*y, np.ones(x.shape[0])]
        b = x**2 + y**2
        W = np.diag(weights)
        Aw = W @ A
        bw = W @ b
        c = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        x0, y0 = c[0], c[1]
        r = np.sqrt(c[2] + x0**2 + y0**2)
        d = np.sqrt((x - x0)**2 + (y - y0)**2)
        weights = 1.0 / np.maximum(np.abs(d - r), 1e-6)
        weights /= np.max(weights)
    return x0, y0, r

def fit_hyper_ls(x, y):
    x = np.array(x)
    y = np.array(y)
    D = np.column_stack((x * x + y * y, x, y, np.ones_like(x)))

    # Constraint matrix
    S = np.dot(D.T, D)

    C = np.zeros((4, 4))
    C[0, 3] = C[3, 0] = 2
    C[1, 1] = C[2, 2] = 1

    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S) @ C)
        cond = np.isreal(eigvals)
        eigvec = eigvecs[:, cond][:, 0].real
        A, B, C_, D_ = eigvec

        x0 = -B / (2 * A)
        y0 = -C_ / (2 * A)
        r = np.sqrt((B**2 + C_**2 - 4 * A * D_) / (4 * A**2))
    except:
        x0, y0, r = 0, 0, 0

    return x0, y0, r

def fit_m_estimator(x, y, iterations=10, delta=1.0):
    x = np.array(x)
    y = np.array(y)
    weights = np.ones_like(x)
    for _ in range(iterations):
        A = np.c_[2*x, 2*y, np.ones_like(x)]
        b = x**2 + y**2
        W = np.diag(weights)
        try:
            c = np.linalg.lstsq(W @ A, W @ b, rcond=None)[0]
        except:
            break
        x0, y0 = c[0], c[1]
        r = np.sqrt(c[2] + x0**2 + y0**2)
        d = np.sqrt((x - x0)**2 + (y - y0)**2)
        res = np.abs(d - r)
        weights = np.where(res <= delta, 1, delta / res)
    return x0, y0, r

def fit_lmeds(x, y):
    x = np.array(x)
    y = np.array(y)
    points = np.stack([x, y], axis=1)
    best_median = np.inf
    best_circle = (0, 0, 0)

    for _ in range(100):
        sample = points[random.sample(range(len(points)), 3)]
        try:
            A = np.c_[2*sample[:, 0], 2*sample[:, 1], np.ones(3)]
            b = sample[:, 0]**2 + sample[:, 1]**2
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            xc, yc = c[0], c[1]
            r = np.sqrt(c[2] + xc**2 + yc**2)
            d = np.sqrt((x - xc)**2 + (y - yc)**2)
            residuals = np.abs(d - r)
            median_residual = np.median(residuals)
            if median_residual < best_median:
                best_median = median_residual
                best_circle = (xc, yc, r)
        except:
            continue

    return best_circle


def fit_tls(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    M = np.column_stack((A, b))

    _, _, Vt = np.linalg.svd(M)
    v = Vt[-1, :]

    if abs(v[3]) < 1e-12:
        return np.nan, np.nan, np.nan

    # TLS: [A | b] @ [c1, c2, c3, -1]^T ≈ 0
    c = -v[:3] / v[3]

    x0, y0 = c[0], c[1]
    r2 = c[2] + x0**2 + y0**2

    if r2 < 0:
        return x0, y0, np.nan

    r = np.sqrt(r2)
    return x0, y0, r

def fit_bayesian(x, y):
    x = np.array(x)
    y = np.array(y)
    x0 = np.mean(x)
    y0 = np.mean(y)
    r = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2))
    noise = np.random.normal(0, 0.5, 100)
    r_samples = r + noise
    r_mean = np.mean(r_samples)
    return x0, y0, r_mean

def fit_gradient_descent(x, y, lr=1e-3, iterations=1000):
    x = np.array(x)
    y = np.array(y)
    x0, y0 = np.mean(x), np.mean(y)
    r = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2))

    for _ in range(iterations):
        d = np.sqrt((x - x0)**2 + (y - y0)**2)
        dr = d - r
        dx0 = np.mean((x0 - x) * dr / d)
        dy0 = np.mean((y0 - y) * dr / d)
        dr0 = -np.mean(dr)

        x0 -= lr * dx0
        y0 -= lr * dy0
        r -= lr * dr0

    return x0, y0, r

def fit_edcircle(x, y):
    x = np.array(x)
    y = np.array(y)
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x**2 + y**2)
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    D, E, F = c
    x0 = -D / 2
    y0 = -E / 2
    r = np.sqrt((D**2 + E**2) / 4 - F)
    return x0, y0, r

def remove_outliers_zscore(x, y, threshold=3.0):
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2)
    z = (r - np.mean(r)) / np.std(r)
    mask = np.abs(z) < threshold
    return x[mask], y[mask]

def remove_outliers_mad(x, y, threshold=3.5):
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.sqrt((x - np.median(x))**2 + (y - np.median(y))**2)
    mad = np.median(np.abs(r - np.median(r)))
    mask = np.abs(r - np.median(r)) / (mad + 1e-6) < threshold
    return x[mask], y[mask]

def remove_outliers_dbscan(x, y, eps=0.3, min_samples=5):
    x = np.asarray(x)
    y = np.asarray(y)
    coords = np.column_stack((x, y))

    # Normalize
    coords_scaled = StandardScaler().fit_transform(coords)

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_scaled)
    mask = db.labels_ != -1
    return x[mask], y[mask]


def remove_outliers_lof(x, y, n_neighbors=20):
    x = np.asarray(x)
    y = np.asarray(y)
    coords = np.column_stack((x, y))
    if len(x) < n_neighbors:
        print('LOF 20den az örnek var')
        return x, y

    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    mask = lof.fit_predict(coords) == 1
    return x[mask], y[mask]


def remove_outliers_percentile(x, y, lower=2.275, upper=97.725):
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2)
    low, high = np.percentile(r, [lower, upper])
    mask = (r >= low) & (r <= high)
    return x[mask], y[mask]

def remove_cluster_outliers_knn(x, y, k=10, low_factor=0.25, high_factor=2.0):
    """
    deneme4.m içindeki noktalar arası mesafe tabanlı cluster removal kısmının
    Python karşılığıdır. Her nokta için en yakın k komşu mesafesi toplanır.
    Bu toplam median değerin çok altında veya üstünde ise nokta cluster tipi
    outlier kabul edilir.
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    n = len(x)
    if n <= k + 1:
        return x, y, np.array([], dtype=float), np.array([], dtype=float)

    P = np.column_stack((x, y))

    # MATLAB'daki şu kısmın karşılığı:
    # D2 = xx + xx' - 2*(P*P'); D = sqrt(D2)
    xx = np.sum(P ** 2, axis=1)
    D2 = xx[:, None] + xx[None, :] - 2.0 * (P @ P.T)
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    np.fill_diagonal(D, np.inf)

    D_sorted = np.sort(D, axis=1)
    sum_k = np.sum(D_sorted[:, :k], axis=1)
    ttt = np.median(sum_k)

    idx_flag = (sum_k < (ttt * low_factor)) | (sum_k > (ttt * high_factor))

    x_cluster_removed = x[idx_flag]
    y_cluster_removed = y[idx_flag]
    x_clean = x[~idx_flag]
    y_clean = y[~idx_flag]

    return x_clean, y_clean, x_cluster_removed, y_cluster_removed


def _remove_outliers_local_zscore_once(
        x, y,
        threshold=3,
        window_size=50,
        std_window=50,
        x_cluster_removed=None,
        y_cluster_removed=None,
        debug_mode=0,
        iter_id=1,
        plot_result=False):
    """
    deneme4.m içindeki remove_outliers_local_zscore_proposed fonksiyonunun
    tek iterasyonluk Python karşılığıdır.
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    if x_cluster_removed is None:
        x_cluster_removed = np.array([], dtype=float)
    if y_cluster_removed is None:
        y_cluster_removed = np.array([], dtype=float)

    if len(x) == 0:
        return x, y, 0.0

    # 1. Centering and polar transformation
    xc = np.mean(x)
    yc = np.mean(y)

    theta = np.arctan2(y - yc, x - xc)
    r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    # 2. Sort by angle
    idx = np.argsort(theta)
    theta_sorted = theta[idx]
    r_sorted = r[idx]
    x_sorted = x[idx]
    y_sorted = y[idx]

    # 3. Local std calculation
    std_list = []
    stride = 20
    last_start = len(r_sorted) - std_window

    if last_start >= 0:
        for i in range(0, last_start + 1, stride):
            std_list.append(np.std(r_sorted[i:i + std_window], ddof=1))
        global_std = np.median(std_list)
    else:
        global_std = np.std(r_sorted, ddof=1)

    if not np.isfinite(global_std) or global_std < 1e-12:
        global_std = 1e-12

    # 4. Outlier removal
    n = len(r_sorted)
    mask = np.ones(n, dtype=bool)

    if n < window_size:
        mean_r = np.mean(r_sorted)
        outliers = np.abs(r_sorted - mean_r) > threshold * global_std
        mask = ~outliers
    else:
        if debug_mode == 1:
            plt.figure(figsize=(8, 6))
            h_orig, = plt.plot(theta_sorted, r_sorted, 'r.', label='Outlier')
            h_clean, = plt.plot([], [], 'b.', label='Original')
            h_win, = plt.plot([], [], 'go', markersize=6, label='Window')
            plt.xlabel(r'$\theta$ (radian)')
            plt.ylabel('r')
            plt.legend()
            plt.grid(True)

        for i in range(n - window_size + 1):
            window_idx = slice(i, i + window_size)
            window = r_sorted[window_idx]

            mean_r = np.mean(window)
            outliers = np.abs(window - mean_r) > threshold * global_std
            mask[window_idx] = mask[window_idx] & (~outliers)

            if debug_mode == 1:
                r_clean_dbg = r_sorted[mask]
                theta_clean_dbg = theta_sorted[mask]
                h_clean.set_data(theta_clean_dbg, r_clean_dbg)
                h_win.set_data(theta_sorted[window_idx], r_sorted[window_idx])
                plt.title(f'Step {i + 1} / {n - window_size + 1}')
                plt.pause(0.02)

    # 5. Filtered coordinates
    r_clean = r_sorted[mask]
    theta_clean = theta_sorted[mask]

    x_filt = r_clean * np.cos(theta_clean) + xc
    y_filt = r_clean * np.sin(theta_clean) + yc

    if plot_result:
        # Proposed method ile silinenleri doğru göstermek için mask=false olanlar çizilir.
        removed_by_proposed = ~mask

        plt.figure(figsize=(8, 6))
        plt.plot(theta_sorted[removed_by_proposed], r_sorted[removed_by_proposed], 'r.',
                 label='Removed by Proposed Method')

        if len(x_cluster_removed) > 0:
            theta_cluster_removed = np.arctan2(y_cluster_removed - yc, x_cluster_removed - xc)
            r_cluster_removed = np.sqrt((x_cluster_removed - xc) ** 2 +
                                        (y_cluster_removed - yc) ** 2)
            plt.plot(theta_cluster_removed, r_cluster_removed, 'mo', markersize=6,
                     label='Removed by Cluster Method')

        plt.plot(theta_clean, r_clean, 'b.', label='Filtered')
        plt.xlabel(r'$\theta$ (radian)')
        plt.ylabel('r')
        plt.title(f'Outlier Removal (Polar) - Iteration {iter_id}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(x_sorted[removed_by_proposed], y_sorted[removed_by_proposed], 'r.',
                 label='Removed by Proposed Method')

        if len(x_cluster_removed) > 0:
            plt.plot(x_cluster_removed, y_cluster_removed, 'mo', markersize=6,
                     label='Removed by Cluster Method')

        plt.plot(x_filt, y_filt, 'b.', label='Filtered')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Outlier Removal (Cartesian) - Iteration {iter_id}')
        plt.axis('equal')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    removed_ratio = 1.0 - (len(x_filt) / len(x))
    return x_filt, y_filt, removed_ratio


def remove_outliers_local_zscore_proposed(
        x, y,
        threshold=3,
        window_size=50,
        std_window=50,
        num_iterations=6,
        removal_ratio_threshold=0.10,
        cluster_removal_mode=1,
        debug_mode=0,
        plot_result=False):
    """
    Güncel Proposed yöntem:
    1) İsteğe bağlı KNN tabanlı cluster removal uygulanır.
    2) Local polar z-score filtresi iteratif olarak çalışır.
    3) Bir iterasyonda silinen oran removal_ratio_threshold değerinden büyükse
       o iterasyon kabul edilmez ve işlem durdurulur.
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    x_cluster_removed = np.array([], dtype=float)
    y_cluster_removed = np.array([], dtype=float)

    if cluster_removal_mode == 1:
        x, y, x_cluster_removed, y_cluster_removed = remove_cluster_outliers_knn(x, y)

    x_iter = x.copy()
    y_iter = y.copy()

    for iter_id in range(1, num_iterations + 1):
        x_new, y_new, removed_ratio = _remove_outliers_local_zscore_once(
            x_iter, y_iter,
            threshold=threshold,
            window_size=window_size,
            std_window=std_window,
            x_cluster_removed=x_cluster_removed,
            y_cluster_removed=y_cluster_removed,
            debug_mode=debug_mode,
            iter_id=iter_id,
            plot_result=plot_result
        )

        print(f'Iter {iter_id} -> removed ratio: {removed_ratio:.3f}')

        # deneme4.m ile aynı mantık: fazla agresif silme varsa iterasyon kabul edilmez.
        if removed_ratio > removal_ratio_threshold:
            print(f'Iter {iter_id} ignored (too many points removed)')
            break

        x_iter = x_new
        y_iter = y_new

    return x_iter, y_iter

# Outlier removal stratejileri
outlier_methods = {
    'Z-Score': remove_outliers_zscore,
    'MAD': remove_outliers_mad,
    'DBSCAN': remove_outliers_dbscan,
    'LOF': remove_outliers_lof,
    'Percentile': remove_outliers_percentile,
    'Proposed': remove_outliers_local_zscore_proposed,
    'None': lambda x, y: (np.array(x), np.array(y))
}

# --- Sentetik veri üretimi ---

def generate_synthetic_ellipse(
        xc=500,
        yc=500,
        a=670,
        b=690,
        n_points=1000,
        sigma=1,
        cluster_ratio=0.10,
        near_ratio=0.10,
        random_outliers=0,
        random_seed=42):
    """
    deneme4.m içindeki generate_synthetic_ellipse fonksiyonunun Python karşılığıdır.

    Not:
    - cluster_ratio ve near_ratio yüzdesel oran olarak verilir.
    - cluster_outliers = round(n_points * cluster_ratio)
    - near_ellipse_outliers = round(n_points * near_ratio)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    cluster_outliers = int(round(n_points * cluster_ratio))
    near_ellipse_outliers = int(round(n_points * near_ratio))

    # 1) İdeal elips noktaları
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = xc + a * np.cos(theta)
    y = yc + b * np.sin(theta)

    # 2) Gaussian noise eklenmiş inlier noktaları
    x_in = x + sigma * np.random.randn(len(x))
    y_in = y + sigma * np.random.randn(len(y))

    # 3) Cluster outlier
    # 3) Cluster outlier
    if cluster_outliers > 0:
    
        theta0 = 2*np.pi*np.random.rand()   # 🔥 rastgele yön
        offset = 5                          # mesafe sabit
    
        # cluster merkezi
        x_center = xc + (a + offset) * np.cos(theta0)
        y_center = yc + (b + offset) * np.sin(theta0)
    
        # cluster dağılımı
        sigma_cluster = 1
        x_out_cluster = x_center + sigma_cluster * np.random.randn(cluster_outliers)
        y_out_cluster = y_center + sigma_cluster * np.random.randn(cluster_outliers)
    
    else:
        x_out_cluster = np.array([], dtype=float)
        y_out_cluster = np.array([], dtype=float)

    # 4) Near-ellipse outlier
    if near_ellipse_outliers > 0:
        theta_o = 2 * np.pi * np.random.rand(near_ellipse_outliers)
        scale = 1 + 0.01 * np.random.randn(near_ellipse_outliers)
        x_out_near = xc + scale * a * np.cos(theta_o)
        y_out_near = yc + scale * b * np.sin(theta_o)
    else:
        x_out_near = np.array([], dtype=float)
        y_out_near = np.array([], dtype=float)

    # 5) Random outlier
    if random_outliers > 0:
        span = 1.5 * max(a, b) * 2
        x_out_random = (xc - span) + (2 * span) * np.random.rand(random_outliers)
        y_out_random = (yc - span) + (2 * span) * np.random.rand(random_outliers)
    else:
        x_out_random = np.array([], dtype=float)
        y_out_random = np.array([], dtype=float)

    # 6) Tüm outlierları birleştir
    x_out = np.concatenate([x_out_cluster, x_out_near, x_out_random])
    y_out = np.concatenate([y_out_cluster, y_out_near, y_out_random])

    # 7) Tüm noktaları birleştir
    X = np.concatenate([x_in, x_out])
    Y = np.concatenate([y_in, y_out])

    labels = np.concatenate([
        np.zeros(len(x_in), dtype=int),
        np.ones(len(x_out_cluster), dtype=int),
        2 * np.ones(len(x_out_near), dtype=int),
        3 * np.ones(len(x_out_random), dtype=int),
    ])

    # 8) Karıştır
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]
    labels = labels[idx]

    return {
        'x_in': x_in,
        'y_in': y_in,
        'x_out_cluster': x_out_cluster,
        'y_out_cluster': y_out_cluster,
        'x_out_near': x_out_near,
        'y_out_near': y_out_near,
        'x_out_random': x_out_random,
        'y_out_random': y_out_random,
        'X': X,
        'Y': Y,
        'labels': labels,
        'cluster_outliers': cluster_outliers,
        'near_ellipse_outliers': near_ellipse_outliers,
        'random_outliers': random_outliers,
    }


def plot_synthetic_ellipse_data(data, xc=500, yc=500, a=670, b=690, title='Sentetik Elips Verisi'):
    """deneme4.m içindeki orijinal veri çizimine karşılık gelen yardımcı çizim fonksiyonu."""
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    plt.axis('equal')

    if len(data['x_in']) > 0:
        plt.plot(data['x_in'], data['y_in'], 'b.', label='Inlier')
    if len(data['x_out_cluster']) > 0:
        plt.plot(data['x_out_cluster'], data['y_out_cluster'], 'ro', markersize=5, label='Cluster Outlier')
    if len(data['x_out_near']) > 0:
        plt.plot(data['x_out_near'], data['y_out_near'], 'mo', markersize=5, label='Near-Ellipse Outlier')
    if len(data['x_out_random']) > 0:
        plt.plot(data['x_out_random'], data['y_out_random'], 'go', markersize=5, label='Random Outlier')

    tt = np.linspace(0, 2 * np.pi, 400)
    x_true = xc + a * np.cos(tt)
    y_true = yc + b * np.sin(tt)
    plt.plot(x_true, y_true, 'k-', linewidth=1.5, label='Gerçek Elips')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# --- Ana işlem ---

def frange(start, stop, step, decimals=6):
    """MATLAB tarzı start:step:stop aralığını kapsayıcı üretir."""
    vals = []
    v = start
    while v <= stop + step / 2:
        vals.append(round(v, decimals))
        v += step
    return vals


def main():
    # Sabit sentetik veri parametreleri
    xc = 500
    yc = 500
    base_radius = 680
    true_diameter = 2 * base_radius  # Beklenen çap: 1360
    n_points = 1000
    random_outliers = 0
    first_random_seed = 42

    # Parametre tarama aralıkları
    # sigma: 0:0.1:1
    sigma_values = frange(0.0, 1.0, 0.2, decimals=3)

    # elipsellik: 0:1:10
    # ellipticity = 0  -> a=680, b=680
    # ellipticity = 10 -> a=670, b=690
    ellipticity_values = list(range(0, 11, 2))

    # cluster outlier ratio: 0:0.001:0.03
    cluster_ratio_values = frange(0.0, 0.03, 0.01, decimals=4)

    # cluster outlier removal true/false
    cluster_removal_modes = [False, True]

    # near ellipse outlier ratio: 0:0.001:0.01
    near_ratio_values = frange(0.0, 0.03, 0.01, decimals=4)

    # Tüm kombinasyonları çalıştırmak için None bırakın.
    # Deneme amacıyla örn. 1000 yazarsanız ilk 1000 kombinasyon çalışır.
    MAX_COMBINATIONS = None

    # Çoklu taramada grafik açmak çok yavaşlatır.
    plot_first_sample = False

    sonuc_listesi = []

    fitting_methods = [
        ('Geometric LS', fit_geometric_ls),
        ('Pratt', fit_pratt),
        ('Taubin', fit_taubin),
        ('RANSAC', fit_ransac),
        ('IRLS', fit_irls),
        ('Hyper LS', fit_hyper_ls),
        ('M-Estimator', fit_m_estimator),
        ('LMedS', fit_lmeds),
        ('TLS', fit_tls),
        ('Bayesian', fit_bayesian),
        ('Gradient Descent', fit_gradient_descent),
        ('EDCircle', fit_edcircle)
    ]

    # Proposed dışındaki yöntemler cluster_removal_mode parametresinden etkilenmez.
    # Bu nedenle gereksiz tekrar olmasın diye normal yöntemler bir kez,
    # Proposed ise cluster removal False ve True için ayrı ayrı çalıştırılır.
    non_proposed_methods = {
        name: func for name, func in outlier_methods.items()
        if name != 'Proposed'
    }

    total_base = (len(sigma_values) * len(ellipticity_values) *
                  len(cluster_ratio_values) * len(near_ratio_values))
    total_proposed = total_base * len(cluster_removal_modes)
    total_rows_expected = total_base * len(non_proposed_methods) + total_proposed

    print(f"Base kombinasyon sayısı: {total_base}")
    print(f"Proposed deney sayısı: {total_proposed}")
    print(f"Beklenen Excel satırı: {total_rows_expected}")

    sample_id = 0
    row_counter = 0

    for sigma in sigma_values:
        for ellipticity in ellipticity_values:
            a = base_radius - ellipticity
            b = base_radius + ellipticity

            for cluster_ratio in cluster_ratio_values:
                for near_ratio in near_ratio_values:
                    sample_id += 1

                    if MAX_COMBINATIONS is not None and sample_id > MAX_COMBINATIONS:
                        break

                    random_seed = first_random_seed + sample_id - 1
                    data = generate_synthetic_ellipse(
                        xc=xc,
                        yc=yc,
                        a=a,
                        b=b,
                        n_points=n_points,
                        sigma=sigma,
                        cluster_ratio=cluster_ratio,
                        near_ratio=near_ratio,
                        random_outliers=random_outliers,
                        random_seed=random_seed
                    )

                    edges_x = data['X']
                    edges_y = data['Y']
                    dosya_adi = f"synthetic_{sample_id:06d}"

                    if sample_id == 1 and plot_first_sample:
                        plot_synthetic_ellipse_data(
                            data, xc=xc, yc=yc, a=a, b=b,
                            title=f'{dosya_adi} - Sentetik Elips Verisi'
                        )

                    if len(edges_x) <= 3:
                        print(f"{dosya_adi} -> Yetersiz sentetik veri")
                        continue

                    # 1) Proposed dışındaki yöntemler
                    for outlier_name, outlier_func in non_proposed_methods.items():
                        try:
                            x_filt, y_filt = outlier_func(edges_x, edges_y)
                            x_filt = np.asarray(x_filt, dtype=float)
                            y_filt = np.asarray(y_filt, dtype=float)
                        except Exception as e:
                            print(f"{dosya_adi} ({outlier_name}) -> Outlier removal hatası: {e}")
                            continue

                        if len(x_filt) <= 3:
                            print(f"{dosya_adi} ({outlier_name}) -> Filtre sonrası yetersiz nokta")
                            continue

                        row = make_result_row(
                            dosya_adi=dosya_adi,
                            random_seed=random_seed,
                            outlier_name=outlier_name,
                            cluster_removal_mode=None,
                            edges_x=edges_x,
                            x_filt=x_filt,
                            data=data,
                            xc=xc,
                            yc=yc,
                            a=a,
                            b=b,
                            sigma=sigma,
                            ellipticity=ellipticity,
                            cluster_ratio=cluster_ratio,
                            near_ratio=near_ratio
                        )

                        run_fitting_methods(row, fitting_methods, x_filt, y_filt)
                        sonuc_listesi.append(row)
                        row_counter += 1

                    # 2) Proposed yöntemi: cluster removal False ve True olarak iki kez
                    for cluster_removal_mode in cluster_removal_modes:
                        outlier_name = 'Proposed'
                        try:
                            x_filt, y_filt = remove_outliers_local_zscore_proposed(
                                edges_x,
                                edges_y,
                                cluster_removal_mode=int(cluster_removal_mode),
                                plot_result=(sample_id == 1 and plot_first_sample)
                            )
                            x_filt = np.asarray(x_filt, dtype=float)
                            y_filt = np.asarray(y_filt, dtype=float)
                        except Exception as e:
                            print(f"{dosya_adi} ({outlier_name}, cluster={cluster_removal_mode}) -> Outlier removal hatası: {e}")
                            continue

                        if len(x_filt) <= 3:
                            print(f"{dosya_adi} ({outlier_name}, cluster={cluster_removal_mode}) -> Filtre sonrası yetersiz nokta")
                            continue

                        row = make_result_row(
                            dosya_adi=dosya_adi,
                            random_seed=random_seed,
                            outlier_name=outlier_name,
                            cluster_removal_mode=cluster_removal_mode,
                            edges_x=edges_x,
                            x_filt=x_filt,
                            data=data,
                            xc=xc,
                            yc=yc,
                            a=a,
                            b=b,
                            sigma=sigma,
                            ellipticity=ellipticity,
                            cluster_ratio=cluster_ratio,
                            near_ratio=near_ratio
                        )

                        run_fitting_methods(row, fitting_methods, x_filt, y_filt)
                        sonuc_listesi.append(row)
                        row_counter += 1

                    if sample_id % 100 == 0:
                        print(f"İşlenen temel kombinasyon: {sample_id} / {total_base}, Excel satırı: {row_counter}")

                if MAX_COMBINATIONS is not None and sample_id >= MAX_COMBINATIONS:
                    break
            if MAX_COMBINATIONS is not None and sample_id >= MAX_COMBINATIONS:
                break
        if MAX_COMBINATIONS is not None and sample_id >= MAX_COMBINATIONS:
            break

    df = pd.DataFrame(sonuc_listesi)

    # Gerçek çapa göre error hesabı
    # Error = ölçülen çap - gerçek çap.
    # Gerçek çap bu çalışmada 2*base_radius = 1360 kabul edilmiştir.
    error_df = create_error_dataframe(df, fitting_methods, true_diameter=true_diameter)

    # İstenen özet tablolar:
    # Satırlar: outlier removal algoritmaları
    # Sütunlar: fitting methodları
    mae_table, std_table, count_table = create_comparison_tables(error_df, fitting_methods)

    output_name = 'sentetik_test_v4_method_comparison.xlsx'
    with pd.ExcelWriter(output_name, engine='openpyxl') as writer:
        mae_table.to_excel(writer, sheet_name='Mean Absolute Error')
        std_table.to_excel(writer, sheet_name='Std Abs Error')
        count_table.to_excel(writer, sheet_name='Valid Count')

        # Ham sonuçları da kontrol için ekliyoruz. İstersen bu iki satırı silebilirsin.
        df.to_excel(writer, sheet_name='Raw Diameter Results', index=False)
        error_df.to_excel(writer, sheet_name='Raw Error Results', index=False)

        # Basit Excel biçimlendirme
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = 'B2'
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells:
                    try:
                        val_len = len(str(cell.value)) if cell.value is not None else 0
                        max_len = max(max_len, val_len)
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 28)

    print(f"Özet karşılaştırma Excel dosyası oluşturuldu: {output_name}")

    print('\n===== Mean Absolute Error Tablosu =====')
    print(mae_table)

    print('\n===== Absolute Error Standart Sapma Tablosu =====')
    print(std_table)

    # Spyder konsoluna en iyi 5 kombinasyonu yazdır
    print_best_5_methods(error_df, fitting_methods, top_n=5)


def make_result_row(
        dosya_adi,
        random_seed,
        outlier_name,
        cluster_removal_mode,
        edges_x,
        x_filt,
        data,
        xc,
        yc,
        a,
        b,
        sigma,
        ellipticity,
        cluster_ratio,
        near_ratio):
    """Tek Excel satırı için ortak meta bilgileri oluşturur."""
    return {
        'Dosya': dosya_adi,
        'Random Seed': random_seed,
        'Outlier Removal': outlier_name,
        'Cluster Removal Mode': cluster_removal_mode,
        'Sigma': sigma,
        'Ellipticity': ellipticity,
        'Cluster Ratio': cluster_ratio,
        'Near Ellipse Ratio': near_ratio,
        'Input Points': len(edges_x),
        'Filtered Points': len(x_filt),
        'Removed Points': len(edges_x) - len(x_filt),
        'Removed Ratio': 1 - (len(x_filt) / len(edges_x)),
        'Inlier Count': len(data['x_in']),
        'Cluster Outlier Count': len(data['x_out_cluster']),
        'Near Outlier Count': len(data['x_out_near']),
        'Random Outlier Count': len(data['x_out_random']),
        'True xc': xc,
        'True yc': yc,
        'Ellipse a': a,
        'Ellipse b': b,
        'True Diameter': 2 * ((a + b) / 2),
    }


def run_fitting_methods(row, fitting_methods, x_filt, y_filt):
    """Tüm circle fitting yöntemlerini çalıştırıp sonucu row içine ekler."""
    for name, func in fitting_methods:
        start = time.perf_counter()
        try:
            fit_xc, fit_yc, R = func(x_filt, y_filt)
            diameter = 2 * R
        except Exception as e:
            fit_xc = None
            fit_yc = None
            R = None
            diameter = None
            print(f"{row['Dosya']} ({row['Outlier Removal']}) -> {name} hatası: {e}")

        duration_ms = (time.perf_counter() - start) * 1000
        row[name] = diameter
        row[f'{name} Radius'] = R
        row[f'{name} Center X'] = fit_xc
        row[f'{name} Center Y'] = fit_yc
        row[f'{name} Time (ms)'] = duration_ms


def create_error_dataframe(df, fitting_methods, true_diameter=1360):
    """
    Çap sonuçlarını error sonuçlarına çevirir.

    Error hesabı:
        error = measured_diameter - true_diameter

    Aynı zamanda mutlak hata sütunları da eklenir:
        abs_error = abs(error)
    """
    error_df = df.copy()

    if 'True Diameter' not in error_df.columns:
        error_df['True Diameter'] = true_diameter

    for name, _ in fitting_methods:
        if name not in error_df.columns:
            continue

        measured = pd.to_numeric(error_df[name], errors='coerce')
        true_vals = pd.to_numeric(error_df['True Diameter'], errors='coerce')

        error_df[f'{name} Error'] = measured - true_vals
        error_df[f'{name} Abs Error'] = np.abs(error_df[f'{name} Error'])

    return error_df


def create_comparison_tables(error_df, fitting_methods):
    """
    İstenen özet karşılaştırma tablolarını üretir.

    Satırlar:
        Outlier removal algoritmaları.
        Proposed için Cluster Removal Mode ayrı gösterilir.

    Sütunlar:
        Fitting methodları.

    Tablo 1:
        Mean Absolute Error

    Tablo 2:
        Absolute Error değerlerinin standart sapması
    """
    work = error_df.copy()

    def row_label(row):
        outlier_name = row['Outlier Removal']
        cluster_mode = row.get('Cluster Removal Mode', None)

        if outlier_name == 'Proposed':
            return f"Proposed | Cluster Removal = {cluster_mode}"
        return str(outlier_name)

    work['Removal Algorithm'] = work.apply(row_label, axis=1)

    removal_order = [
        'Z-Score',
        'MAD',
        'DBSCAN',
        'LOF',
        'Percentile',
        'None',
        'Proposed | Cluster Removal = False',
        'Proposed | Cluster Removal = True',
    ]

    mae_table = pd.DataFrame(index=removal_order)
    std_table = pd.DataFrame(index=removal_order)
    count_table = pd.DataFrame(index=removal_order)

    for method_name, _ in fitting_methods:
        abs_col = f'{method_name} Abs Error'

        if abs_col not in work.columns:
            continue

        values = pd.to_numeric(work[abs_col], errors='coerce')
        temp = work.copy()
        temp[abs_col] = values

        grouped = temp.groupby('Removal Algorithm')[abs_col]

        mae_table[method_name] = grouped.mean()
        std_table[method_name] = grouped.std(ddof=1)
        count_table[method_name] = grouped.count()

    # Sadece gerçekten oluşan satırları tut.
    mae_table = mae_table.dropna(how='all')
    std_table = std_table.loc[mae_table.index]
    count_table = count_table.loc[mae_table.index]

    # Daha okunabilir çıktı için küçük sayıları 6 ondalıkla yuvarla.
    mae_table = mae_table.round(6)
    std_table = std_table.round(6)

    return mae_table, std_table, count_table


def print_best_5_methods(error_df, fitting_methods, top_n=5):
    """
    Outlier removal + fitting kombinasyonlarını ortalama mutlak hataya göre sıralar
    ve en iyi 5 sonucu Spyder konsoluna basar.
    """
    records = []

    for name, _ in fitting_methods:
        abs_col = f'{name} Abs Error'
        err_col = f'{name} Error'

        if abs_col not in error_df.columns:
            continue

        grouped = error_df.groupby(['Outlier Removal', 'Cluster Removal Mode'], dropna=False).agg(
            Mean_Abs_Error=(abs_col, 'mean'),
            Median_Abs_Error=(abs_col, 'median'),
            Std_Abs_Error=(abs_col, 'std'),
            Mean_Error=(err_col, 'mean'),
            Valid_Count=(abs_col, 'count')
        ).reset_index()

        grouped['Fitting Method'] = name
        records.append(grouped)

    if not records:
        print('En iyi 5 yöntem hesaplanamadı. Error sütunları bulunamadı.')
        return

    summary = pd.concat(records, ignore_index=True)
    summary = summary.replace({np.nan: None})
    summary = summary.sort_values('Mean_Abs_Error', ascending=True).head(top_n)

    print(f'\n===== En iyi {top_n} yöntem / Mean Absolute Error değerine göre =====')
    for rank, (_, row) in enumerate(summary.iterrows(), start=1):
        cluster_mode = row['Cluster Removal Mode']
        if cluster_mode is None:
            cluster_text = '-'
        else:
            cluster_text = str(cluster_mode)

        print(
            f"{rank}) Outlier: {row['Outlier Removal']} | "
            f"Cluster Removal: {cluster_text} | "
            f"Fitting: {row['Fitting Method']} | "
            f"Mean Abs Error: {row['Mean_Abs_Error']:.6f} | "
            f"Median Abs Error: {row['Median_Abs_Error']:.6f} | "
            f"Mean Error: {row['Mean_Error']:.6f} | "
            f"N: {int(row['Valid_Count'])}"
        )

    summary_output_name = 'sentetik_test_v5_best5_summary.xlsx'
    summary.to_excel(summary_output_name, index=False)
    print(f"En iyi {top_n} özet dosyası oluşturuldu: {summary_output_name}")


if __name__ == "__main__":
    main()
