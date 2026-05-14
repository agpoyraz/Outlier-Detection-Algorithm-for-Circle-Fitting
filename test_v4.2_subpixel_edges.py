# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:08:04 2026

@author: gokhanpoyraz
"""

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
from scipy.optimize import least_squares
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
import random
import time
import matplotlib


import cv2
from subpixel_edges import subpixel_edges

# NumPy 1.20+ uyumluluğu için
if not hasattr(np, 'bool'):
    np.bool = bool

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
    

# ------------------------------------------------------------
# True diameter values for resim1.png ... resim45.png in mm
# ------------------------------------------------------------
TRUE_DIAMETERS_MM = [
    23.6644,
    23.6731,
    23.6711,
    23.6812,
    23.6748,
    23.6678,
    23.6731,
    23.6677,
    23.6687,
    23.6659,
    23.6636,
    23.6655,
    23.6638,
    23.6673,
    23.6641,
    23.6838,
    23.6834,
    23.6914,
    23.6614,
    23.6696,
    23.6612,
    23.6643,
    23.6748,
    23.6699,
    23.6578,
    23.6635,
    23.6780,
    23.6678,
    23.6911,
    23.6616,
    23.6639,
    23.6650,
    23.6749,
    23.6646,
    23.6722,
    23.6597,
    23.6641,
    23.6701,
    23.6683,
    23.6751,
    23.6685,
    23.6696,
    23.6731,
    23.6690,
    23.6832
]

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
    x = np.array(x)
    y = np.array(y)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    M = np.column_stack((A, b))
    _, _, Vt = np.linalg.svd(M)
    v = Vt[-1, :]
    c = -v[:3] / v[3]
    x0, y0 = c[0], c[1]
    r = np.sqrt(c[2] + x0**2 + y0**2)
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
        removal_ratio_threshold=0.05,
        cluster_removal_mode=1,
        debug_mode=0,
        plot_result=True):
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

        # İlk iterasyonda fazla silme varsa:
        # sonucu kabul et ama devam etme.
        if removed_ratio > removal_ratio_threshold:
        
            print(f'Iter {iter_id} exceeded threshold -> stopping iterations')
        
            # İlk iterasyonsa kabul et
            if iter_id == 1:
                x_iter = x_new
                y_iter = y_new
        
            break
        
        # Normal durumda iterasyonu kabul et
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

    # Proposed method - Cluster ON
    'Proposed - Cluster ON': lambda x, y: remove_outliers_local_zscore_proposed(
        x, y,
        cluster_removal_mode=1
    ),

    # Proposed method - Cluster OFF
    'Proposed - Cluster OFF': lambda x, y: remove_outliers_local_zscore_proposed(
        x, y,
        cluster_removal_mode=0
    ),

    'None': lambda x, y: (np.array(x), np.array(y))
}

# --- Ana işlem ---

def read_image_cv_unicode(path):
    """
    Türkçe karakter içeren Windows path'leri için OpenCV tabanlı okuma.
    cv2.imread yerine np.fromfile + cv2.imdecode kullanılır.
    """
    img = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    return img


def main():
    # NumPy 1.20+ compatibility fixes for subpixel_edges package
    np.bool = bool
    np.int = int
    np.float = float
    np.object = object

    # Subpixel edge parametreleri
    # Kullanılacak yapı:
    # edges = subpixel_edges(img_gray, 25, 2, 2)
    subpixel_threshold = 25
    subpixel_iters = 2
    subpixel_order = 2

    klasor = r"C:\\Users\\gokhanpoyraz\\Desktop\\Çalışmalar\\Circle_Fitting_Comparison\\Circle-Fitting\\Images_Outer"

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

    sonuc_listesi = []

    # Her yöntem için resim1 üzerinden hesaplanacak mm/px katsayısı
    scale_dict = {}

    for i in range(1, len(TRUE_DIAMETERS_MM) + 1):
        dosya_adi = f"resim{i}.png"
        tam_yol = os.path.join(klasor, dosya_adi)
        true_mm = TRUE_DIAMETERS_MM[i - 1]

        try:
            img = read_image_cv_unicode(tam_yol)

            if img is None:
                print(f"{dosya_adi} -> Görüntü okunamadı: {tam_yol}")
                continue

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

            edges = subpixel_edges(
                img_gray,
                subpixel_threshold,
                subpixel_iters,
                subpixel_order
            )

            edges_x = np.asarray(edges.x, dtype=float).flatten()
            edges_y = np.asarray(edges.y, dtype=float).flatten()

            print(f"{dosya_adi} -> Subpixel edge nokta sayısı: {len(edges_x)}")

            if len(edges_x) <= 3:
                print(f"{dosya_adi} -> Yetersiz kenar verisi")
                continue

            for outlier_name, outlier_func in outlier_methods.items():

                try:
                    x_filt, y_filt = outlier_func(edges_x, edges_y)
                    x_filt = np.asarray(x_filt, dtype=float)
                    y_filt = np.asarray(y_filt, dtype=float)
                except Exception as e:
                    print(f"{dosya_adi} ({outlier_name}) -> Outlier removal hatası: {e}")
                    continue

                if len(x_filt) <= 3:
                    continue

                row = {
                    'Dosya': dosya_adi,
                    'True Diameter (mm)': true_mm,
                    'Outlier Removal': outlier_name
                }

                for name, func in fitting_methods:
                    start = time.time()
                    key = (outlier_name, name)

                    try:
                        a, b, R = func(x_filt, y_filt)
                        diameter_px = 2 * R

                        # resim1 üzerinden her yöntem için mm/px katsayısı
                        if i == 1:
                            if np.isfinite(diameter_px) and diameter_px != 0:
                                scale_dict[key] = true_mm / diameter_px
                            else:
                                scale_dict[key] = np.nan

                        mm_per_px = scale_dict.get(key, np.nan)
                        measured_mm = diameter_px * mm_per_px
                        error_mm = measured_mm - true_mm
                        abs_error_mm = abs(error_mm)

                    except Exception as e:
                        diameter_px = np.nan
                        mm_per_px = scale_dict.get(key, np.nan)
                        measured_mm = np.nan
                        error_mm = np.nan
                        abs_error_mm = np.nan
                        print(f"{dosya_adi} ({outlier_name}) -> {name} hatası: {e}")

                    duration = time.time() - start

                    row[f"{name} Diameter (px)"] = diameter_px
                    row[f"{name} mm_per_px"] = mm_per_px
                    row[f"{name} Measured (mm)"] = measured_mm
                    row[f"{name} Error (mm)"] = error_mm
                    row[f"{name} Abs Error (mm)"] = abs_error_mm
                    row[f"{name} Time (s)"] = duration

                sonuc_listesi.append(row)

        except Exception as e:
            print(f"{dosya_adi} -> Hata: {e}")

    df = pd.DataFrame(sonuc_listesi)

    summary_rows = []
    if not df.empty:
        for outlier_name in df['Outlier Removal'].dropna().unique():
            sub = df[df['Outlier Removal'] == outlier_name]
            row = {'Outlier Removal': outlier_name}
            for name, _ in fitting_methods:
                col = f"{name} Abs Error (mm)"
                row[f"{name} MAE (mm)"] = sub[col].mean()
                row[f"{name} STD Abs Error (mm)"] = sub[col].std()
            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    output_excel = "cap_karsilastirma_subpixel_edges_mm_error.xlsx"

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Results", index=False)
        df_summary.to_excel(writer, sheet_name="MAE Summary", index=False)

    print(f"Excel dosyası oluşturuldu: {output_excel}")

    print("\n--- resim1 tabanlı mm/px katsayıları ---")
    for key, val in scale_dict.items():
        print(f"{key[0]} + {key[1]} -> {val:.10f} mm/px")

if __name__ == "__main__":
    main()
