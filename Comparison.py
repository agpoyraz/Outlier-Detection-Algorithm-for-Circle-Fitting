import matplotlib.pyplot as plt 
from PIL import Image
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


from devernay_edges import DevernayEdges

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
    x = np.array(x)
    y = np.array(y)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    M = np.column_stack((A, b))
    _, _, Vt = np.linalg.svd(M)
    v = Vt[-1, :]
    c = v[:3] / v[3]
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

def remove_outliers_local_zscore_proposed(x, y, threshold=3, window_size=60, std_window=60):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # 1. Centering and polar transformation
    xc = np.mean(x)
    yc = np.mean(y)

    theta = np.arctan2(y - yc, x - xc)
    r = np.sqrt((x - xc)**2 + (y - yc)**2)

    # 2. Sort by angle
    idx = np.argsort(theta)
    theta_sorted = theta[idx]
    r_sorted = r[idx]
    x_sorted = x[idx]
    y_sorted = y[idx]

    # 3. Local std calculation
    std_list = []
    stride = 20
    for i in range(0, len(r_sorted) - std_window + 1, stride):
        std_list.append(np.std(r_sorted[i:i+std_window]))
    global_std = np.median(std_list)

    # 4. Outlier removal
    n = len(r_sorted)
    mask = np.ones(n, dtype=bool)
    for i in range(n - window_size + 1):
        window = r_sorted[i:i+window_size]
        mean = np.mean(window)
        std = np.std(window)
        outliers = np.abs(window - mean) > threshold * global_std
        mask[i:i+window_size] &= ~outliers

    # 5. Filtered coordinates
    r_clean = r_sorted[mask]
    theta_clean = theta_sorted[mask]
    x_filt = r_clean * np.cos(theta_clean) + xc
    y_filt = r_clean * np.sin(theta_clean) + yc

    # # --- Grafik 1: Polar koordinatlarda ---
    # plt.figure(figsize=(8, 6))
    # plt.plot(theta_sorted, r_sorted, 'r.', label='Original')
    # plt.plot(theta_clean, r_clean, 'b.', label='Filtered')
    # plt.xlabel(r'$\theta$ (radian)')
    # plt.ylabel('r')
    # plt.title('Outlier Removal in Polar Coordinates')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # --- Grafik 2: Kartezyen koordinatlarda ---
    # plt.figure(figsize=(8, 6))
    # plt.plot(x_sorted, y_sorted, 'r.', label='Original')
    # plt.plot(x_filt, y_filt, 'b.', label='Filtered')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Outlier Removal in Cartesian Coordinates')
    # plt.axis('equal')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return x_filt, y_filt

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

# --- Ana işlem ---

def main():
    sigma = 1.0
    high_treshold = 15.0
    low_threshold = 5.0
    klasor = r"C:\\Users\\gokhanpoyraz\\Desktop\\Çalışmalar\\Circle_Fitting_Comparison\\Concentricity_Measurement-main\\Images_Outer"
    sonuc_listesi = []

    # Tüm fitting fonksiyonları bu scriptte tanımlı, dışarıdan import gereksiz

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

    # For each image
    for i in range(1, 3):
        dosya_adi = f"resim{i}.png"
        tam_yol = os.path.join(klasor, dosya_adi)
        try:
            image_binary = Image.open(tam_yol)
            devernayEdges = DevernayEdges(image_binary, sigma, high_treshold, low_threshold)
            edges_x, edges_y = devernayEdges.detect_edges()
    
            if len(edges_x) > 3:
                for outlier_name, outlier_func in outlier_methods.items():
                    try:
                        # Measure the time for the outlier removal
                        start_outlier_time = time.time()
                        x_filt, y_filt = outlier_func(edges_x, edges_y)
                        outlier_duration = time.time() - start_outlier_time
    
                        x_filt = np.array(x_filt)
                        y_filt = np.array(y_filt)
    
                        if len(x_filt) <= 3:
                            continue
    
                        row = {'Dosya': dosya_adi, 'Outlier Removal': outlier_name, 'Outlier Removal Time (s)': outlier_duration}
    
                        for name, func in fitting_methods:
                            start = time.time()
                            try:
                                r = 2 * func(x_filt, y_filt)[2]
                            except Exception as e:
                                r = None
                                print(f"{dosya_adi} ({outlier_name}) -> {name} hatası: {e}")
    
                            duration = time.time() - start
                            row[name] = r
                            row[f"{name} Time (s)"] = duration
    
                        sonuc_listesi.append(row)
                    except Exception as e:
                        print(f"{dosya_adi} ({outlier_name}) -> Outlier removal hatası: {e}")
                        continue
            else:
                print(f"{dosya_adi} -> Yetersiz kenar verisi")
        except Exception as e:
            print(f"{dosya_adi} -> Hata: {e}")

    df = pd.DataFrame(sonuc_listesi)
    df.to_excel("cap_karsilastirma_full_20.06.2025_th3_w60_60.xlsx", index=False)
    print("Excel dosyası oluşturuldu: cap_karsilastirma_full.xlsx")

if __name__ == "__main__":
    main()
