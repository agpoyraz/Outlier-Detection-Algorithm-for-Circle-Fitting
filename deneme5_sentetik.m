%{ Outlier removal algoritması için elips bir görüntü oluşturuyoruz. bu
%görüntünün noktaları mevcut. Noktalara gauss gürültüsü sigma, çembere
%yakın yerlerdeki outliarlar near_ellips_outliers, küme halinde outlierlar
%cluster_outlier eklenebiliyor. outlier sayısı ise cluster için adet, near
%için ise yüzdesel olarak oluşturuluyor. Algoritma içerisinde ek olarak bir
%de cluster removal algoritması oluşturulmuştur. Bu da henüz algoritma
%çalıştırılmadan önce clusterları tespit etmektedir. Sonrasında outlier
%removal algoritması çalıştırılmaktadır. Bu çalışmada iteratif yaklaşım
%denenecektir. deneme4 ten farklı olarak cluster outlier farklı noktalarda
%oluşabiliyor.
%}


%% synthetic_ellipse_outlier_demo.m
clc;
clear;
close all;

%% Parametreler: Ellipse Creation
xc = 500;
yc = 500;
a = 688;
b = 690;
n_points = 3600;
sigma = 0.5;
cluster_ratio = 0.01; % Yüzdelik olarak nokta sayısını belirliyoruz.
near_ratio = 0.01; % Yüzdelik olarak nokta sayısını belirliyoruz.
cluster_outliers = round(n_points * cluster_ratio);
near_ellipse_outliers = round(n_points * near_ratio);
random_outliers = 0;
random_seed = 42; %% hep aynı üretmesin istiyorsan random_seed = []; kullan

%% Parametreler: Methods
cluster_removal_mode = 0; % 1: aktif, 0: pasif,  cluster removal algoritmasının kullanılıp kullanılmayacağı 

x_cluster_removed = [];
y_cluster_removed = [];

debug_mode = 0; % 0 kapalı, video gibi bir plot oluşturmak için 

threshold = 3;
window_size = 50;
std_window = 50;
num_iterations = 3; % kaç kez çalıştırılacak, otomatik olarak removal_ratio_threshold kadar siliyorsa duracak.
removal_ratio_threshold = 0.05; % %10

%% Sentetik elips verisi üret
[x_in, y_in, ...
 x_out_cluster, y_out_cluster, ...
 x_out_near, y_out_near, ...
 x_out_random, y_out_random, ...
 X, Y, labels] = generate_synthetic_ellipse( ...
    xc, yc, a, b, n_points, sigma, ...
    cluster_outliers, near_ellipse_outliers, random_outliers, random_seed);



x = X(:);
y = Y(:);
%% Histogram Tabanlı Cluster Type Outer Detection Part
figure;
histogram2(x, y, [30 30], 'DisplayStyle', 'tile', 'ShowEmptyBins', 'on');
xlabel('x');
ylabel('y');
title('2D Histogram / Heatmap');
colorbar
axis tight

% 1. Histogram hesapla
[N, edgesX, edgesY] = histcounts2(x, y, [30 30]);

% 2. Median (non-zero)
vals = N(:);
vals_nonzero = vals(vals > 0);
med_val = median(vals_nonzero);

% 3. Threshold maskesi (yoğun bin'ler)
mask = N > 2 * med_val;

% 4. Bin indexlerini bul
[row_idx, col_idx] = find(mask);

% 5. Bu bin'lere düşen noktaları bul
selected_idx = false(size(x));

for k = 1:length(row_idx)
    i = row_idx(k);
    j = col_idx(k);

    x_min = edgesX(i);
    x_max = edgesX(i+1);

    y_min = edgesY(j);
    y_max = edgesY(j+1);

    idx = (x >= x_min & x < x_max) & ...
          (y >= y_min & y < y_max);

    selected_idx = selected_idx | idx;
end

% 6. Görselleştir
figure;
hold on; grid on; axis equal;

plot(x, y, 'b.', 'DisplayName', 'Tüm Noktalar');
plot(x(selected_idx), y(selected_idx), 'ro', 'MarkerSize', 6, ...
     'DisplayName', 'Yoğun Bölgeler');

legend;
title('Median*2 Üstü Yoğunlukta Olan Noktalar');
xlabel('x'); ylabel('y');

%% Noktalar Arası Mesafe Tabanlı Cluster Type Outer Detection Part

if cluster_removal_mode == 1

    % x ve y sütun vektörü olsun
    x = x(:);
    y = y(:);

    n = length(x);

    % Koordinat matrisi
    P = [x y];

    % ---- Öklid mesafesi ----
    xx = sum(P.^2, 2);
    D2 = xx + xx' - 2*(P*P');
    D2(D2 < 0) = 0;
    D = sqrt(D2);

    % diagonal'i ignore et
    D(1:n+1:end) = inf;

    % ---- Sıralama ----
    D_sorted = sort(D, 2);

    % ---- İlk 10 komşu toplamı ----
    k = 10;
    sum10 = sum(D_sorted(:, 1:k), 2);

    % ---- Median ----
    ttt = median(sum10);

    % ---- Outlier tespiti ----
    idx_flag = (sum10 < (ttt/4)) | (sum10 > (ttt*2));

    % 🔴 ASIL EKLENEN KISIM: VERİYİ GERÇEKTEN TEMİZLE
    x_before_cluster = x;
    y_before_cluster = y;
    
    x_cluster_removed = x(idx_flag);
    y_cluster_removed = y(idx_flag);
    
    x = x(~idx_flag);
    y = y(~idx_flag);

    % ---- Görselleştirme ----
    figure;
    hold on; grid on; axis equal;
    
    plot(x, y, 'b.', 'DisplayName', 'Kalan Noktalar');
    
    if exist('x_cluster_removed', 'var') && ~isempty(x_cluster_removed)
        plot(x_cluster_removed, y_cluster_removed, 'r.', ...
            'MarkerSize', 14, ...
            'DisplayName', 'Cluster Removal ile Silinen Noktalar');
    end
    
    legend;
    title('Cluster Removal Sonrası Veri');
    xlabel('x');
    ylabel('y');
    hold off;

else
    disp('Cluster removal mode kapali - KNN kismi atlandi');
end

%% Orijinal veriyi çiz
figure('Name','Sentetik Elips Verisi');
hold on;
grid on;
axis equal;

if ~isempty(x_in)
    plot(x_in, y_in, 'b.', 'DisplayName', 'Inlier');
end

if ~isempty(x_out_cluster)
    plot(x_out_cluster, y_out_cluster, 'r.', 'MarkerSize', 14, 'DisplayName', 'Cluster Outlier');
end

if ~isempty(x_out_near)
    plot(x_out_near, y_out_near, 'mo', 'MarkerSize', 6, 'DisplayName', 'Near-Ellipse Outlier');
end

if ~isempty(x_out_random)
    plot(x_out_random, y_out_random, 'go', 'MarkerSize', 6, 'DisplayName', 'Random Outlier');
end

tt = linspace(0, 2*pi, 400);
x_true = xc + a*cos(tt);
y_true = yc + b*sin(tt);
plot(x_true, y_true, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Elips');

xlabel('x');
ylabel('y');
title('Sentetik Elips Verisi');
legend('Location','best');
hold off;

%% Önerilen yöntem ile outlier temizleme
x_iter = x;
y_iter = y;

for iter = 1:num_iterations

    prev_n = length(x_iter);

    [x_new, y_new, removed_ratio] = remove_outliers_local_zscore_proposed( ...
        x_iter, y_iter, threshold, window_size, std_window, ...
        x_cluster_removed, y_cluster_removed, debug_mode, iter);

    fprintf('Iter %d -> removed ratio: %.3f\n', iter, removed_ratio);

    % 🚨 Eğer çok agresif silme varsa bu iterasyonu KABUL ETME
    if removed_ratio > removal_ratio_threshold
        fprintf('Iter %d ignored (too many points removed)\n', iter);
        break;
    end

    % ✔ kabul edilen iterasyon
    x_iter = x_new;
    y_iter = y_new;

end

x_filt = x_iter;
y_filt = y_iter;

%% Bilgi yazdır
fprintf('Inlier sayisi               : %d\n', length(x_in));
fprintf('Cluster outlier sayisi      : %d\n', length(x_out_cluster));
fprintf('Near-ellipse outlier sayisi : %d\n', length(x_out_near));
fprintf('Random outlier sayisi       : %d\n', length(x_out_random));
fprintf('Toplam nokta sayisi         : %d\n', length(X));
fprintf('Filtre sonrasi nokta sayisi : %d\n', length(x_filt));
fprintf('Silinen nokta sayisi        : %d\n', length(X) - length(x_filt));

%% -------- Local functions --------

function [x_filt, y_filt, removed_ratio] = remove_outliers_local_zscore_proposed(x, y, threshold, window_size, std_window, x_cluster_removed, y_cluster_removed, debug_mode, iter_id)
% Polar koordinatlarda local z-score benzeri yöntemle outlier temizleme

    if nargin < 3, threshold = 3; end
    if nargin < 4, window_size = 60; end
    if nargin < 5, std_window = 60; end
    if nargin < 6, x_cluster_removed = []; end
    if nargin < 7, y_cluster_removed = []; end
    if nargin < 8, debug_mode = 0; end
    if nargin < 9, iter_id = 1; end

    x = x(:);
    y = y(:);

    % 1. Centering and polar transformation
    xc = mean(x);
    yc = mean(y);

    theta = atan2(y - yc, x - xc);
    r = sqrt((x - xc).^2 + (y - yc).^2);

    % 2. Sort by angle
    [theta_sorted, idx] = sort(theta);
    r_sorted = r(idx);
    x_sorted = x(idx);
    y_sorted = y(idx);

    % 3. Local std calculation
    std_list = [];
    stride = 20;
    last_start = length(r_sorted) - std_window + 1;

    if last_start >= 1
        for i = 1:stride:last_start
            std_list(end+1,1) = std(r_sorted(i:i+std_window-1)); %#ok<AGROW>
        end
        global_std = median(std_list);
    else
        global_std = std(r_sorted);
    end

    if global_std < 1e-12
        global_std = global_std + 1e-12;
    end

    % 4. Outlier removal
    n = length(r_sorted);
    mask = true(n,1);

    if n < window_size
        mean_r = mean(r_sorted);
        outliers = abs(r_sorted - mean_r) > threshold * global_std;
        mask = ~outliers;
    else
        figure('Name','Polar Debug View');
        h_orig = plot(theta_sorted, r_sorted, 'r.', 'DisplayName','Outlier');
        hold on;
        h_clean = plot(nan, nan, 'b.', 'DisplayName','Original');
        h_win = plot(nan, nan, 'go', 'MarkerSize',6, 'DisplayName','Window');
        
        xlabel('\theta (radian)');
        ylabel('r');
        legend;
        grid on;
        for i = 1:(n - window_size + 1)
            window_idx = i:(i+window_size-1);
            window = r_sorted(window_idx);
        
            mean_r = mean(window);
            outliers = abs(window - mean_r) > threshold * global_std;
        
            % mask güncelle
            mask(window_idx) = mask(window_idx) & ~outliers;
        
            % temizlenmiş veri
            r_clean = r_sorted(mask);
            theta_clean = theta_sorted(mask);
        
            % window noktaları
            theta_win = theta_sorted(window_idx);
            r_win = r_sorted(window_idx);
        
            % --- grafik güncelle ---
            set(h_clean, 'XData', theta_clean, 'YData', r_clean);
            set(h_win, 'XData', theta_win, 'YData', r_win);
        
            title(sprintf('Step %d / %d', i, n-window_size+1));
        
            if i==100
                ss=58;
            end
            if debug_mode == 1
                drawnow;
                pause(0.02);
            end
        end
        hold off
    end

    % 5. Filtered coordinates
    r_clean = r_sorted(mask);
    theta_clean = theta_sorted(mask);

    x_filt = r_clean .* cos(theta_clean) + xc;
    y_filt = r_clean .* sin(theta_clean) + yc;

    % Proposed method ile bu iterasyonda silinen noktalar
    theta_removed = theta_sorted(~mask);
    r_removed     = r_sorted(~mask);
    x_removed     = x_sorted(~mask);
    y_removed     = y_sorted(~mask);

    % --- Grafik 1: Polar koordinatlarda ---
    figure('Name','Outlier Removal in Polar Coordinates');
    hold on;
    
    % Cluster removal ile tespit edilip ana veriden çıkarılan noktalar
    % kırmızı dolu nokta olarak gösterilir.
    if iter_id == 1 && ~isempty(x_cluster_removed)
        theta_cluster_removed = atan2(y_cluster_removed - yc, ...
                                      x_cluster_removed - xc);
    
        r_cluster_removed = sqrt((x_cluster_removed - xc).^2 + ...
                                 (y_cluster_removed - yc).^2);
    
        plot(theta_cluster_removed, r_cluster_removed, 'm.', ...
            'MarkerSize', 8, ...
            'DisplayName', 'Removed by Cluster Method');
    end

    % Proposed method ile bu iterasyonda silinen noktalar
    if ~isempty(theta_removed)
        plot(theta_removed, r_removed, 'r.', ...
            'MarkerSize', 8, ...
            'DisplayName', 'Removed by Proposed Method');
    end
    
    % Final kalan veri
    plot(theta_clean, r_clean, 'b.', ...
        'MarkerSize', 8, ...
        'DisplayName', 'Filtered');
    
    xlabel('\theta (radian)');
    ylabel('r');
    title(sprintf('Outlier Removal (Polar) - Iteration %d', iter_id));
    legend('Location','best');
    grid on;
    hold off;

    % --- Grafik 2: Kartezyen koordinatlarda ---
    figure('Name','Outlier Removal in Cartesian Coordinates');
    hold on;
    
    % Cluster removal ile tespit edilen noktalar kırmızı dolu nokta
    if iter_id == 1 && ~isempty(x_cluster_removed)
        plot(x_cluster_removed, y_cluster_removed, 'm.', ...
            'MarkerSize', 8, ...
            'DisplayName', 'Removed by Cluster Method');
    end

    % Proposed method ile bu iterasyonda silinen noktalar kırmızı dolu nokta
    if ~isempty(x_removed)
        plot(x_removed, y_removed, 'r.', ...
            'MarkerSize', 8, ...
            'DisplayName', 'Removed by Proposed Method');
    end
    
    % Final kalan veri mavi
    plot(x_filt, y_filt, 'b.', ...
        'MarkerSize', 8, ...
        'DisplayName', 'Filtered');
    
    xlabel('x');
    ylabel('y');
    title(sprintf('Outlier Removal (Cartesian) - Iteration %d', iter_id));
    axis equal;
    legend('Location','best');
    grid on;
    hold off;

    removed_ratio = 1 - (length(x_filt) / length(x));
end

function [x_in, y_in, ...
          x_out_cluster, y_out_cluster, ...
          x_out_near, y_out_near, ...
          x_out_random, y_out_random, ...
          X, Y, labels] = generate_synthetic_ellipse( ...
          xc, yc, a, b, n_points, sigma, ...
          cluster_outliers, near_ellipse_outliers, random_outliers, random_seed)
% Sentetik elips verisi üretir

    if nargin >= 10 && ~isempty(random_seed)
        rng(random_seed);
    end

    % 1) İdeal elips noktaları
    theta = linspace(0, 2*pi, n_points + 1);
    theta(end) = [];

    x = xc + a*cos(theta);
    y = yc + b*sin(theta);

    % 2) Gaussian noise eklenmiş inlier noktaları
    x_in = x(:) + sigma * randn(length(x), 1);
    y_in = y(:) + sigma * randn(length(y), 1);

    % 3) Cluster outlier
    if cluster_outliers > 0
        theta0 = 2*pi*rand();   % 🔥 rastgele yön
        offset = 5;             % çembere uzaklık (SABİT)
    
        % cluster merkezi (elipsin biraz dışı)
        x_center = xc + (a + offset) * cos(theta0);
        y_center = yc + (b + offset) * sin(theta0);
    
        % cluster noktaları (küme şeklinde)
        sigma_cluster = 1;
        x_out_cluster = x_center + sigma_cluster * randn(cluster_outliers, 1);
        y_out_cluster = y_center + sigma_cluster * randn(cluster_outliers, 1);
    else
        x_out_cluster = [];
        y_out_cluster = [];
    end

    % 4) Near-ellipse outlier
    if near_ellipse_outliers > 0
        theta_o = 2*pi*rand(near_ellipse_outliers, 1);
        scale = 1 + 0.01*randn(near_ellipse_outliers, 1);

        x_out_near = xc + scale .* a .* cos(theta_o);
        y_out_near = yc + scale .* b .* sin(theta_o);
    else
        x_out_near = [];
        y_out_near = [];
    end

    % 5) Random outlier
    if random_outliers > 0
        span = 1.5 * max(a, b) * 2;
        x_out_random = (xc - span) + (2 * span) * rand(random_outliers, 1);
        y_out_random = (yc - span) + (2 * span) * rand(random_outliers, 1);
    else
        x_out_random = [];
        y_out_random = [];
    end

    % 6) Tüm outlierları birleştir
    x_out = [x_out_cluster; x_out_near; x_out_random];
    y_out = [y_out_cluster; y_out_near; y_out_random];

    % 7) Tüm noktaları birleştir
    X = [x_in; x_out];
    Y = [y_in; y_out];

    labels = [ ...
        zeros(length(x_in), 1); ...
        ones(length(x_out_cluster), 1); ...
        2 * ones(length(x_out_near), 1); ...
        3 * ones(length(x_out_random), 1)];

    % 8) Karıştır
    idx = randperm(length(X));
    X = X(idx);
    Y = Y(idx);
    labels = labels(idx);
end