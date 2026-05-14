%% real_image_proposed_outlier_demo.m
clc;
clear;
close all;

addpath(genpath('.'));

%% Görüntü oku
image_path = 'C:\Users\gokhanpoyraz\Desktop\Çalışmalar\Circle_Fitting_Comparison\Circle-Fitting\Images_Outer\resim36.png';   % BURAYI DEĞİŞTİR
image = imread(image_path);

if size(image,3) == 3
    image_gray = rgb2gray(image);
else
    image_gray = image;
end

figure;
imshow(image_gray, []);
title('Orijinal Görüntü');

%% Subpixel edge extraction
threshold_edge = 15;
edges = subpixelEdges(image_gray, threshold_edge, 'SmoothingIter', 1);

x_original = edges.x(:);
y_original = edges.y(:);
visEdges(edges)

fprintf('Başlangıç nokta sayısı: %d\n', length(x_original));

figure;
plot(x_original, y_original, 'b.');
axis equal;
grid on;
xlabel('x');
ylabel('y');
title('Subpixel Edge Points');

%% Parametreler
cluster_removal_mode = 1;   % 1: aktif, 0: pasif

threshold = 3;
window_size = 50;
std_window = 50;
num_iterations = 6;
removal_ratio_threshold = 0.1;
debug_mode = 0;

x = x_original;
y = y_original;

x_cluster_removed = [];
y_cluster_removed = [];

%% Cluster removal
if cluster_removal_mode == 1

    x = x(:);
    y = y(:);

    n = length(x);
    P = [x y];

    xx = sum(P.^2, 2);
    D2 = xx + xx' - 2*(P*P');
    D2(D2 < 0) = 0;
    D = sqrt(D2);

    D(1:n+1:end) = inf;

    D_sorted = sort(D, 2);

    k = 10;
    sum10 = sum(D_sorted(:, 1:k), 2);

    ttt = median(sum10);

    idx_flag = (sum10 < (ttt/4)) | (sum10 > (ttt*1.5));

    x_cluster_removed = x(idx_flag);
    y_cluster_removed = y(idx_flag);

    x = x(~idx_flag);
    y = y(~idx_flag);

    figure;
    hold on;
    grid on;
    axis equal;

    plot(x, y, 'b.', 'DisplayName', 'Kalan Noktalar');

    if ~isempty(x_cluster_removed)
        plot(x_cluster_removed, y_cluster_removed, 'm.', ...
            'MarkerSize', 10, ...
            'DisplayName', 'Cluster Removal ile Silinen Noktalar');
    end

    xlabel('x');
    ylabel('y');
    title('Cluster Removal Sonrası Veri');
    legend('Location','best');
    hold off;

else
    disp('Cluster removal mode kapali - KNN kismi atlandi');
end

%% Proposed yöntem
x_iter = x;
y_iter = y;

for iter = 1:num_iterations

    [x_new, y_new, removed_ratio] = remove_outliers_local_zscore_proposed( ...
        x_iter, y_iter, threshold, window_size, std_window, ...
        x_cluster_removed, y_cluster_removed, debug_mode, iter);

    fprintf('Iter %d -> removed ratio: %.3f\n', iter, removed_ratio);

    if removed_ratio > removal_ratio_threshold
        fprintf('Iter %d exceeded threshold -> stopping iterations\n', iter);
    
        % İlk iterasyonda fazla silme varsa:
        % sonucu kabul et ama devam etme.
        if iter == 1
            x_iter = x_new;
            y_iter = y_new;
        end
    
        break;
    end
    
    % Normal durumda iterasyonu kabul et
    x_iter = x_new;
    y_iter = y_new;
end

x_filt = x_iter;
y_filt = y_iter;

%% Final karşılaştırma
figure;
hold on;
grid on;
axis equal;

plot(x_original, y_original, '.', ...
    'Color', [0.7 0.7 0.7], ...
    'DisplayName', 'Başlangıç Noktaları');

if ~isempty(x_cluster_removed)
    plot(x_cluster_removed, y_cluster_removed, 'm.', ...
        'MarkerSize', 10, ...
        'DisplayName', 'Cluster Removal ile Silinen');
end

plot(x_filt, y_filt, 'b.', ...
    'MarkerSize', 8, ...
    'DisplayName', 'Final Kalan Noktalar');

xlabel('x');
ylabel('y');
title('Gerçek Görüntü Üzerinde Proposed Outlier Removal Sonucu');
legend('Location','best');
hold off;

fprintf('\n--- Özet ---\n');
fprintf('Başlangıç nokta sayısı          : %d\n', length(x_original));
fprintf('Cluster removal sonrası nokta   : %d\n', length(x));
fprintf('Final filtre sonrası nokta      : %d\n', length(x_filt));
fprintf('Toplam silinen nokta            : %d\n', length(x_original) - length(x_filt));

%% -------- Local Function --------

function [x_filt, y_filt, removed_ratio] = remove_outliers_local_zscore_proposed( ...
    x, y, threshold, window_size, std_window, ...
    x_cluster_removed, y_cluster_removed, debug_mode, iter_id)

    if nargin < 3, threshold = 3; end
    if nargin < 4, window_size = 50; end
    if nargin < 5, std_window = 50; end
    if nargin < 6, x_cluster_removed = []; end
    if nargin < 7, y_cluster_removed = []; end
    if nargin < 8, debug_mode = 0; end
    if nargin < 9, iter_id = 1; end

    x = x(:);
    y = y(:);

    xc = mean(x);
    yc = mean(y);

    theta = atan2(y - yc, x - xc);
    r = sqrt((x - xc).^2 + (y - yc).^2);

    [theta_sorted, idx] = sort(theta);
    r_sorted = r(idx);
    x_sorted = x(idx);
    y_sorted = y(idx);

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

    if ~isfinite(global_std) || global_std < 1e-12
        global_std = 1e-12;
    end

    n = length(r_sorted);
    mask = true(n,1);

    if n < window_size
        mean_r = mean(r_sorted);
        outliers = abs(r_sorted - mean_r) > threshold * global_std;
        mask = ~outliers;
    else
        if debug_mode == 1
            figure('Name','Polar Debug View');
            h_clean = plot(nan, nan, 'b.', 'DisplayName','Kalan');
            hold on;
            h_win = plot(nan, nan, 'go', 'MarkerSize',6, 'DisplayName','Window');
            plot(theta_sorted, r_sorted, 'r.', 'DisplayName','Tüm Noktalar');
            xlabel('\theta (radian)');
            ylabel('r');
            legend;
            grid on;
        end

        for i = 1:(n - window_size + 1)
            window_idx = i:(i+window_size-1);
            window = r_sorted(window_idx);

            mean_r = mean(window);
            outliers = abs(window - mean_r) > threshold * global_std;

            mask(window_idx) = mask(window_idx) & ~outliers;

            if debug_mode == 1
                set(h_clean, 'XData', theta_sorted(mask), 'YData', r_sorted(mask));
                set(h_win, 'XData', theta_sorted(window_idx), 'YData', r_sorted(window_idx));
                title(sprintf('Step %d / %d', i, n-window_size+1));
                drawnow;
                pause(0.02);
            end
        end
    end

    r_clean = r_sorted(mask);
    theta_clean = theta_sorted(mask);

    x_filt = r_clean .* cos(theta_clean) + xc;
    y_filt = r_clean .* sin(theta_clean) + yc;

    theta_removed = theta_sorted(~mask);
    r_removed = r_sorted(~mask);
    x_removed = x_sorted(~mask);
    y_removed = y_sorted(~mask);

    %% Polar plot
    figure('Name','Outlier Removal in Polar Coordinates');
    hold on;

    if iter_id == 1 && ~isempty(x_cluster_removed)
        theta_cluster_removed = atan2(y_cluster_removed - yc, x_cluster_removed - xc);
        r_cluster_removed = sqrt((x_cluster_removed - xc).^2 + ...
                                 (y_cluster_removed - yc).^2);

        plot(theta_cluster_removed, r_cluster_removed, 'm.', ...
            'MarkerSize', 8, ...
            'DisplayName', 'Removed by Cluster Method');
    end

    if ~isempty(theta_removed)
        plot(theta_removed, r_removed, 'r.', ...
            'MarkerSize', 8, ...
            'DisplayName', 'Removed by Proposed Method');
    end

    plot(theta_clean, r_clean, 'b.', ...
        'MarkerSize', 8, ...
        'DisplayName', 'Filtered');

    xlabel('\theta (radian)');
    ylabel('r');
    title(sprintf('Outlier Removal Polar - Iteration %d', iter_id));
    legend('Location','best');
    grid on;
    hold off;

    %% Cartesian plot
    figure('Name','Outlier Removal in Cartesian Coordinates');
    hold on;

    if iter_id == 1 && ~isempty(x_cluster_removed)
        plot(x_cluster_removed, y_cluster_removed, 'm.', ...
            'MarkerSize', 8, ...
            'DisplayName', 'Removed by Cluster Method');
    end

    if ~isempty(x_removed)
        plot(x_removed, y_removed, 'r.', ...
            'MarkerSize', 8, ...
            'DisplayName', 'Removed by Proposed Method');
    end

    plot(x_filt, y_filt, 'b.', ...
        'MarkerSize', 8, ...
        'DisplayName', 'Filtered');

    xlabel('x');
    ylabel('y');
    title(sprintf('Outlier Removal Cartesian - Iteration %d', iter_id));
    axis equal;
    legend('Location','best');
    grid on;
    hold off;

    removed_ratio = 1 - (length(x_filt) / length(x));
end