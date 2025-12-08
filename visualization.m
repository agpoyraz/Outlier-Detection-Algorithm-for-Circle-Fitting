% Example data
outlier_methods = {'Z-Score', 'MAD', 'DBSCAN', 'LOF', 'Percentile', 'Proposed', 'None'};
fitting_methods = {'Geometric LS', 'Pratt', 'Taubin', 'RANSAC', 'IRLS', 'Hyper LS', 'M-Estimator', 'LMedS', 'TLS', 'EDCircle'};

% Data matrix for Mean Absolute Error (rows for outlier methods, columns for fitting methods)
mae_data = [
    0.0068, 0.0068, 0.0068, 0.0119, 0.0069, 0.0068, 0.0055, 0.0104, 0.0892, 0.0068;
    0.0071, 0.0071, 0.0071, 0.0132, 0.0071, 0.0071, 0.0061, 0.0093, 0.0906, 0.0071;
    0.0065, 0.0065, 0.0065, 0.0116, 0.0066, 0.0065, 0.0055, 0.0103, 0.0895, 0.0065;
    0.0065, 0.0065, 0.0065, 0.0112, 0.0065, 0.0065, 0.0055, 0.0093, 0.0895, 0.0065;
    0.0067, 0.0067, 0.0068, 0.0147, 0.0068, 0.0068, 0.0055, 0.0101, 0.0898, 0.0068;
    0.0053, 0.0053, 0.0053, 0.0072, 0.0053, 0.0053, 0.0054, 0.0088, 0.0910, 0.0053;
    0.0065, 0.0065, 0.0065, 0.0102, 0.0066, 0.0065, 0.0055, 0.0100, 0.0895, 0.0065;
];

% Define markers for each outlier removal method
markers = {'o', 'x', 's', '^', 'd', 'p', 'h'};

% Plot each outlier removal method
figure;
hold on;

for i = 1:length(outlier_methods)
    % Plotting each outlier method across different fitting algorithms with different markers
    plot(1:length(fitting_methods), mae_data(i, :), 'Marker', markers{i}, 'LineStyle', '-', 'DisplayName', outlier_methods{i}, 'LineWidth', 1);
end

% Set the y-axis to logarithmic scale
set(gca, 'YScale', 'log');

% Adding title and labels
title('Logarithmic Scale of Mean Absolute Error for Circle Fitting Algorithms');
xlabel('Fitting Algorithms');
ylabel('Mean Absolute Error (Log Scale)');

% Set x-ticks and labels
xticks(1:length(fitting_methods));
xticklabels(fitting_methods);
xtickangle(45);

% Display the legend
legend('show');

% Display grid
grid on;

hold off;