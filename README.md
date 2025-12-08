# Outlier Detection Algorithm for Circle Fitting

This repository presents a robust outlier detection method designed for circle fitting applications in machine vision and industrial metrology. The algorithm operates in polar coordinates and identifies outliers using both local and global statistical analysis. After removing noisy points, the cleaned dataset can be used with any circle fitting method.

## Features
- Polar coordinate–based transformation
- Local and global standard deviation comparison
- Automatic outlier detection and removal
- Compatible with sub-pixel edge points
- Works with all circle fitting algorithms (LS, Pratt, Taubin, RANSAC, IRLS, etc.)

## Workflow

1- Convert (x, y) edge points to polar coordinates
2- Compute global and local standard deviations
3- Compare local mean values with global deviation
4- Remove outliers
5- Fit the circle using the preferred method

## Applications
- Industrial diameter measurement
- Sub-pixel edge processing
- Washer and circular part inspection
- Robust geometric fitting under noise
- Machine vision quality control systems

## Citation

If you use this method in your academic work, please cite:
**Plain text:**
Poyraz, A. G. (2025). *Outlier Detection Algorithm for Circle Fitting*. GitHub repository. https://github.com/agpoyraz/OUTLIER-DETECTION-ALGORITHM-FOR-CIRCLE-FITTING
