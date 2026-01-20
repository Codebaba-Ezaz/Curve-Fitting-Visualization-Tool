# Curve Fitting Visualization Tool

> A comprehensive Python application for fitting, analyzing, and visualizing mathematical models to experimental data.

![GUI Preview](https://github.com/user-attachments/assets/957fe91c-7de0-4b71-90f6-592751e96bf1)

## 📋 Overview

The **Curve Fitting Visualization Tool** is a powerful desktop application that enables users to:
- Load and preprocess experimental data
- Fit multiple mathematical models (Linear, Polynomial, Power-law, Exponential)
- Visualize fits and compare models
- Calculate detailed error metrics and residual analysis
- Export results as LaTeX equations
- Compare multiple fitting models simultaneously

Perfect for researchers, engineers, and students working with numerical methods and data analysis.

---

## ✨ Features

### 📊 Multiple Fitting Methods
- **Linear Regression** - Standard least squares fitting
- **Polynomial Fitting** - Up to N-th degree polynomials
- **Power-law Model** - $y = ax^b$ form
- **Exponential Model** - $y = ae^{bx}$ form

### 📈 Advanced Analysis
- **Error Metrics**: R², RMSE, MAE, MSE, and more
- **Residual Analysis**: Identify outliers and model fit quality
- **Model Comparison**: Side-by-side comparison of different models
- **Interactive Visualization**: Zoom, pan, and navigate plots

### 💾 Data Management
- Import CSV files with multiple columns
- Built-in sample datasets (linear, polynomial, power-law, exponential)
- Data preview and validation
- Flexible column selection

### 📤 Export Options
- **LaTeX Equations** - Export fitted equations in LaTeX format
- **Graph Export** - Save plots as images
- **Report Generation** - Comprehensive analysis reports

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/curve-fitting-visualization-tool.git
   cd curve-fitting-visualization-tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21.0 | Numerical computations |
| `scipy` | ≥1.7.0 | Scientific computing & optimization |
| `matplotlib` | ≥3.4.0 | Data visualization |
| `pandas` | ≥1.3.0 | Data handling & manipulation |
| `Pillow` | ≥8.3.0 | Image processing |

See [requirements.txt](requirements.txt) for exact version specifications.

---

## 📚 Usage Guide

### Loading Data

1. **Open CSV File**: Click "Load Data" to import your CSV file
2. **Select Columns**: Choose X and Y data columns from your dataset
3. **View Data**: Preview your data in the data table view

### Fitting a Model

1. **Choose Fitting Method**: Select from Linear, Polynomial, Power-law, or Exponential
2. **Set Parameters**: 
   - For polynomial: specify the degree
   - For others: default parameters are pre-configured
3. **Fit Model**: Click "Fit" to compute the model
4. **View Results**: See the fitted equation and statistical metrics

### Comparing Models

1. **Add Models**: Fit different models and add them to comparison
2. **Visual Comparison**: View all models on the same plot
3. **Compare Metrics**: Analyze R², RMSE, and other error metrics side-by-side
4. **Select Best Fit**: Identify the model with the best fit quality

### Analysis & Export

1. **View Residuals**: Analyze residual plots to assess fit quality
2. **Export Equation**: Save the fitted equation as LaTeX
3. **Save Plot**: Export visualization as PNG/PDF
4. **Generate Report**: Create a comprehensive analysis report

---

## 📁 Project Structure

```
curve-fitting-visualization-tool/
├── main.py                  # Main application & GUI
├── fitting_methods.py       # Fitting algorithms & model classes
├── data_handler.py          # Data loading & preprocessing
├── error_metrics.py         # Error calculations & residual analysis
├── visualization.py         # Plotting & visualization utilities
├── requirements.txt         # Python dependencies
├── sample_data/             # Example datasets
│   ├── linear_data.csv
│   ├── polynomial_data.csv
│   ├── power_law_data.csv
│   └── exponential_data.csv
└── Readme.md               # This file
```

---

## 🔧 Supported Models

### Linear Model
$$y = mx + c$$

### Polynomial Model
$$y = a_0 + a_1x + a_2x^2 + \cdots + a_nx^n$$

### Power-law Model
$$y = ax^b$$

### Exponential Model
$$y = ae^{bx}$$

---

## 📊 Sample Datasets

The application includes built-in sample datasets for testing:

- **linear_data.csv** - Linear relationship data
- **polynomial_data.csv** - Polynomial (degree 2-3) data
- **power_law_data.csv** - Power-law relationship data
- **exponential_data.csv** - Exponential growth/decay data

Load these from the "Load Sample Data" menu to quickly explore the tool's capabilities.

---

## 🎯 Error Metrics Explained

- **R² (Coefficient of Determination)**: 0-1 scale, closer to 1 is better
- **RMSE (Root Mean Squared Error)**: Average magnitude of prediction errors
- **MAE (Mean Absolute Error)**: Average absolute prediction errors
- **MSE (Mean Squared Error)**: Average squared prediction errors
- **Residuals**: Differences between observed and predicted values

---

## 💡 Tips & Best Practices

1. **Data Quality**: Ensure your data is clean and free of outliers before fitting
2. **Choose Appropriate Model**: Select a model that matches your data pattern
3. **Validate Results**: Always check residual plots and R² values
4. **Compare Models**: Use the comparison feature to find the best fitting model
5. **Document Results**: Export equations and plots for your reports

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Run `pip install -r requirements.txt` to install dependencies |
| CSV not loading | Ensure CSV has headers and proper formatting |
| Fitting fails | Check that both X and Y columns are numeric and have no NaN values |
| Poor fit quality | Try different models or check for outliers in data |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

**Author**: Ezaz Ahmed (C223009)
**Project**: Numerical Methods

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request with improvements or bug fixes.


**Last Updated**: January 2026
