"""
Error Metrics and Statistical Analysis Module
Calculates various error metrics and statistical measures for curve fitting.

Team Meeqat - Numerical Methods Project
"""

import numpy as np
from scipy import stats


class ErrorMetrics:
    """Calculate error metrics and goodness-of-fit statistics."""
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Calculate Mean Squared Error (MSE)."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        """Calculate Root Mean Squared Error (RMSE)."""
        return np.sqrt(ErrorMetrics.mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """Calculate Mean Absolute Error (MAE)."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r_squared(y_true, y_pred):
        """
        Calculate R-squared (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        where:
            SS_res = Σ(y_i - ŷ_i)²  (residual sum of squares)
            SS_tot = Σ(y_i - ȳ)²    (total sum of squares)
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def adjusted_r_squared(y_true, y_pred, n_parameters):
        """
        Calculate Adjusted R-squared.
        
        Accounts for the number of parameters in the model.
        """
        n = len(y_true)
        r2 = ErrorMetrics.r_squared(y_true, y_pred)
        
        if n - n_parameters - 1 <= 0:
            return r2
        
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_parameters - 1)
        return adj_r2
    
    @staticmethod
    def residuals(y_true, y_pred):
        """Calculate residuals (errors)."""
        return y_true - y_pred
    
    @staticmethod
    def standardized_residuals(y_true, y_pred):
        """Calculate standardized residuals."""
        residuals = ErrorMetrics.residuals(y_true, y_pred)
        std = np.std(residuals)
        if std == 0:
            return residuals
        return residuals / std
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, n_parameters=2):
        """
        Calculate all error metrics at once.
        
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'MSE': ErrorMetrics.mean_squared_error(y_true, y_pred),
            'RMSE': ErrorMetrics.root_mean_squared_error(y_true, y_pred),
            'MAE': ErrorMetrics.mean_absolute_error(y_true, y_pred),
            'R²': ErrorMetrics.r_squared(y_true, y_pred),
            'Adjusted R²': ErrorMetrics.adjusted_r_squared(y_true, y_pred, n_parameters),
            'Residuals': ErrorMetrics.residuals(y_true, y_pred)
        }
        
        return metrics


class ResidualAnalysis:
    """Perform residual analysis for model diagnostics."""
    
    @staticmethod
    def normality_test(residuals):
        """
        Test if residuals are normally distributed using Shapiro-Wilk test.
        
        Returns:
            (statistic, p_value, is_normal)
        """
        if len(residuals) < 3:
            return None, None, None
        
        statistic, p_value = stats.shapiro(residuals)
        is_normal = p_value > 0.05  # 5% significance level
        
        return statistic, p_value, is_normal
    
    @staticmethod
    def durbin_watson(residuals):
        """
        Calculate Durbin-Watson statistic to test for autocorrelation.
        
        DW ≈ 2 indicates no autocorrelation
        DW < 2 indicates positive autocorrelation
        DW > 2 indicates negative autocorrelation
        """
        diff = np.diff(residuals)
        dw = np.sum(diff**2) / np.sum(residuals**2)
        return dw
    
    @staticmethod
    def outlier_detection(residuals, threshold=3):
        """
        Detect outliers using standardized residuals.
        
        Points with |standardized residual| > threshold are considered outliers.
        """
        std_residuals = ErrorMetrics.standardized_residuals(residuals, np.zeros_like(residuals))
        outlier_indices = np.where(np.abs(std_residuals) > threshold)[0]
        
        return outlier_indices, std_residuals


class ModelComparison:
    """Compare multiple fitted models."""
    
    @staticmethod
    def aic(n, mse, n_parameters):
        """
        Calculate Akaike Information Criterion (AIC).
        
        AIC = n*ln(MSE) + 2*k
        where k is the number of parameters
        
        Lower AIC indicates better model.
        """
        if mse <= 0 or n <= 0:
            return np.inf
        
        aic = n * np.log(mse) + 2 * n_parameters
        return aic
    
    @staticmethod
    def bic(n, mse, n_parameters):
        """
        Calculate Bayesian Information Criterion (BIC).
        
        BIC = n*ln(MSE) + k*ln(n)
        where k is the number of parameters
        
        Lower BIC indicates better model.
        """
        if mse <= 0 or n <= 0:
            return np.inf
        
        bic = n * np.log(mse) + n_parameters * np.log(n)
        return bic
    
    @staticmethod
    def compare_models(models_data):
        """
        Compare multiple models.
        
        Args:
            models_data: List of dicts, each containing:
                - 'name': Model name
                - 'y_true': True values
                - 'y_pred': Predicted values
                - 'n_parameters': Number of parameters
        
        Returns:
            DataFrame with comparison metrics
        """
        import pandas as pd
        
        comparison = []
        
        for model in models_data:
            name = model['name']
            y_true = model['y_true']
            y_pred = model['y_pred']
            n_params = model['n_parameters']
            n = len(y_true)
            
            metrics = ErrorMetrics.calculate_all_metrics(y_true, y_pred, n_params)
            
            comparison.append({
                'Model': name,
                'R²': metrics['R²'],
                'Adjusted R²': metrics['Adjusted R²'],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'AIC': ModelComparison.aic(n, metrics['MSE'], n_params),
                'BIC': ModelComparison.bic(n, metrics['MSE'], n_params)
            })
        
        df = pd.DataFrame(comparison)
        return df


# Testing function
if __name__ == "__main__":
    print("Testing Error Metrics Module\n")
    
    # Generate sample data
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    # Calculate metrics
    print("Error Metrics:")
    metrics = ErrorMetrics.calculate_all_metrics(y_true, y_pred, n_parameters=2)
    for key, value in metrics.items():
        if key != 'Residuals':
            print(f"  {key}: {value:.4f}")
    
    print(f"\nResiduals: {metrics['Residuals']}")
    
    # Residual analysis
    print("\nResidual Analysis:")
    residuals = metrics['Residuals']
    stat, p_val, is_normal = ResidualAnalysis.normality_test(residuals)
    print(f"  Normality test p-value: {p_val:.4f}")
    print(f"  Residuals are {'normally' if is_normal else 'not normally'} distributed")
    
    dw = ResidualAnalysis.durbin_watson(residuals)
    print(f"  Durbin-Watson statistic: {dw:.4f}")
