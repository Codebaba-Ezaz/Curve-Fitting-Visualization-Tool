"""
Curve Fitting Methods Module
Implements various numerical methods for curve fitting including:
- Linear Regression
- Polynomial Regression
- Exponential Fitting
- Power-Law Fitting

Team Meeqat - Numerical Methods Project
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class CurveFitter:
    """Base class for all curve fitting methods."""
    
    def __init__(self):
        self.coefficients = None
        self.fitted_function = None
        self.r_squared = None
        self.mse = None
        self.method_name = "Base"
        
    def fit(self, x, y):
        """Fit the curve to the data. To be implemented by subclasses."""
        raise NotImplementedError
        
    def predict(self, x):
        """Predict y values for given x values."""
        if self.fitted_function is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.fitted_function(x)
    
    def calculate_metrics(self, x, y):
        """Calculate R-squared and MSE."""
        y_pred = self.predict(x)
        
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean Squared Error
        self.mse = np.mean((y - y_pred) ** 2)
        
        return self.r_squared, self.mse
    
    def get_equation_string(self):
        """Return a string representation of the fitted equation."""
        raise NotImplementedError


class LinearRegression(CurveFitter):
    """
    Linear Regression using Normal Equations.
    
    Solves: θ = (X^T X)^(-1) X^T y
    Model: y = a + bx
    """
    
    def __init__(self):
        super().__init__()
        self.method_name = "Linear Regression"
        self.slope = None
        self.intercept = None
        
    def fit(self, x, y):
        """Fit linear model using normal equations."""
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        # Add intercept term (column of ones)
        X = np.hstack([np.ones((len(x), 1)), x])
        
        # Normal equation: θ = (X^T X)^(-1) X^T y
        try:
            XtX = X.T @ X
            XtX_inv = np.linalg.inv(XtX)
            self.coefficients = XtX_inv @ X.T @ y
            
            self.intercept = self.coefficients[0]
            self.slope = self.coefficients[1]
            
            # Define fitted function
            self.fitted_function = lambda x_val: self.intercept + self.slope * x_val
            
            # Calculate metrics
            self.calculate_metrics(x.flatten(), y)
            
            return True, "Linear fit successful"
            
        except np.linalg.LinAlgError:
            return False, "Matrix is singular. Cannot compute inverse."
    
    def get_equation_string(self):
        """Return equation as string."""
        sign = '+' if self.intercept >= 0 else ''
        return f"y = {self.slope:.4f}x {sign}{self.intercept:.4f}"


class PolynomialRegression(CurveFitter):
    """
    Polynomial Regression using Vandermonde matrices.
    
    Model: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ
    """
    
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.method_name = f"Polynomial Regression (degree {degree})"
        
    def fit(self, x, y):
        """Fit polynomial model."""
        x = np.array(x)
        y = np.array(y)
        
        if len(x) <= self.degree:
            return False, f"Need at least {self.degree + 1} data points for degree {self.degree} polynomial"
        
        try:
            # Use numpy's polyfit which constructs Vandermonde matrix internally
            self.coefficients = np.polyfit(x, y, self.degree)
            
            # Create polynomial function
            poly = np.poly1d(self.coefficients)
            self.fitted_function = lambda x_val: poly(x_val)
            
            # Calculate metrics
            self.calculate_metrics(x, y)
            
            return True, f"Polynomial fit (degree {self.degree}) successful"
            
        except Exception as e:
            return False, f"Polynomial fitting failed: {str(e)}"
    
    def get_equation_string(self):
        """Return equation as string."""
        terms = []
        for i, coef in enumerate(self.coefficients):
            power = self.degree - i
            if abs(coef) < 1e-10:  # Skip very small coefficients
                continue
            if power == 0:
                terms.append(f"{coef:.4f}")
            elif power == 1:
                terms.append(f"{coef:.4f}x")
            else:
                terms.append(f"{coef:.4f}x^{power}")
        
        equation = " + ".join(terms)
        equation = equation.replace("+ -", "- ")
        return f"y = {equation}"


class ExponentialFit(CurveFitter):
    """
    Exponential Fitting using non-linear least squares.
    
    Model: y = a * e^(bx)
    """
    
    def __init__(self):
        super().__init__()
        self.method_name = "Exponential Fit"
        self.a = None
        self.b = None
        
    def _exponential_func(self, x, a, b):
        """Exponential function."""
        return a * np.exp(b * x)
    
    def fit(self, x, y):
        """Fit exponential model using Levenberg-Marquardt algorithm."""
        x = np.array(x)
        y = np.array(y)
        
        # Check for positive y values (required for exponential)
        if np.any(y <= 0):
            # Try to estimate with linearization anyway
            try:
                # Linearize: ln(y) = ln(a) + bx
                log_y = np.log(np.abs(y) + 1e-10)
                coeffs = np.polyfit(x, log_y, 1)
                b_init = coeffs[0]
                a_init = np.exp(coeffs[1])
            except:
                a_init, b_init = 1.0, 0.1
        else:
            # Initial guess using linearization
            try:
                log_y = np.log(y)
                coeffs = np.polyfit(x, log_y, 1)
                b_init = coeffs[0]
                a_init = np.exp(coeffs[1])
            except:
                a_init, b_init = np.mean(y), 0.1
        
        try:
            # Non-linear least squares fitting
            popt, pcov = curve_fit(
                self._exponential_func, 
                x, y, 
                p0=[a_init, b_init],
                maxfev=5000
            )
            
            self.a, self.b = popt
            self.coefficients = popt
            
            # Define fitted function
            self.fitted_function = lambda x_val: self._exponential_func(x_val, self.a, self.b)
            
            # Calculate metrics
            self.calculate_metrics(x, y)
            
            return True, "Exponential fit successful"
            
        except Exception as e:
            return False, f"Exponential fitting failed: {str(e)}"
    
    def get_equation_string(self):
        """Return equation as string."""
        return f"y = {self.a:.4f} * e^({self.b:.4f}x)"


class PowerLawFit(CurveFitter):
    """
    Power-Law Fitting using logarithmic transformation.
    
    Model: y = a * x^b
    """
    
    def __init__(self):
        super().__init__()
        self.method_name = "Power-Law Fit"
        self.a = None
        self.b = None
        
    def _power_func(self, x, a, b):
        """Power function."""
        return a * np.power(np.abs(x), b)
    
    def fit(self, x, y):
        """Fit power-law model."""
        x = np.array(x)
        y = np.array(y)
        
        # Check for positive values (required for power-law)
        if np.any(x <= 0) or np.any(y <= 0):
            # Try non-linear fitting anyway
            try:
                popt, pcov = curve_fit(
                    self._power_func, 
                    x, y, 
                    p0=[1.0, 1.0],
                    maxfev=5000
                )
                self.a, self.b = popt
            except:
                return False, "Power-law requires positive x and y values"
        else:
            # Linearize: ln(y) = ln(a) + b*ln(x)
            try:
                log_x = np.log(x)
                log_y = np.log(y)
                coeffs = np.polyfit(log_x, log_y, 1)
                self.b = coeffs[0]
                self.a = np.exp(coeffs[1])
            except:
                return False, "Logarithmic transformation failed"
        
        self.coefficients = np.array([self.a, self.b])
        
        # Define fitted function
        self.fitted_function = lambda x_val: self._power_func(x_val, self.a, self.b)
        
        # Calculate metrics
        self.calculate_metrics(x, y)
        
        return True, "Power-law fit successful"
    
    def get_equation_string(self):
        """Return equation as string."""
        return f"y = {self.a:.4f} * x^{self.b:.4f}"


def get_fitter(method_type, **kwargs):
    """
    Factory function to get appropriate fitter.
    
    Args:
        method_type: String - 'linear', 'polynomial', 'exponential', 'power'
        **kwargs: Additional parameters (e.g., degree for polynomial)
    
    Returns:
        CurveFitter instance
    """
    method_type = method_type.lower()
    
    if method_type == 'linear':
        return LinearRegression()
    elif method_type == 'polynomial':
        degree = kwargs.get('degree', 2)
        return PolynomialRegression(degree=degree)
    elif method_type == 'exponential':
        return ExponentialFit()
    elif method_type == 'power':
        return PowerLawFit()
    else:
        raise ValueError(f"Unknown method type: {method_type}")


# Testing function
if __name__ == "__main__":
    # Test with sample data
    print("Testing Curve Fitting Methods\n")
    
    # Linear test
    x_linear = np.array([1, 2, 3, 4, 5])
    y_linear = np.array([2.1, 4.0, 6.1, 7.9, 10.2])
    
    linear_fit = LinearRegression()
    success, msg = linear_fit.fit(x_linear, y_linear)
    print(f"Linear Regression: {msg}")
    print(f"Equation: {linear_fit.get_equation_string()}")
    print(f"R²: {linear_fit.r_squared:.4f}, MSE: {linear_fit.mse:.4f}\n")
    
    # Polynomial test
    x_poly = np.array([0, 1, 2, 3, 4])
    y_poly = np.array([1, 2.5, 6, 10.5, 17])
    
    poly_fit = PolynomialRegression(degree=2)
    success, msg = poly_fit.fit(x_poly, y_poly)
    print(f"Polynomial Regression: {msg}")
    print(f"Equation: {poly_fit.get_equation_string()}")
    print(f"R²: {poly_fit.r_squared:.4f}, MSE: {poly_fit.mse:.4f}\n")
