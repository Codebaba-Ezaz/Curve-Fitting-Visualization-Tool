

import numpy as np
import pandas as pd
import csv
from pathlib import Path


class DataHandler:
    """Handles all data operations including import, export, and validation."""
    
    @staticmethod
    def validate_data(x, y):
        """
        Validate input data.
        
        Returns:
            (bool, str): (is_valid, message)
        """
        if len(x) == 0 or len(y) == 0:
            return False, "Data arrays cannot be empty"
        
        if len(x) != len(y):
            return False, "X and Y arrays must have the same length"
        
        if len(x) < 2:
            return False, "Need at least 2 data points"
        
        # Check for NaN or Inf
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            return False, "Data contains NaN values"
        
        if np.any(np.isinf(x)) or np.any(np.isinf(y)):
            return False, "Data contains infinite values"
        
        return True, "Data is valid"
    
    @staticmethod
    def load_csv(filepath):
        """
        Load data from CSV file.
        
        Expected format:
        - Two columns: x, y
        - First row can be header or data
        
        Returns:
            (x_array, y_array, success, message)
        """
        try:
            # Try reading with pandas
            df = pd.read_csv(filepath)
            
            # Check if it has at least 2 columns
            if df.shape[1] < 2:
                return None, None, False, "CSV must have at least 2 columns"
            
            # Take first two columns
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            
            # Try to convert to float
            try:
                x = x.astype(float)
                y = y.astype(float)
            except ValueError:
                # Maybe first row is header, try again
                df = pd.read_csv(filepath, header=None)
                x = df.iloc[:, 0].values.astype(float)
                y = df.iloc[:, 1].values.astype(float)
            
            # Validate
            is_valid, msg = DataHandler.validate_data(x, y)
            if not is_valid:
                return None, None, False, msg
            
            return x, y, True, f"Loaded {len(x)} data points"
            
        except Exception as e:
            return None, None, False, f"Error loading CSV: {str(e)}"
    
    @staticmethod
    def load_txt(filepath):
        """
        Load data from TXT file (space or tab separated).
        
        Returns:
            (x_array, y_array, success, message)
        """
        try:
            # Try loading as space/tab separated
            data = np.loadtxt(filepath)
            
            if data.ndim == 1:
                return None, None, False, "TXT file must have 2 columns"
            
            if data.shape[1] < 2:
                return None, None, False, "TXT file must have at least 2 columns"
            
            x = data[:, 0]
            y = data[:, 1]
            
            # Validate
            is_valid, msg = DataHandler.validate_data(x, y)
            if not is_valid:
                return None, None, False, msg
            
            return x, y, True, f"Loaded {len(x)} data points"
            
        except Exception as e:
            return None, None, False, f"Error loading TXT: {str(e)}"
    
    @staticmethod
    def save_csv(filepath, x, y, fitted_x=None, fitted_y=None):
        """
        Save data and fitted curve to CSV.
        
        Args:
            filepath: Output file path
            x, y: Original data
            fitted_x, fitted_y: Fitted curve data (optional)
        """
        try:
            data = {
                'x_original': x,
                'y_original': y
            }
            
            if fitted_x is not None and fitted_y is not None:
                data['x_fitted'] = fitted_x
                data['y_fitted'] = fitted_y
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
            return True, f"Data saved to {filepath}"
            
        except Exception as e:
            return False, f"Error saving CSV: {str(e)}"
    
    @staticmethod
    def generate_sample_data(data_type, n_points=20, noise_level=0.1):
        """
        Generate sample data for testing.
        
        Args:
            data_type: 'linear', 'polynomial', 'exponential', 'power', 'sine'
            n_points: Number of points to generate
            noise_level: Standard deviation of Gaussian noise
        
        Returns:
            (x_array, y_array)
        """
        np.random.seed(42)  # For reproducibility
        
        if data_type == 'linear':
            x = np.linspace(0, 10, n_points)
            y = 2.5 * x + 3 + np.random.normal(0, noise_level * 10, n_points)
            
        elif data_type == 'polynomial':
            x = np.linspace(-2, 2, n_points)
            y = 2*x**2 - 3*x + 1 + np.random.normal(0, noise_level * 5, n_points)
            
        elif data_type == 'exponential':
            x = np.linspace(0, 3, n_points)
            y = 2 * np.exp(0.5 * x) + np.random.normal(0, noise_level * 2, n_points)
            
        elif data_type == 'power':
            x = np.linspace(1, 10, n_points)
            y = 2 * x**1.5 + np.random.normal(0, noise_level * 5, n_points)
            
        elif data_type == 'sine':
            x = np.linspace(0, 2*np.pi, n_points)
            y = np.sin(x) + np.random.normal(0, noise_level, n_points)
            
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return x, y


class DataTable:
    """Manages data in a table format for GUI display."""
    
    def __init__(self):
        self.x = []
        self.y = []
        
    def add_point(self, x_val, y_val):
        """Add a data point."""
        self.x.append(float(x_val))
        self.y.append(float(y_val))
        
    def remove_point(self, index):
        """Remove a data point at given index."""
        if 0 <= index < len(self.x):
            self.x.pop(index)
            self.y.pop(index)
            
    def update_point(self, index, x_val, y_val):
        """Update a data point at given index."""
        if 0 <= index < len(self.x):
            self.x[index] = float(x_val)
            self.y[index] = float(y_val)
            
    def clear(self):
        """Clear all data."""
        self.x = []
        self.y = []
        
    def set_data(self, x, y):
        """Set data from arrays."""
        self.x = list(x)
        self.y = list(y)
        
    def get_data(self):
        """Get data as numpy arrays."""
        return np.array(self.x), np.array(self.y)
    
    def get_size(self):
        """Get number of data points."""
        return len(self.x)
    
    def sort_by_x(self):
        """Sort data by x values."""
        if len(self.x) > 0:
            sorted_indices = np.argsort(self.x)
            self.x = [self.x[i] for i in sorted_indices]
            self.y = [self.y[i] for i in sorted_indices]


# Testing function
if __name__ == "__main__":
    print("Testing Data Handler\n")
    
    # Test sample data generation
    print("Generating sample data...")
    for dtype in ['linear', 'polynomial', 'exponential', 'power']:
        x, y = DataHandler.generate_sample_data(dtype, n_points=10)
        print(f"{dtype}: Generated {len(x)} points")
        is_valid, msg = DataHandler.validate_data(x, y)
        print(f"  Validation: {msg}\n")
    
    # Test DataTable
    print("Testing DataTable...")
    table = DataTable()
    table.add_point(1, 2)
    table.add_point(2, 4)
    table.add_point(3, 6)
    print(f"Added 3 points, size: {table.get_size()}")
    
    x, y = table.get_data()
    print(f"Data: x={x}, y={y}")
