

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


class CurvePlotter:
    """Handles plotting of data points and fitted curves."""
    
    def __init__(self, figure=None):
        """Initialize plotter with optional matplotlib figure."""
        if figure is None:
            self.fig = Figure(figsize=(10, 6))
        else:
            self.fig = figure
        
        self.ax = None
        self.canvas = None
        
    def create_plot(self, parent_widget=None):
        """Create the main plot axes."""
        self.ax = self.fig.add_subplot(111)
        
        if parent_widget is not None:
            self.canvas = FigureCanvasTkAgg(self.fig, master=parent_widget)
            self.canvas.draw()
            return self.canvas.get_tk_widget()
        
        return None
    
    def plot_data_and_fit(self, x_data, y_data, fitter=None, show_residuals=False):
        """
        Plot original data points and fitted curve.
        
        Args:
            x_data: Original x data
            y_data: Original y data
            fitter: Fitted model (CurveFitter instance)
            show_residuals: If True, show residual lines
        """
        self.ax.clear()
        
        # Plot original data points
        self.ax.scatter(x_data, y_data, color='blue', s=50, alpha=0.6, 
                       label='Data Points', zorder=3)
        
        # Plot fitted curve if available
        if fitter is not None and fitter.fitted_function is not None:
            # Generate smooth curve
            x_min, x_max = np.min(x_data), np.max(x_data)
            x_range = x_max - x_min
            x_smooth = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
            
            try:
                y_smooth = fitter.predict(x_smooth)
                self.ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                           label=f'{fitter.method_name}', zorder=2)
                
                # Show residuals
                if show_residuals:
                    y_pred = fitter.predict(x_data)
                    for i in range(len(x_data)):
                        self.ax.plot([x_data[i], x_data[i]], 
                                   [y_data[i], y_pred[i]], 
                                   'g--', alpha=0.5, linewidth=1)
                
            except Exception as e:
                print(f"Error plotting fitted curve: {e}")
        
        # Formatting
        self.ax.set_xlabel('x', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('y', fontsize=12, fontweight='bold')
        self.ax.set_title('Curve Fitting Visualization', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='best', fontsize=10)
        
        if self.canvas:
            self.canvas.draw()
    
    def plot_residuals(self, x_data, residuals):
        """
        Plot residuals vs x.
        
        Args:
            x_data: X values
            residuals: Residual values
        """
        self.ax.clear()
        
        # Residual plot
        self.ax.scatter(x_data, residuals, color='red', s=50, alpha=0.6)
        self.ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Formatting
        self.ax.set_xlabel('x', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        self.ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        if self.canvas:
            self.canvas.draw()
    
    def plot_residual_histogram(self, residuals):
        """
        Plot histogram of residuals.
        
        Args:
            residuals: Residual values
        """
        self.ax.clear()
        
        # Histogram
        n, bins, patches = self.ax.hist(residuals, bins=15, color='skyblue', 
                                       edgecolor='black', alpha=0.7)
        
        # Add normal distribution overlay
        mu = np.mean(residuals)
        sigma = np.std(residuals)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * 
             np.exp(-0.5 * (1 / sigma * (x - mu))**2))
        
        # Scale to match histogram
        y = y * len(residuals) * (bins[1] - bins[0])
        self.ax.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
        
        # Formatting
        self.ax.set_xlabel('Residual Value', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        self.ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        if self.canvas:
            self.canvas.draw()
    
    def plot_multiple_models(self, x_data, y_data, fitters_dict):
        """
        Plot multiple fitted models for comparison.
        
        Args:
            x_data: Original x data
            y_data: Original y data
            fitters_dict: Dict of {name: fitter} pairs
        """
        self.ax.clear()
        
        # Plot original data
        self.ax.scatter(x_data, y_data, color='black', s=50, alpha=0.8, 
                       label='Data Points', zorder=5)
        
        # Color cycle
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Generate smooth x values
        x_min, x_max = np.min(x_data), np.max(x_data)
        x_range = x_max - x_min
        x_smooth = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
        
        # Plot each fitted curve
        for idx, (name, fitter) in enumerate(fitters_dict.items()):
            if fitter is not None and fitter.fitted_function is not None:
                try:
                    y_smooth = fitter.predict(x_smooth)
                    color = colors[idx % len(colors)]
                    
                    label = f"{name} (R²={fitter.r_squared:.3f})"
                    self.ax.plot(x_smooth, y_smooth, color=color, linewidth=2, 
                               label=label, alpha=0.7, zorder=3)
                except Exception as e:
                    print(f"Error plotting {name}: {e}")
        
        # Formatting
        self.ax.set_xlabel('x', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('y', fontsize=12, fontweight='bold')
        self.ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='best', fontsize=9)
        
        if self.canvas:
            self.canvas.draw()
    
    def clear_plot(self):
        """Clear the plot."""
        if self.ax:
            self.ax.clear()
            if self.canvas:
                self.canvas.draw()
    
    def save_plot(self, filename, dpi=300):
        """
        Save the current plot to file.
        
        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        try:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            return True, f"Plot saved to {filename}"
        except Exception as e:
            return False, f"Error saving plot: {str(e)}"


class InteractivePlotter(CurvePlotter):
    """Extended plotter with interactive features."""
    
    def __init__(self, figure=None):
        super().__init__(figure)
        self.selected_point = None
        self.drag_enabled = False
        
    def enable_point_dragging(self, x_data, y_data, callback=None):
        """
        Enable dragging of data points.
        
        Args:
            x_data: X data array
            y_data: Y data array
            callback: Function to call when point is moved (index, new_x, new_y)
        """
        self.drag_enabled = True
        self.x_data = x_data
        self.y_data = y_data
        self.callback = callback
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
    
    def _on_press(self, event):
        """Handle mouse press event."""
        if not self.drag_enabled or event.inaxes != self.ax:
            return
        
        # Find closest point
        if len(self.x_data) == 0:
            return
        
        distances = np.sqrt((self.x_data - event.xdata)**2 + 
                          (self.y_data - event.ydata)**2)
        
        # Normalize by axis ranges
        x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        
        normalized_distances = np.sqrt(
            ((self.x_data - event.xdata) / x_range)**2 + 
            ((self.y_data - event.ydata) / y_range)**2
        )
        
        min_idx = np.argmin(normalized_distances)
        
        # Select point if close enough
        if normalized_distances[min_idx] < 0.05:  # 5% of plot size
            self.selected_point = min_idx
    
    def _on_release(self, event):
        """Handle mouse release event."""
        self.selected_point = None
    
    def _on_motion(self, event):
        """Handle mouse motion event."""
        if self.selected_point is None or event.inaxes != self.ax:
            return
        
        # Update point position
        if self.callback:
            self.callback(self.selected_point, event.xdata, event.ydata)


def create_subplot_figure(n_rows=1, n_cols=1, figsize=(12, 8)):
    """
    Create a figure with multiple subplots.
    
    Returns:
        (fig, axes)
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.tight_layout(pad=3.0)
    return fig, axes


def export_to_latex(fitter, filename):
    """
    Export fitted equation to LaTeX format.
    
    Args:
        fitter: CurveFitter instance
        filename: Output filename
    """
    try:
        equation = fitter.get_equation_string()
        
        # Convert to LaTeX
        latex_eq = equation.replace('*', r'\cdot ')
        latex_eq = latex_eq.replace('e^', r'e^{') + '}'
        latex_eq = latex_eq.replace('x^', 'x^{') 
        
        latex_content = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\begin{{document}}

\\section*{{Fitted Equation}}

\\begin{{equation}}
{latex_eq}
\\end{{equation}}

\\textbf{{Statistics:}}
\\begin{{itemize}}
    \\item $R^2 = {fitter.r_squared:.4f}$
    \\item $MSE = {fitter.mse:.4f}$
\\end{{itemize}}

\\end{{document}}
"""
        
        with open(filename, 'w') as f:
            f.write(latex_content)
        
        return True, f"LaTeX exported to {filename}"
    
    except Exception as e:
        return False, f"Error exporting LaTeX: {str(e)}"


# Testing function
if __name__ == "__main__":
    print("Testing Visualization Module\n")
    
    # Generate sample data
    x = np.linspace(0, 10, 20)
    y = 2*x + 1 + np.random.normal(0, 2, 20)
    
    # Create standalone plot (not for Tkinter)
    plotter = CurvePlotter()
    plotter.fig, plotter.ax = plt.subplots(figsize=(8, 6))
    
    # Import fitting module for testing
    import sys
    sys.path.append('.')
    from fitting_methods import LinearRegression
    
    fitter = LinearRegression()
    fitter.fit(x, y)
    
    plotter.plot_data_and_fit(x, y, fitter, show_residuals=True)
    print(f"Plot created with equation: {fitter.get_equation_string()}")
    print(f"R² = {fitter.r_squared:.4f}")
    
    plt.show()
