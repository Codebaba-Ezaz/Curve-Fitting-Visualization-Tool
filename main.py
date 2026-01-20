"""
Curve Fitting Visualization Tool

Copyright (c) 2026 Ezaz Ahmed (C223009)
Numerical Methods Project
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from fitting_methods import get_fitter
from data_handler import DataHandler, DataTable
from error_metrics import ErrorMetrics, ResidualAnalysis
from visualization import CurvePlotter, export_to_latex


class CurveFittingApp:
    """Main application class for the Curve Fitting Visualization Tool."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Curve Fitting Visualization Tool")
        self.root.geometry("1400x900")
        
        # Data storage
        self.data_table = DataTable()
        self.current_fitter = None
        self.comparison_fitters = {}
        
        # Setup UI
        self.setup_styles()
        self.create_widgets()
        
        # Load default sample data
        self.load_sample_data('linear')
        
    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Arial', 10), foreground='#27ae60')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='#e74c3c')
        
        style.configure('Action.TButton', font=('Arial', 10, 'bold'))
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(title_frame, text="Curve Fitting Visualization Tool", 
                 style='Title.TLabel').pack()
        ttk.Label(title_frame, text="Ezaz Ahmed (C223009) - Numerical Methods Project", 
                 style='Info.TLabel').pack()
        
        # Left Panel - Data & Controls
        left_panel = ttk.Frame(main_container, relief='ridge', borderwidth=2)
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.create_data_panel(left_panel)
        self.create_control_panel(left_panel)
        
        # Center Panel - Visualization
        center_panel = ttk.Frame(main_container, relief='ridge', borderwidth=2)
        center_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.create_plot_panel(center_panel)
        
        # Right Panel - Results & Analysis
        right_panel = ttk.Frame(main_container, relief='ridge', borderwidth=2)
        right_panel.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.create_results_panel(right_panel)
        
        # Configure grid weights
        main_container.columnconfigure(0, weight=1, minsize=300)
        main_container.columnconfigure(1, weight=3, minsize=600)
        main_container.columnconfigure(2, weight=1, minsize=300)
        main_container.rowconfigure(1, weight=1)
        
    def create_data_panel(self, parent):
        """Create data input panel."""
        data_frame = ttk.LabelFrame(parent, text="Data Input", padding="10")
        data_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Data table
        table_frame = ttk.Frame(data_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview for data
        columns = ('Index', 'X', 'Y')
        self.data_tree = ttk.Treeview(table_frame, columns=columns, show='headings',
                                      yscrollcommand=y_scroll.set,
                                      xscrollcommand=x_scroll.set,
                                      height=10)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=70, anchor=tk.CENTER)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        y_scroll.config(command=self.data_tree.yview)
        x_scroll.config(command=self.data_tree.xview)
        
        # Data buttons
        button_frame = ttk.Frame(data_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="Add Point", command=self.add_data_point,
                  style='Action.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Remove", command=self.remove_data_point).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear All", command=self.clear_data).pack(side=tk.LEFT, padx=2)
        
        # Import/Export buttons
        io_frame = ttk.Frame(data_frame)
        io_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(io_frame, text="Import CSV", command=self.import_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=2)
        
        # Sample data
        sample_frame = ttk.LabelFrame(data_frame, text="Load Sample Data", padding="5")
        sample_frame.pack(fill=tk.X, pady=(5, 0))
        
        samples = ['Linear', 'Polynomial', 'Exponential', 'Power']
        for sample in samples:
            ttk.Button(sample_frame, text=sample, 
                      command=lambda s=sample.lower(): self.load_sample_data(s),
                      width=12).pack(side=tk.LEFT, padx=2)
        
    def create_control_panel(self, parent):
        """Create fitting method control panel."""
        control_frame = ttk.LabelFrame(parent, text="Fitting Controls", padding="10")
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        # Method selection
        ttk.Label(control_frame, text="Select Method:", style='Subtitle.TLabel').pack(anchor=tk.W)
        
        self.method_var = tk.StringVar(value='linear')
        methods = [
            ('Linear Regression', 'linear'),
            ('Polynomial Regression', 'polynomial'),
            ('Exponential Fit', 'exponential'),
            ('Power-Law Fit', 'power')
        ]
        
        for text, value in methods:
            ttk.Radiobutton(control_frame, text=text, variable=self.method_var, 
                          value=value, command=self.on_method_change).pack(anchor=tk.W, pady=2)
        
        # Polynomial degree (shown only for polynomial)
        self.poly_frame = ttk.Frame(control_frame)
        self.poly_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(self.poly_frame, text="Polynomial Degree:").pack(side=tk.LEFT)
        self.degree_var = tk.IntVar(value=2)
        degree_spinner = ttk.Spinbox(self.poly_frame, from_=1, to=5, 
                                    textvariable=self.degree_var, width=10)
        degree_spinner.pack(side=tk.LEFT, padx=5)
        
        self.poly_frame.pack_forget()  # Hide initially
        
        # Fit button
        ttk.Button(control_frame, text="Fit Curve", command=self.fit_curve,
                  style='Action.TButton').pack(fill=tk.X, pady=(10, 0))
        
        # Options
        options_frame = ttk.LabelFrame(control_frame, text="Display Options", padding="5")
        options_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.show_residuals_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Show Residuals", 
                       variable=self.show_residuals_var,
                       command=self.update_plot).pack(anchor=tk.W)
        
        self.show_equation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Equation",
                       variable=self.show_equation_var,
                       command=self.update_results).pack(anchor=tk.W)
        
        # Model comparison
        comparison_frame = ttk.LabelFrame(control_frame, text="Model Comparison", padding="5")
        comparison_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(comparison_frame, text="Add to Comparison", 
                  command=self.add_to_comparison).pack(fill=tk.X, pady=2)
        ttk.Button(comparison_frame, text="Compare All Models",
                  command=self.compare_models).pack(fill=tk.X, pady=2)
        ttk.Button(comparison_frame, text="Clear Comparison",
                  command=self.clear_comparison).pack(fill=tk.X, pady=2)
        
    def create_plot_panel(self, parent):
        """Create main plotting panel."""
        plot_frame = ttk.Frame(parent, padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(plot_frame, text="Visualization", style='Subtitle.TLabel').pack()
        
        # Matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.plotter = CurvePlotter(self.fig)
        
        canvas_widget = self.plotter.create_plot(plot_frame)
        canvas_widget.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        
        toolbar = NavigationToolbar2Tk(self.plotter.canvas, toolbar_frame)
        toolbar.update()
        
        # View buttons
        view_frame = ttk.Frame(plot_frame)
        view_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(view_frame, text="Data & Fit", command=self.show_main_plot).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_frame, text="Residuals", command=self.show_residual_plot).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_frame, text="Distribution", command=self.show_residual_histogram).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_frame, text="Export Plot", command=self.export_plot).pack(side=tk.RIGHT, padx=2)
        
    def create_results_panel(self, parent):
        """Create results and statistics panel."""
        results_frame = ttk.Frame(parent, padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(results_frame, text="Results & Analysis", style='Subtitle.TLabel').pack()
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, width=35, height=30,
                                                      wrap=tk.WORD, font=('Courier', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Action buttons
        action_frame = ttk.Frame(results_frame)
        action_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(action_frame, text="Copy Results", command=self.copy_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Export Report", command=self.export_report).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Export LaTeX", command=self.export_latex).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Help", command=self.show_help).pack(side=tk.RIGHT, padx=2)
        
    def on_method_change(self):
        """Handle method selection change."""
        method = self.method_var.get()
        
        # Show/hide polynomial degree selector
        if method == 'polynomial':
            self.poly_frame.pack(fill=tk.X, pady=(5, 0))
        else:
            self.poly_frame.pack_forget()
    
    def update_data_table_display(self):
        """Update the data table display."""
        # Clear existing items
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Add current data
        x, y = self.data_table.get_data()
        for i, (xi, yi) in enumerate(zip(x, y)):
            self.data_tree.insert('', 'end', values=(i+1, f'{xi:.4f}', f'{yi:.4f}'))
    
    def add_data_point(self):
        """Add a new data point."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Data Point")
        dialog.geometry("300x150")
        
        ttk.Label(dialog, text="X Value:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        x_entry = ttk.Entry(dialog, width=20)
        x_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(dialog, text="Y Value:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        y_entry = ttk.Entry(dialog, width=20)
        y_entry.grid(row=1, column=1, padx=10, pady=10)
        
        def add_point():
            try:
                x_val = float(x_entry.get())
                y_val = float(y_entry.get())
                self.data_table.add_point(x_val, y_val)
                self.data_table.sort_by_x()
                self.update_data_table_display()
                self.update_plot()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        ttk.Button(dialog, text="Add", command=add_point).grid(row=2, column=0, columnspan=2, pady=10)
    
    def remove_data_point(self):
        """Remove selected data point."""
        selection = self.data_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a point to remove")
            return
        
        item = self.data_tree.item(selection[0])
        index = int(item['values'][0]) - 1
        
        self.data_table.remove_point(index)
        self.update_data_table_display()
        self.update_plot()
    
    def clear_data(self):
        """Clear all data."""
        if messagebox.askyesno("Confirm", "Clear all data points?"):
            self.data_table.clear()
            self.current_fitter = None
            self.update_data_table_display()
            self.plotter.clear_plot()
            self.results_text.delete(1.0, tk.END)
    
    def load_sample_data(self, data_type):
        """Load sample data."""
        try:
            x, y = DataHandler.generate_sample_data(data_type, n_points=20, noise_level=0.1)
            self.data_table.set_data(x, y)
            self.update_data_table_display()
            self.update_plot()
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, f"Loaded {data_type} sample data\n{len(x)} points\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample data: {str(e)}")
    
    def import_csv(self):
        """Import data from CSV file."""
        filename = filedialog.askopenfilename(
            title="Import CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            x, y, success, msg = DataHandler.load_csv(filename)
            if success:
                self.data_table.set_data(x, y)
                self.update_data_table_display()
                self.update_plot()
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showerror("Error", msg)
    
    def export_csv(self):
        """Export data to CSV file."""
        if self.data_table.get_size() == 0:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            x, y = self.data_table.get_data()
            
            fitted_x, fitted_y = None, None
            if self.current_fitter is not None:
                fitted_x = np.linspace(np.min(x), np.max(x), 100)
                fitted_y = self.current_fitter.predict(fitted_x)
            
            success, msg = DataHandler.save_csv(filename, x, y, fitted_x, fitted_y)
            if success:
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showerror("Error", msg)
    
    def fit_curve(self):
        """Fit curve to data."""
        x, y = self.data_table.get_data()
        
        # Validate data
        is_valid, msg = DataHandler.validate_data(x, y)
        if not is_valid:
            messagebox.showerror("Error", msg)
            return
        
        method = self.method_var.get()
        
        try:
            # Get fitter
            if method == 'polynomial':
                degree = self.degree_var.get()
                fitter = get_fitter(method, degree=degree)
            else:
                fitter = get_fitter(method)
            
            # Fit
            success, msg = fitter.fit(x, y)
            
            if success:
                self.current_fitter = fitter
                self.update_plot()
                self.update_results()
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showerror("Error", msg)
                
        except Exception as e:
            messagebox.showerror("Error", f"Fitting failed: {str(e)}")
    
    def update_plot(self):
        """Update the main plot."""
        x, y = self.data_table.get_data()
        
        if len(x) == 0:
            self.plotter.clear_plot()
            return
        
        show_residuals = self.show_residuals_var.get()
        self.plotter.plot_data_and_fit(x, y, self.current_fitter, show_residuals)
    
    def show_main_plot(self):
        """Show main data and fit plot."""
        self.update_plot()
    
    def show_residual_plot(self):
        """Show residual plot."""
        if self.current_fitter is None:
            messagebox.showwarning("Warning", "Fit a curve first")
            return
        
        x, y = self.data_table.get_data()
        residuals = ErrorMetrics.residuals(y, self.current_fitter.predict(x))
        self.plotter.plot_residuals(x, residuals)
    
    def show_residual_histogram(self):
        """Show residual histogram."""
        if self.current_fitter is None:
            messagebox.showwarning("Warning", "Fit a curve first")
            return
        
        x, y = self.data_table.get_data()
        residuals = ErrorMetrics.residuals(y, self.current_fitter.predict(x))
        self.plotter.plot_residual_histogram(residuals)
    
    def update_results(self):
        """Update results text area."""
        self.results_text.delete(1.0, tk.END)
        
        if self.current_fitter is None:
            self.results_text.insert(1.0, "No model fitted yet.\n\nFit a curve to see results.")
            return
        
        x, y = self.data_table.get_data()
        y_pred = self.current_fitter.predict(x)
        
        # Calculate metrics
        metrics = ErrorMetrics.calculate_all_metrics(y, y_pred, 
                                                     n_parameters=len(self.current_fitter.coefficients))
        
        # Format results
        results = f"{'='*40}\n"
        results += f"  {self.current_fitter.method_name}\n"
        results += f"{'='*40}\n\n"
        
        if self.show_equation_var.get():
            results += f"FITTED EQUATION:\n"
            results += f"{self.current_fitter.get_equation_string()}\n\n"
        
        results += f"GOODNESS OF FIT:\n"
        results += f"  R² = {metrics['R²']:.6f}\n"
        results += f"  Adjusted R² = {metrics['Adjusted R²']:.6f}\n\n"
        
        results += f"ERROR METRICS:\n"
        results += f"  MSE  = {metrics['MSE']:.6f}\n"
        results += f"  RMSE = {metrics['RMSE']:.6f}\n"
        results += f"  MAE  = {metrics['MAE']:.6f}\n\n"
        
        # Residual analysis
        residuals = metrics['Residuals']
        stat, p_val, is_normal = ResidualAnalysis.normality_test(residuals)
        
        results += f"RESIDUAL ANALYSIS:\n"
        results += f"  Mean Residual = {np.mean(residuals):.6f}\n"
        results += f"  Std Residual  = {np.std(residuals):.6f}\n"
        
        if p_val is not None:
            results += f"  Normality (p-value) = {p_val:.4f}\n"
            results += f"  Distribution: {'Normal' if is_normal else 'Non-normal'}\n"
        
        dw = ResidualAnalysis.durbin_watson(residuals)
        results += f"  Durbin-Watson = {dw:.4f}\n\n"
        
        # Coefficients
        results += f"COEFFICIENTS:\n"
        for i, coef in enumerate(self.current_fitter.coefficients):
            results += f"  a{i} = {coef:.6f}\n"
        
        self.results_text.insert(1.0, results)
    
    def add_to_comparison(self):
        """Add current model to comparison list."""
        if self.current_fitter is None:
            messagebox.showwarning("Warning", "Fit a curve first")
            return
        
        name = f"{self.current_fitter.method_name}"
        
        # Check if already exists
        counter = 1
        original_name = name
        while name in self.comparison_fitters:
            name = f"{original_name} ({counter})"
            counter += 1
        
        # Store a copy
        import copy
        self.comparison_fitters[name] = copy.deepcopy(self.current_fitter)
        
        messagebox.showinfo("Success", f"Added {name} to comparison")
    
    def compare_models(self):
        """Compare all models in comparison list."""
        if len(self.comparison_fitters) == 0:
            messagebox.showwarning("Warning", "No models to compare. Add models first.")
            return
        
        x, y = self.data_table.get_data()
        
        # Plot all models
        self.plotter.plot_multiple_models(x, y, self.comparison_fitters)
        
        # Show comparison table
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Model Comparison")
        comparison_window.geometry("700x400")
        
        # Create treeview
        columns = ('Model', 'R²', 'Adj R²', 'MSE', 'RMSE')
        tree = ttk.Treeview(comparison_window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.CENTER)
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add data
        for name, fitter in self.comparison_fitters.items():
            y_pred = fitter.predict(x)
            metrics = ErrorMetrics.calculate_all_metrics(y, y_pred, 
                                                         n_parameters=len(fitter.coefficients))
            
            tree.insert('', 'end', values=(
                name,
                f"{metrics['R²']:.4f}",
                f"{metrics['Adjusted R²']:.4f}",
                f"{metrics['MSE']:.4f}",
                f"{metrics['RMSE']:.4f}"
            ))
    
    def clear_comparison(self):
        """Clear comparison list."""
        if messagebox.askyesno("Confirm", "Clear all comparison models?"):
            self.comparison_fitters.clear()
            messagebox.showinfo("Success", "Comparison list cleared")
    
    def export_plot(self):
        """Export current plot."""
        filename = filedialog.asksaveasfilename(
            title="Export Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            success, msg = self.plotter.save_plot(filename, dpi=300)
            if success:
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showerror("Error", msg)
    
    def export_report(self):
        """Export full analysis report."""
        if self.current_fitter is None:
            messagebox.showwarning("Warning", "Fit a curve first")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                report = self.results_text.get(1.0, tk.END)
                
                # Add header
                header = f"""
CURVE FITTING ANALYSIS REPORT
Ezaz Ahmed (C223009) - Numerical Methods Project
=====================================

Generated: {np.datetime64('today')}
Data Points: {self.data_table.get_size()}

"""
                
                with open(filename, 'w') as f:
                    f.write(header + report)
                
                messagebox.showinfo("Success", f"Report exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def export_latex(self):
        """Export fitted equation to LaTeX format."""
        if self.current_fitter is None:
            messagebox.showwarning("Warning", "Fit a curve first")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export LaTeX",
            defaultextension=".tex",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")]
        )
        
        if filename:
            success, msg = export_to_latex(self.current_fitter, filename)
            if success:
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showerror("Error", msg)
    
    def copy_results(self):
        """Copy results to clipboard."""
        results = self.results_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(results)
        messagebox.showinfo("Success", "Results copied to clipboard")
    
    def show_help(self):
        """Show help dialog."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - Curve Fitting Tool")
        help_window.geometry("600x500")
        
        help_text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=('Arial', 10))
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_content = """
CURVE FITTING VISUALIZATION TOOL
=================================

Quick Start:
1. Load sample data or import your own CSV file
2. Select a fitting method (Linear, Polynomial, Exponential, Power-Law)
3. Click "Fit Curve" to apply the method
4. View results and error metrics in the right panel

Fitting Methods:
----------------
• Linear Regression: y = ax + b
  Best for data with linear relationships

• Polynomial Regression: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ
  Use degree 2-5 for curved data
  Warning: High degrees may overfit

• Exponential Fit: y = a * e^(bx)
  For exponential growth/decay
  Requires positive y values

• Power-Law Fit: y = a * x^b
  For power relationships
  Requires positive x and y values

Data Management:
---------------
• Add Point: Manually add individual data points
• Remove: Select and remove points from table
• Import CSV: Load data from CSV file (2 columns: x, y)
• Export CSV: Save data and fitted curve
• Sample Data: Load pre-generated example datasets

Visualization:
-------------
• Data & Fit: Main plot with data points and fitted curve
• Residuals: Plot of errors (y - ŷ) vs x
• Distribution: Histogram of residual values
• Show Residuals: Toggle vertical error lines

Analysis Metrics:
----------------
• R²: Coefficient of determination (0-1, higher is better)
• MSE: Mean Squared Error
• RMSE: Root Mean Squared Error
• MAE: Mean Absolute Error
• Normality Test: Check if residuals are normally distributed
• Durbin-Watson: Test for autocorrelation in residuals

Model Comparison:
----------------
1. Fit multiple models and click "Add to Comparison"
2. Click "Compare All Models" to view all fits together
3. Compare R², MSE, and other metrics

Tips:
-----
• Use at least n+1 data points for degree n polynomial
• Check residual plots for systematic patterns
• R² close to 1 indicates good fit
• Normally distributed residuals suggest good model
• Try multiple methods and compare results

Developer:
----------
Ezaz Ahmed (C223009)

Course: Numerical Methods
Institution: IIUC, CSE Department
"""
        
        help_text.insert(1.0, help_content)
        help_text.config(state='disabled')


def main():
    """Main entry point."""
    root = tk.Tk()
    app = CurveFittingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
