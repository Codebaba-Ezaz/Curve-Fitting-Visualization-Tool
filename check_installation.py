

import sys
import importlib

print("=" * 60)
print("Curve Fitting Tool - Installation Verification")
print("Ezaz Ahmed (C223009) - Numerical Methods Project")
print("=" * 60)
print()

# Check Python version
print("1. Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} (OK)")
else:
    print(f"   ✗ Python {sys.version_info.major}.{sys.version_info.minor} (Need 3.8+)")
    sys.exit(1)

print()

# Check required packages
required_packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'pandas': 'Pandas',
    'PIL': 'Pillow',
    'tkinter': 'Tkinter'
}

print("2. Checking required packages...")
all_installed = True

for module_name, display_name in required_packages.items():
    try:
        if module_name == 'tkinter':
            import tkinter
        else:
            module = importlib.import_module(module_name)
            if hasattr(module, '__version__'):
                version = module.__version__
            else:
                version = 'installed'
        
        print(f"   ✓ {display_name}: {version}")
    except ImportError:
        print(f"   ✗ {display_name}: NOT INSTALLED")
        all_installed = False

print()

# Check project files
print("3. Checking project files...")
import os

required_files = [
    'main.py',
    'fitting_methods.py',
    'data_handler.py',
    'error_metrics.py',
    'visualization.py',
    'requirements.txt',
    'README.md'
]

for filename in required_files:
    if os.path.exists(filename):
        print(f"   ✓ {filename}")
    else:
        print(f"   ✗ {filename} - MISSING")
        all_installed = False

print()

# Check sample data
print("4. Checking sample data...")
sample_dir = 'sample_data'
if os.path.exists(sample_dir):
    sample_files = os.listdir(sample_dir)
    print(f"   ✓ sample_data folder ({len(sample_files)} files)")
    for f in sample_files:
        print(f"      - {f}")
else:
    print(f"   ⚠ sample_data folder missing (optional)")

print()

# Final verdict
print("=" * 60)
if all_installed:
    print("✓ INSTALLATION COMPLETE")
    print()
    print("All dependencies are installed correctly!")
    print()
    print("To run the application:")
    print("    python main.py")
    print()
else:
    print("✗ INSTALLATION INCOMPLETE")
    print()
    print("Some dependencies are missing.")
    print()
    print("To install missing packages:")
    print("    pip install -r requirements.txt")
    print()

print("=" * 60)
