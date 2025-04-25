import sys
import os

# Add the site-packages directory to the Python path
site_packages = '/Users/jgh407/.pyenv/versions/3.12.0/lib/python3.12/site-packages'
if site_packages not in sys.path:
    sys.path.append(site_packages)

try:
    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    coef_list_test = [0, 0.5]
    ang_seq = QuantumSignalProcessingPhases(coef_list_test, signal_operator="Wz")
    print(ang_seq)
except ImportError as e:
    print(f"Error importing pyqsp: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")