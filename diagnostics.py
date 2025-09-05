import sys, traceback

print("Python:", sys.version)
try:
    import numpy as np
    print("numpy:", np.__version__)
except Exception as e:
    print("numpy import failed:", e)

qt = None
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    qt = "PyQt5"
    print("Using PyQt5")
except Exception as e1:
    print("PyQt5 import failed:", e1)
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        qt = "PySide6"
        print("Using PySide6")
    except Exception as e2:
        print("PySide6 import failed:", e2)
        print("\nNo Qt bindings installed. Install one of:")
        print("  pip install PyQt5 matplotlib numpy")
        print("  or")
        print("  pip install PySide6 matplotlib numpy")
        sys.exit(1)

try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
    # Force Qt backend
    try:
        matplotlib.use("QtAgg", force=True)
    except Exception as e:
        print("Could not set QtAgg backend:", e)
    import matplotlib.pyplot as plt
    print("matplotlib backend:", matplotlib.get_backend())
except Exception as e:
    print("matplotlib import failed:", e)

# Try creating a minimal Qt app and window (no display shown)
try:
    app = QtWidgets.QApplication([])
    w = QtWidgets.QWidget()
    w.setWindowTitle("EMTAS-4 Diagnostics")
    print("\nQt can create a QApplication successfully. GUI should run.")
    # Don't start the event loop here
except Exception as e:
    print("\nQt failed to create QApplication:", e)
    traceback.print_exc()
    sys.exit(2)

print("\nDiagnostics completed successfully.")
