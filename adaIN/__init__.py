import os
import sys

current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)


if module_path not in sys.path:
    sys.path.append(module_path)
