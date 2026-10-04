import os
import sys

# Make `import api...` work when pytest is run from server/ or the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
