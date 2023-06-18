import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Add the directory containing poselib to the module search path
sys.path.append('EXTERNALS')

import lamda_pnp_solver
