import numpy as np
import random
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
from collections import OrderedDict
from dataclasses import dataclass
import pickle
import gzip
import os
import cv2
import glob
import matplotlib as mpl
import tkinter as tk
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import json
import matplotlib.ticker as ticker
from enum import Enum, auto
import copy
import re
import subprocess
import cv2
import traceback
import math
from math import cos, pi
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from typing import List, Dict
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import functools
import sys
import bisect
import threading
import pprint
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from itertools import combinations, permutations
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from data_class import *

script_dir = os.path.dirname(os.path.realpath(__file__))
image_files = sorted(glob.glob(f"{script_dir}/../../../../dataset/blob*.bmp"))
print(f"image_files length {len(image_files)}")
frame_cnt = 0
while True:
    if frame_cnt >= len(image_files):
        break
    frame_0 = cv2.imread(image_files[frame_cnt])
    filename = f"IMAGE Mode {os.path.basename(image_files[frame_cnt])}"
    if frame_0 is None or frame_0.size == 0:
        print(f"Failed to load {image_files[frame_cnt]}, frame_cnt:{frame_cnt}")
        continue
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('n'):                
        frame_cnt += 1
    elif key & 0xFF == ord('b'):
        frame_cnt -= 1
    cv2.imshow('IMAGE', frame_0)

cv2.destroyAllWindows()
