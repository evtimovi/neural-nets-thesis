from util import performance as perf
from util import processimgs as pimg
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import random
import sys
from vggface import networks as vggn

network = vggn.VGGFaceMEBWithL2(
