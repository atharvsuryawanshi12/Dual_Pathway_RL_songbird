import json 
import os
import numpy as np 
import matplotlib.pyplot as plt
from numpy import core
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from functions import *
from model import NN
from env import build_and_run

# load parameters from json file
params_path = "JSON\params.json"
# Open the file and read the contents
with open(params_path, "r") as f:
    parameters = json.load(f)

# running conditions
N_SYLL = parameters['run']['N_SYLL']
ANNEALING = parameters['modes']['ANNEALING']

if N_SYLL > 5 or N_SYLL < 1:
    ValueError('Invalid number of syllables')

RANDOM_SEED = 40 #np.random.randint(0, 1000)
print(f'Random seed is {RANDOM_SEED}')
np.random.seed(RANDOM_SEED)

build_and_run(RANDOM_SEED, annealing = True, plot = True, parameters = parameters, NN = NN)
