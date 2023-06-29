import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


diamonds = sns.load_dataset("diamonds")
diamonds.head()
