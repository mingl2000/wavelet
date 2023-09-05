from mySSA import mySSA
import pandas as pd
import numpy as np
from YahooData import *
import matplotlib.pyplot as plt
def  SSACaterpillar(df):
    ssa = mySSA(df)
    K = len(df)
    suspected_seasonality = 5

    ssa.embed(embedding_dimension=K, suspected_frequency=suspected_seasonality, verbose=True)
    ssa.decompose(verbose=True)
    return ssa.Xs



