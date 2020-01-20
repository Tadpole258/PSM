import warnings 
import pandas as pd
import numpy as np
import json
import patsy
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm 
import seaborn as sns
import sys

%matplotlib inline
warnings.filterwarnings('ignore')


x_col = pd_bin.columns.tolist()[:-1]
print(x_col)
y_col = pd_bin.columns.tolist()[-1:]
print(y_col)
treated = pd_bin[pd_bin['sales'] == 1] #实验组数据
control = pd_bin[pd_bin['sales'] == 0] #对照组数据

treated.describe().round(2)   #PSM前对照组和实验组的差异
control.describe().round(2)

#pd_bin.groupby(['sales']).size()  #原数据样本量大小
treated_sample=treated.reset_index(drop=True) # 实验组
control_sample=control.reset_index(drop=True) #对照组，数据量必须大于实验组数据量
