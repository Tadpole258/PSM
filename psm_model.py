#!/usr/bin/env python3

import sys
import json
import pandas as pd
import numpy as np
import patsy
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm 
import seaborn as sns

import warnings

%matplotlib inline
warnings.filterwarnings('ignore')

#万能读取文件的方式
for decode in ('gbk','utf-8','gb18030'):
    try:    
        pur_data = pd.read_csv('sales_purchase.csv',encoding=decode,error_bad_lines=False,index_col=0)
        # pd_info.drop(['Unnamed: 0'],axis=1,inplace=True)
        print('data-' +  decode + '-success!!')
        break
    except:
        pass

"""
#字段相关系数热力图
pur_corr = pur_data.corr()
plt.subplots(figsize=(12, 12)) # 设置画面大小
sns.heatmap(pur_corr, annot=True, vmax=1, square=True, cmap="Reds")
plt.show()
""""
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
