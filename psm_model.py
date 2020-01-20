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

cols_pur = pur_data.columns.tolist()
x_col,y_col = cols_pur[:-1],cols_pur[-1:]

treated = pur_data[pur_data['sales'] == 1] #实验组数据
control = pur_data[pur_data['sales'] == 0] #对照组数据

treated_sample=treated.reset_index(drop=True) #实验组
control_sample=control.reset_index(drop=True) #对照组(数据量必须大于实验组数据量)

# PSM前对照组和实验组的差异
# treated.describe().round(2)   
# control.describe().round(2)

pur_combine = control_sample.append(treated_sample).reset_index(drop=True)#两个筛选的数据集为回归基础数据
y_f,x_f = patsy.dmatrices('{} ~ {}'.format(y_col[0], '+'.join(x_col)), data=pur_combine, return_type='dataframe')
formula = '{} ~ {}'.format(y_col[0], '+'.join(x_col))
print('Formula:\n' + formula)
print('Majority Lenth: ', len(control_sample))
print('Minority Lenth: ', len(treated_sample)) #确定回归方程

#output: Formula:
#output: sales ~ season+gender+type+price+style+goods+fabric+design+shape+pattern
#output: Majority Lenth:  2019
#output: Minority Lenth:  175

i, errors = 0, 0 
nmodels=20 #可指定归回模型个数 
models, model_accuracy = [], [] #模型保存,模型准确性保存
# coef = []
while i < nmodels and errors < 5:
    sys.stdout.write('\r{}: {}\{}'.format("Fitting Models on Balanced Samples", i+1, nmodels)) #第几个模型     
    pur_c = control_sample.sample(len(treated_sample)).append(treated_sample).dropna().reset_index(drop=True) #模型选择相同的对照组和控制组样本
    y_samp, X_samp = patsy.dmatrices(formula, data=pur_c, return_type='dataframe') #选出模型的自变量和因变量
    glm = GLM(y_samp, X_samp, family=sm.families.Binomial()) #逻辑回归模型
    try:
        res = glm.fit()
        # coef.append(res.params.to_dict())
        # results = sm.OLS(y_samp,X_samp).fit()
        preds = [1.0 if i >= .5 else 0.0 for i in res.predict(X_samp)]
        preds = pd.DataFrame(preds,columns=y_samp.columns)
        b=y_samp.reset_index(drop=True)
        a=preds.reset_index(drop=True)
        ab_score=((a.sort_index().sort_index(axis=1) == b.sort_index().sort_index(axis=1)).sum() * 1.0 / len(y_samp)).values[0] # 模型预测准确性得分
        model_accuracy.append(ab_score)
        models.append(res)
        i += 1
    except Exception as e:
        errors += 1
        print('Error: {}'.format(e))

print("\nAverage Accuracy:", "{}%". format(round(np.mean(model_accuracy) * 100, 2))) # 所有模型的平均准确性

pur_combine['scores'] =  [i for i in res.predict(x_f)]

threshold, nmatches, method = 0.001, 1, 'min'

test_scores = pur_combine[pur_combine[y_col[0]]==True][['scores']]
ctrl_scores = pur_combine[pur_combine[y_col[0]]==False][['scores']]
result, match_ids = [], []
for i in range(len(test_scores)):
    match_id = i
    score = test_scores.iloc[i]
    matches = abs(ctrl_scores - score).sort_values('scores').head(nmatches)
    chosen = np.random.choice(matches.index, nmatches, replace=False)
    result.extend([test_scores.index[i]] + list(chosen))
    match_ids.extend([i] * (len(chosen)+1))
    ctrl_scores=ctrl_scores.drop(chosen,axis=0)
matched_data =pur_combine.loc[result]
matched_data['match_id'] = match_ids
matched_data['record_id'] = matched_data.index
# matched_data
