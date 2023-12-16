import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import warnings
# 过滤所有警告
warnings.filterwarnings("ignore")


# 读取并合并ff5数据
data1 = pd.read_csv("factor_result/2016FF5.csv")
data2 = pd.read_csv("factor_result/2017FF5.csv")
data_ff5 = pd.concat([data1,data2], ignore_index=True, sort=True)
data_ff5 = data_ff5.reset_index(drop=True)
# 读取分组超额收益率数据
for fac in ['BM', 'Inv', 'OP']:
    data1 = pd.read_csv("group_result/2016_" + fac + "_Size.csv")
    data2 = pd.read_csv("group_result/2017_" + fac + "_Size.csv")
    if fac == 'BM':
        Size_BM = pd.concat([data1,data2], ignore_index=True, sort=True)
    elif fac == 'Inv':
        Size_Inv = pd.concat([data1,data2], ignore_index=True, sort=True)
    elif fac == 'OP':
        Size_OP = pd.concat([data1,data2], ignore_index=True, sort=True)
# 对BM_Size分组进行五因子回归
regression_BM = pd.DataFrame(columns=['Size', 'BM', 'Const', 'Mktrf', 'SMB', 'HML', 'RMW', 'CMA', 'R-squared'])
regression_BM_pvalue = pd.DataFrame(columns=['Size', 'BM', 'Const_p', 'Mktrf_p', 'SMB_p', 'HML_p', 'RMW_p', 'CMA_p'])
regression_BM_tvalue = pd.DataFrame(columns=['Size', 'OP', 'Const_t', 'Mktrf_t', 'SMB_t', 'HML_t', 'RMW_t', 'CMA_t'])
x_ = Size_BM['row_count'].value_counts().shape[0]
y_ = Size_BM['col_count'].value_counts().shape[0]
for i in range(1, x_ + 1):
    for j in range(1, y_ + 1):
        X = data_ff5.loc[0:19, ['CMA', 'HML', 'Mktrf', 'RMW', 'SMB']]
        X = sm.add_constant(X)
        y = Size_BM[(Size_BM['row_count'] == i) & (Size_BM['col_count'] == j)].loc[:, 'ExcessReturn']
        y = y.reset_index(drop=True)
        model = sm.OLS(y, X)
        results = model.fit()
        regression_BM.loc[len(regression_BM)] = pd.Series({'Size': i, 'BM': j, 'Const': results.params[0],
                                               'Mktrf': results.params[1], 'SMB': results.params[2],
                                               'HML': results.params[3], 'RMW': results.params[4],
                                               'CMA': results.params[5], 'R-squared': results.rsquared})
        regression_BM_pvalue.loc[len(regression_BM_pvalue)] = pd.Series({'Size': i, 'BM': j, 'Const_p': results.pvalues[0],
                                                             'Mktrf_p': results.pvalues[1], 'SMB_p': results.pvalues[2],
                                                             'HML_p': results.pvalues[3], 'RMW_p': results.pvalues[4],
                                                             'CMA_p': results.pvalues[5]})
        regression_BM_tvalue.loc[len(regression_BM_tvalue)] = pd.Series({'Size': i, 'OP': j, 'Const_t': results.tvalues[0],
                                                                         'Mktrf_t': results.tvalues[1], 'SMB_t': results.tvalues[2],
                                                                         'HML_t': results.tvalues[3], 'RMW_t': results.tvalues[4],
                                                                         'CMA_t': results.tvalues[5]})
# 对Inv_Size分组进行五因子回归
regression_Inv = pd.DataFrame(columns=['Size', 'Inv', 'Const', 'Mktrf', 'SMB', 'HML', 'RMW', 'CMA', 'R-squared'])
regression_Inv_pvalue = pd.DataFrame(columns=['Size', 'Inv', 'Const_p', 'Mktrf_p', 'SMB_p', 'HML_p', 'RMW_p', 'CMA_p'])
regression_Inv_tvalue = pd.DataFrame(columns=['Size', 'OP', 'Const_t', 'Mktrf_t', 'SMB_t', 'HML_t', 'RMW_t', 'CMA_t'])
x_ = Size_Inv['row_count'].value_counts().shape[0]
y_ = Size_Inv['col_count'].value_counts().shape[0]
for i in range(1, x_ + 1):
    for j in range(1, y_ + 1):
        X = data_ff5.loc[0:19, ['CMA', 'HML', 'Mktrf', 'RMW', 'SMB']]
        X = sm.add_constant(X)
        y = Size_Inv[(Size_Inv['row_count'] == i) & (Size_Inv['col_count'] == j)].loc[:, 'ExcessReturn']
        y = y.reset_index(drop=True)
        model = sm.OLS(y, X)
        results = model.fit()
        regression_Inv.loc[len(regression_Inv)] = pd.Series({'Size': i, 'Inv': j, 'Const': results.params[0],
                                                 'Mktrf': results.params[1], 'SMB': results.params[2],
                                                 'HML': results.params[3], 'RMW': results.params[4],
                                                 'CMA': results.params[5], 'R-squared': results.rsquared})
        regression_Inv_pvalue.loc[len(regression_Inv_pvalue)] = pd.Series({'Size': i, 'Inv': j, 'Const_p': results.pvalues[0],
                                                               'Mktrf_p': results.pvalues[1],
                                                               'SMB_p': results.pvalues[2], 'HML_p': results.pvalues[3],
                                                               'RMW_p': results.pvalues[4],
                                                               'CMA_p': results.pvalues[5]})
        regression_Inv_tvalue.loc[len(regression_Inv_tvalue)] = pd.Series({'Size': i, 'OP': j, 'Const_t': results.tvalues[0],
                                                                     'Mktrf_t': results.tvalues[1], 'SMB_t': results.tvalues[2],
                                                                     'HML_t': results.tvalues[3], 'RMW_t': results.tvalues[4],
                                                                     'CMA_t': results.tvalues[5]})
# 对OP_Size分组进行五因子回归
regression_OP = pd.DataFrame(columns=['Size', 'OP', 'Const', 'Mktrf', 'SMB', 'HML', 'RMW', 'CMA', 'R-squared'])
regression_OP_pvalue = pd.DataFrame(columns=['Size', 'OP', 'Const_p', 'Mktrf_p', 'SMB_p', 'HML_p', 'RMW_p', 'CMA_p'])
regression_OP_tvalue = pd.DataFrame(columns=['Size', 'OP', 'Const_t', 'Mktrf_t', 'SMB_t', 'HML_t', 'RMW_t', 'CMA_t'])
x_ = Size_OP['row_count'].value_counts().shape[0]
y_ = Size_OP['col_count'].value_counts().shape[0]
for i in range(1, x_ + 1):
    for j in range(1, y_ + 1):
        X = data_ff5.loc[0:19, ['CMA', 'HML', 'Mktrf', 'RMW', 'SMB']]
        X = sm.add_constant(X)
        y = Size_OP[(Size_OP['row_count'] == i) & (Size_OP['col_count'] == j)].loc[:, 'ExcessReturn']
        y = y.reset_index(drop=True)
        model = sm.OLS(y, X)
        results = model.fit()
        regression_OP.loc[len(regression_OP)] = pd.Series({'Size': i, 'OP': j, 'Const': results.params[0],
                                               'Mktrf': results.params[1], 'SMB': results.params[2],
                                               'HML': results.params[3], 'RMW': results.params[4],
                                               'CMA': results.params[5], 'R-squared': results.rsquared})
        regression_OP_pvalue.loc[len(regression_OP_pvalue)] = pd.Series({'Size': i, 'OP': j, 'Const_p': results.pvalues[0],
                                                             'Mktrf_p': results.pvalues[1], 'SMB_p': results.pvalues[2],
                                                             'HML_p': results.pvalues[3], 'RMW_p': results.pvalues[4],
                                                             'CMA_p': results.pvalues[5]})
        regression_OP_tvalue.loc[len(regression_OP_tvalue)] = pd.Series({'Size': i, 'OP': j, 'Const_t': results.tvalues[0],
                                                                 'Mktrf_t': results.tvalues[1], 'SMB_t': results.tvalues[2],
                                                                 'HML_t': results.tvalues[3], 'RMW_t': results.tvalues[4],
                                                                 'CMA_t': results.tvalues[5]})

if not os.path.exists("regression_result/"):
    os.mkdir("regression_result/")
# 保存表
regression_BM.to_csv('regression_result/regression_BM.csv', index=False)
regression_BM_pvalue.to_csv('regression_result/regression_BM_pvalue.csv', index=False)
regression_BM_tvalue.to_csv('regression_result/regression_BM_tvalue.csv', index=False)
regression_Inv.to_csv('regression_result/regression_Inv.csv', index=False)
regression_Inv_pvalue.to_csv('regression_result/regression_Inv_pvalue.csv', index=False)
regression_Inv_tvalue.to_csv('regression_result/regression_Inv_tvalue.csv', index=False)
regression_OP.to_csv('regression_result/regression_OP.csv', index=False)
regression_OP_pvalue.to_csv('regression_result/regression_OP_pvalue.csv', index=False)
regression_OP_tvalue.to_csv('regression_result/regression_OP_tvalue.csv', index=False)


# 变量间相互回归
regression_factors = pd.DataFrame(columns=['DepVar','Const','Mktrf','SMB','HML','RMW','CMA','R-squared'])
regression_factors_tvalue = pd.DataFrame(columns=['DepVar','Const_t','Mktrf_t','SMB_t','HML_t','RMW_t','CMA_t'])
regression_factors_pvalue = pd.DataFrame(columns=['DepVar','Const_p','Mktrf_p','SMB_p','HML_p','RMW_p','CMA_p'])
# 被解释变量——Mktrf
y = data_ff5.loc[:,['Mktrf']]
X = data_ff5.loc[:,['SMB','HML','RMW','CMA']]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
regression_factors.loc[len(regression_factors)] = pd.Series({'DepVar': 'Mktrf','Const': results.params[0],
                                                 'Mktrf': 0, 'SMB': results.params[1],
                                                 'HML': results.params[2], 'RMW': results.params[3],
                                                 'CMA': results.params[4], 'R-squared': results.rsquared})
regression_factors_tvalue.loc[len(regression_factors_tvalue)] = pd.Series({'DepVar': 'Mktrf','Const_t': results.tvalues[0],
                                                               'Mktrf_t': 0, 'SMB_t': results.tvalues[1],
                                                               'HML_t': results.tvalues[2], 'RMW_t': results.tvalues[3],
                                                               'CMA_t': results.tvalues[4]})
regression_factors_pvalue.loc[len(regression_factors_pvalue)] = pd.Series({'DepVar': 'Mktrf','Const_p': results.pvalues[0],
                                                               'Mktrf_p': 0, 'SMB_p': results.pvalues[1],
                                                               'HML_p': results.pvalues[2], 'RMW_p': results.pvalues[3],
                                                               'CMA_p': results.pvalues[4]})
# 被解释变量——SMB
y = data_ff5.loc[:,['SMB']]
X = data_ff5.loc[:,['Mktrf','HML','RMW','CMA']]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
regression_factors.loc[len(regression_factors)] = pd.Series({'DepVar': 'SMB','Const': results.params[0],
                                                 'Mktrf': results.params[1], 'SMB': 0,
                                                 'HML': results.params[2], 'RMW': results.params[3],
                                                 'CMA': results.params[4], 'R-squared': results.rsquared})
regression_factors_tvalue.loc[len(regression_factors_tvalue)] = pd.Series({'DepVar': 'SMB','Const_t': results.tvalues[0],
                                                               'Mktrf_t': results.tvalues[1], 'SMB_t': 0,
                                                               'HML_t': results.tvalues[2], 'RMW_t': results.tvalues[3],
                                                               'CMA_t': results.tvalues[4]})
regression_factors_pvalue.loc[len(regression_factors_pvalue)] = pd.Series({'DepVar': 'SMB','Const_p': results.pvalues[0],
                                                               'Mktrf_p': results.pvalues[1], 'SMB_p': 0,
                                                               'HML_p': results.pvalues[2], 'RMW_p': results.pvalues[3],
                                                               'CMA_p': results.pvalues[4]})
# 被解释变量——HML
y = data_ff5.loc[:,['HML']]
X = data_ff5.loc[:,['Mktrf','SMB','RMW','CMA']]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
regression_factors.loc[len(regression_factors)] = pd.Series({'DepVar': 'HML','Const': results.params[0],
                                                 'Mktrf': results.params[1], 'SMB': results.params[2],
                                                 'HML': 0, 'RMW': results.params[3],
                                                 'CMA': results.params[4], 'R-squared': results.rsquared})
regression_factors_tvalue.loc[len(regression_factors_tvalue)] = pd.Series({'DepVar': 'HML','Const_t': results.tvalues[0],
                                                               'Mktrf_t': results.tvalues[1], 'SMB_t': results.tvalues[2],
                                                               'HML_t': 0, 'RMW_t': results.tvalues[3],
                                                               'CMA_t': results.tvalues[4]})
regression_factors_pvalue.loc[len(regression_factors_pvalue)] = pd.Series({'DepVar': 'HML','Const_p': results.pvalues[0],
                                                               'Mktrf_p': results.pvalues[1], 'SMB_p': results.pvalues[2],
                                                               'HML_p': 0, 'RMW_p': results.pvalues[3],
                                                               'CMA_p': results.pvalues[4]})
# 被解释变量——RMW
y = data_ff5.loc[:,['RMW']]
X = data_ff5.loc[:,['Mktrf','SMB','HML','CMA']]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
regression_factors.loc[len(regression_factors)] = pd.Series({'DepVar': 'RMW','Const': results.params[0],
                                                 'Mktrf': results.params[1], 'SMB': results.params[2],
                                                 'HML': results.params[3], 'RMW': 0,
                                                 'CMA': results.params[4], 'R-squared': results.rsquared})
regression_factors_tvalue.loc[len(regression_factors_tvalue)] = pd.Series({'DepVar': 'RMW','Const_t': results.tvalues[0],
                                                               'Mktrf_t': results.tvalues[1], 'SMB_t': results.tvalues[2],
                                                               'HML_t': results.tvalues[3], 'RMW_t': 0,
                                                               'CMA_t': results.tvalues[4]})
regression_factors_pvalue.loc[len(regression_factors_pvalue)] = pd.Series({'DepVar': 'RMW','Const_p': results.pvalues[0],
                                                               'Mktrf_p': results.pvalues[1], 'SMB_p': results.pvalues[2],
                                                               'HML_p': results.pvalues[3], 'RMW_p': 0,
                                                               'CMA_p': results.pvalues[4]})
# 被解释变量——CMA
y = data_ff5.loc[:,['CMA']]
X = data_ff5.loc[:,['Mktrf','SMB','HML','RMW']]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
regression_factors.loc[len(regression_factors)] = pd.Series({'DepVar': 'CMA','Const': results.params[0],
                                                 'Mktrf': results.params[0], 'SMB': results.params[1],
                                                 'HML': results.params[2], 'RMW': results.params[3],
                                                 'CMA': 0, 'R-squared': results.rsquared})
regression_factors_tvalue.loc[len(regression_factors_tvalue)] = pd.Series({'DepVar': 'SMB','Const_t': results.tvalues[0],
                                                               'Mktrf_t': results.tvalues[1], 'SMB_t': results.tvalues[2],
                                                               'HML_t': results.tvalues[3], 'RMW_t': results.tvalues[4],
                                                               'CMA_t': 0})
regression_factors_pvalue.loc[len(regression_factors_pvalue)] = pd.Series({'DepVar': 'SMB','Const_p': results.pvalues[0],
                                                               'Mktrf_p': results.pvalues[1], 'SMB_p': results.pvalues[2],
                                                               'HML_p': results.pvalues[3], 'RMW_p': results.pvalues[4],
                                                               'CMA_p': 0})

regression_factors.to_csv('regression_result/regression_factors.csv', index=False)
regression_factors_tvalue.to_csv('regression_result/regression_factors_tvalue.csv', index=False)
regression_factors_pvalue.to_csv('regression_result/regression_factors_pvalue.csv', index=False)