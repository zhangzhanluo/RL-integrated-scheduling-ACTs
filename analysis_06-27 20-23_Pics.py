"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20220630
    Description: 为IEEM会议整理数据和制图
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import scipy.stats as st
from matplotlib import pyplot as plt

folder = '01_Data/06-27 20-23/'
detail_result_path = folder + 'data.csv'
cut = 50000
detail_result = pd.read_csv(detail_result_path)[:cut]
plt.style.use(['00_Files/ijpr_fullwidth.mplstyle', '00_Files/ijpr_halfwidth.mplstyle'])

# %% 收敛性
detail_result_rolling = detail_result[:cut][['return']].rolling(window=1000, center=True).mean()
plt.figure()
plt.plot(detail_result_rolling['return'], c='#305497')
plt.xlabel('Epoch')
plt.ylabel('Average Return')
plt.annotate('rise sharply', xy=(1200, 0.885), xytext=(5200, 0.885),
            arrowprops=dict(arrowstyle="->"))
plt.annotate('fluctuate', xy=(5000, 0.99), xytext=(500, 1.08),
            arrowprops=dict(arrowstyle="->"))
plt.annotate('rise again', xy=(7500, 1), xytext=(12000, 0.98),
            arrowprops=dict(arrowstyle="->"))
plt.annotate('converge to a high level', xy=(40000, 1.08), xytext=(25000, 1.01),
            arrowprops=dict(arrowstyle="->"))
plt.subplots_adjust(left=0.14, bottom=0.2, right=0.98, top=0.99)
plt.savefig(folder + 'convergence curve.tiff')
plt.show()

# %% 泛化性
training_result_file = folder + 'rl_training_result.csv'
test_result_file = folder + 'rl_test_result.csv'
training_result = pd.read_csv(training_result_file)
training_result = training_result[training_result['epochs'] <= cut]
test_result = pd.read_csv(test_result_file)
test_result = test_result[test_result['epochs'] <= cut]
training_result.set_index('epochs', inplace=True, drop=True)
training_result.sort_index(inplace=True)
training_result['average'] = training_result.mean(axis=1)
test_result.set_index('epochs', inplace=True, drop=True)
test_result.sort_index(inplace=True)
test_result['average'] = test_result.mean(axis=1)

all_result = pd.concat([training_result['average'], test_result['average']], axis=1)
all_result.columns = ['training_instances', 'test_instances']
all_result = all_result/3600

y = all_result['test_instances'].values
x = all_result[['training_instances']]
x['intercept'] = 1
X = x.values
params = np.linalg.inv(X.T@X)@X.T@y
y_predict = X@params
r2 = r2_score(y, y_predict)

result = st.linregress(all_result['training_instances'], all_result['test_instances'])


plt.figure()
plt.scatter(all_result['training_instances'], all_result['test_instances'], s=7, c='#305497', label='Agent')
plt.xlabel('Makespan of Training Instances (h)')
plt.ylabel('Makespan of Test Instances (h)')
x_line = [2, 7.5]
y_line = [_x*params[0] + params[1] for _x in x_line]
plt.plot(x_line, y_line, c='#305497', label='Ordinary Least Squares Fitting')
plt.text(5.5, 5, 'y={:.2f}x{:.2f}'.format(params[0], params[1]))
plt.legend(fontsize=8, frameon=False)
plt.subplots_adjust(left=0.13, bottom=0.2, right=0.98, top=0.99)
plt.savefig(folder + 'train test comparison.tiff')
plt.show()


# %% 整理数据
rule_training_result_file = folder+'rule_training_result.csv'
rule_test_result_file = folder + 'rule_test_result.csv'
rule_test_result = pd.read_csv(rule_test_result_file, index_col=0)
rule_training_result = pd.read_csv(rule_training_result_file, index_col=0)
training_idx = training_result['average'].idxmin()
all_training_result = rule_training_result.T.join(training_result.T[training_idx])
all_training_result.sort_index(inplace=True)
all_training_result = all_training_result.T
all_training_result['average'] = all_training_result.mean(axis=1)
all_training_result = all_training_result.T
all_test_result = rule_test_result.T.join(test_result.T[training_idx])
all_test_result.sort_index(inplace=True)
all_test_result = all_test_result.T
all_test_result['average'] = all_test_result.mean(axis=1)
all_test_result = all_test_result.T

all_training_result = all_training_result / 3600
all_training_result = all_training_result.round(2)
all_training_result.to_csv(folder + 'all_training_result.csv')
all_test_result = all_test_result / 3600
all_test_result = all_test_result.round(2)
all_test_result.to_csv(folder + 'all_test_result.csv')
