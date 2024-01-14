import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def rmse(y_estimate,y_true):
    return math.sqrt(mean_squared_error(y_estimate,y_true))

def adj_R2(y_estimate,y_true,p):
    return 1-((1-r2_score(y_estimate,y_true))*(len(y_estimate)-1)/(len(y_estimate)-1-p))

def R2(y_estimate,y_true):
    return r2_score(y_pred=y_estimate,y_true=y_true)

def mae(y_estimate,y_true):
    return  mean_absolute_error(y_estimate,y_true)
y_e=[1,2,3]
y_t=[3,2,1]
#回归函数拟合效果差于取平均值时R2会为负数

