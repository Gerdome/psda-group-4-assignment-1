# Standalone version of the task 9 Jupyter Notebook
# Requires: Numpy, Matplot, Pandas, Seaborn 

DATA_SET_PATH = "CMAPSSData/"

PLOT_DATA_SET = True
PLOT_MODEL_TRAIN = False

# Import basic packages
import pandas as pd
import seaborn as sns
import numpy as np
sns.set()

# Set up matplot
import matplotlib.pyplot as plt
plt.rcParams['ytick.labelsize'] = "x-large"
plt.rcParams['xtick.labelsize'] = "x-large"
plt.rcParams['axes.labelsize'] = "x-large"
plt.rcParams['figure.titlesize'] = "x-large"

# Import some scikit objects
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error


# - - - - - - - - - - - - - - - - -
#  DATA SET LOADING / EXPLORATION
# - - - - - - - - - - - - - - - - -

# Load some data from the data set
Path_to_data = DATA_SET_PATH
column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                's15', 's16', 's17', 's18', 's19', 's20', 's21']    # Read moren on this in the suppementray description file I added

# A: Training data set
train_FD001 = pd.read_table(Path_to_data + "train_FD001.txt", header=None, delim_whitespace=True)
train_FD001.columns = column_name

# B: Test data set
test_FD001 = pd.read_table(Path_to_data + "test_FD001.txt", header=None, delim_whitespace=True)
test_FD001.columns = column_name

# And: RUL for test data set
RUL_FD001 = pd.read_table(Path_to_data + "RUL_FD001.txt", header=None, delim_whitespace=True)
train_FD001.head()

def add_RUL(col):
    # Reverse the cycle evolution, where remaining time of a machine is 0 at the failure.
    # It is assumed here that the state of the machine is linearly deteriorating
    return col[::-1]-1

# Calculate RUL for each time point of each engine from the training data
train_FD001['rul'] = train_FD001[['engine_id', 'cycle']].groupby('engine_id').transform(add_RUL)

if PLOT_DATA_SET:
    # Visualize the RUL curve of some engines (1,2,3,4,5,6)
    g = sns.PairGrid(data=train_FD001.reset_index().query('engine_id < 7'),
                        x_vars=["index"],
                        y_vars=['rul'],
                        hue="engine_id", height=3, aspect=2.5)

    g = g.map(plt.plot, alpha=1)
    g = g.add_legend()

    # Visualize some sensor curves of some engines 
    g = sns.PairGrid(data=train_FD001.query('engine_id < 5') ,
                        x_vars=["rul"],
                        y_vars=['s1','s2'],
                        hue="engine_id", height=3, aspect=2.5)

    g = g.map(plt.plot, alpha=1)
    g = g.add_legend()
    plt.show()

# As shown in the figure, some sensors are not related to RUL. 
# The values of some sensors change with the state of the machine. 
# Visualization can help filter features

# Distribution of maximum life cycle
if PLOT_DATA_SET:
    train_FD001[['engine_id', 'rul']].groupby('engine_id').apply(np.max)["rul"].hist(bins=20)

    plt.xlabel("max life cycle")
    plt.ylabel("Count")
    plt.show()

# Prepare the data and normalization
train_y = train_FD001['rul']
features = train_FD001.columns.drop(['engine_id', 'cycle', 'rul'])
train_x = train_FD001[features]
test_x = test_FD001[features]

# z score normalization
mean = train_x.mean()
std = train_x.std()
std.replace(0, 1, inplace=True)

train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

from sklearn.utils import shuffle
x, y = shuffle(train_x, train_y)


# ...add more visualization here!


# - - - - - - - - - - - - - - - - -
#            MODELING
# - - - - - - - - - - - - - - - - -

print("Modelling with Random Forest Regressor... (this takes a while)")

# A: Random Forest with default Hyper parameters
rf_model = RandomForestRegressor()
rf_model.fit(x,y)
rf_prediction = rf_model.predict(train_x)

if PLOT_MODEL_TRAIN:
    plt.plot(rf_prediction[:500])
    plt.plot(train_FD001["rul"][:500])
    plt.show()

print("Modelling with Lasso model...")

# B: Lasso model with default Hyper parameters
ls_model = LassoCV()
ls_model.fit(x,y)
ls_prediction = ls_model.predict(train_x)

if PLOT_MODEL_TRAIN:
    plt.plot(ls_prediction[:500])
    plt.plot(train_FD001["rul"][:500])
    plt.show()


# ...add more models here! Hyper parameter tuning, neural nets etc.


# - - - - - - - - - - - - - - - - -
#        EVALUATION OF MODELS
# - - - - - - - - - - - - - - - - -

# Since only the value at one time point is used, it can be seen that a lot of data in the test set is not used
test_x['engine_id'] = test_FD001['engine_id']
test_input = []
for id in test_x['engine_id'].unique():
    test_input.append(test_x[test_x['engine_id']==id].iloc[-1,:-1].values)

test_input = np.array(test_input)

# A: Random forest
rf_test_prediction = rf_model.predict(test_input)
rf_rmse = np.sqrt(mean_squared_error(rf_test_prediction, RUL_FD001.values.reshape(-1)))
print("The RMSE of random forest on test dataset FD001 is ", rf_rmse)

# B: Lasso model
ls_test_prediction = ls_model.predict(test_input)
ls_rmse = np.sqrt(mean_squared_error(ls_test_prediction, RUL_FD001.values.reshape(-1)))
print("The RMSE of Lasso model on test dataset FD001 is ", ls_rmse)