# Standalone version of the task 9 Jupyter Notebook
# Requires: Numpy, Matplot, Pandas, Seaborn 

DATA_SET_PATH = "CMAPSSData/"

PLOT_DATA_SET = False
PLOT_MODEL_TRAIN = True

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

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve


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


# ...add more visualization here!


# - - - - - - - - - - - - - - - - -
#            MODELING
# - - - - - - - - - - - - - - - - -

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


print("Modelling with Random Forest Regressor... (this takes a while)")

# A: Random Forest with default Hyper parameters
# tune maxfeatures
# rf_model = RandomForestRegressor(n_estimators=500, max_features=3)  # 3 was best based on grid search

# param_grid = {'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 26, 27, 30, 35, 40]}

# search = GridSearchCV(rf_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=5, n_jobs=-1)
# search.fit(x,y)

# print("*********************************************************************************")
# print("*********************************************************************************")
# print("=== RESULTS ===\n", search.cv_results_)
# print("=== BEST ===\n", search.best_params_)

# exit(1)

# rf_model.fit(x,y)
# rf_prediction = rf_model.predict(train_x)

# if PLOT_MODEL_TRAIN:
#     plt.plot(rf_prediction[:500], label="Prediction Random Forest Default")
#     # plt.plot(train_FD001["rul"][:500], label="Train RUL")
#     plt.legend(loc="upper left")
#     # plt.show()

print("Modelling with Lasso model...")

# B: Lasso model with default Hyper parameters
ls_model = LassoCV(alphas=[0.05, 0.2, 0.5, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 4.0, 6.0])

# param_grid = {'alpha': [0.05, 0.2, 0.5, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 4.0, 6.0]}

# search = GridSearchCV(ls_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=5, n_jobs=-1)
# search.fit(x,y)

# print("*********************************************************************************")
# print("*********************************************************************************")
# print("=== RESULTS ===\n", search.cv_results_)
# print("=== BEST ===\n", search.best_params_)

# exit(1)


ls_model.fit(x,y)
ls_prediction = ls_model.predict(train_x)

# C: Logistic Regression
# lr_model = LogisticRegression(random_state=0)
# lr_model.fit(x,y)
# lr_prediction = lr_model.predict(train_x)

# D: MLP Regressor
print("Modelling with MLP Regressor... (this takes a while)")
mlp_model = MLPRegressor(hidden_layer_sizes=(120), activation='relu', solver='adam', alpha=0.001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100, shuffle=True, random_state=4, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

# param_grid = {'hidden_layer_sizes': [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]}
# param_grid = [
#         {
#             'activation' : ['identity', 'logistic', 'tanh', 'relu'],
#             'solver' : ['lbfgs', 'sgd', 'adam'],
#             'hidden_layer_sizes': [
#              (10,),(20,),(30,),(40,),(50,),(60,),(70,),(80,),(90,),(100,),(110,), (120,),(130,),(140,)
#              ]
#         }
#        ]

# param_grid = {'random_state': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30, 35, 40, 41, 42]}
# param_grid = [
#         {
#             'activation' : ['identity', 'logistic', 'tanh', 'relu'],
#             'solver' : ['lbfgs', 'sgd', 'adam'],
#             'hidden_layer_sizes': [
#              (10,),(20,),(30,),(40,),(50,),(60,),(70,),(80,),(90,),(100,),(110,), (120,),(130,),(140,)
#              ]
#         }
#        ]


# param_grid = [
#     {'learning_rate_init': [0.01, 0.001, 0.0001, 0.00001],
#     'max_iter': [100, 300, 600, 1000]}
# ]



# search = GridSearchCV(mlp_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=5, n_jobs=-1)
# search.fit(x,y)

# print("*********************************************************************************")
# print("*********************************************************************************")
# print("=== RESULTS ===\n", search.cv_results_)
# print("=== BEST ===\n", search.best_params_)

# exit(1)

# train_sizes, train_scores, valid_scores = learning_curve(mlp_model, x, y, max_iter=[200, 500, 1000], cv=5)
# plt.plot(training_sizes, train_scores)

mlp_model.fit(x,y)
mlp_prediction = mlp_model.predict(train_x)

# loss_values = mlp_model.estimator.loss_curve_
# plt.plot(loss_values)

# if PLOT_MODEL_TRAIN:
#     plt.plot(ls_prediction[:500], label="Prediction Lasso CV Default")
#     plt.plot(train_FD001["rul"][:500], label="Train RUL")
#     # plt.plot(lr_prediction[:500], label="Prediction Logistic Regression Default")
#     plt.plot(mlp_prediction[:500], label="Prediction MLP Default")

#     plt.legend(loc="upper left")
#     plt.show()

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

# # A: Random forest
# rf_test_prediction = rf_model.predict(test_input)
# rf_rmse = np.sqrt(mean_squared_error(rf_test_prediction, RUL_FD001.values.reshape(-1)))
# print("The RMSE of random forest on test dataset FD001 is ", rf_rmse)

# B: Lasso model
ls_test_prediction = ls_model.predict(test_input)
ls_rmse = np.sqrt(mean_squared_error(ls_test_prediction, RUL_FD001.values.reshape(-1)))
print("The RMSE of Lasso model on test dataset FD001 is ", ls_rmse)


# D: MLP
mlp_test_prediction = mlp_model.predict(test_input)
mlp_rmse = np.sqrt(mean_squared_error(mlp_test_prediction, RUL_FD001.values.reshape(-1)))
print("The RMSE of MLP model on test dataset FD001 is ", mlp_rmse)

# best = 100000
# nbest = -1

# for k in range(2, 100):
#     results = []

#     # Perform 10 times and get average result
#     for i in range(5):
#         x, y = shuffle(train_x, train_y)

#         # Train
#         rf_model = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=int(k), min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
#         rf_model.fit(x,y)
#         rf_prediction = rf_model.predict(train_x)
     
#         # Eval
#         test_x['engine_id'] = test_FD001['engine_id']
#         test_input = []
#         for id in test_x['engine_id'].unique():
#             test_input.append(test_x[test_x['engine_id']==id].iloc[-1,:-1].values)

#         test_input = np.array(test_input)

#         rf_test_prediction = rf_model.predict(test_input)
#         rf_rmse = np.sqrt(mean_squared_error(rf_test_prediction, RUL_FD001.values.reshape(-1)))
#         print("The RMSE of random forest", k, "on test dataset FD001 is ", rf_rmse)

#         results.append(rf_rmse)

#     average = sum(results) / float(len(results))
#     print("-> Average for n=", k, ":", average)

#     if average < best:
#         best = average
#         nbest = k

# print("Best:", nbest, "@", best)

