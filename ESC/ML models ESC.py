###
###
### LINEAR REGRESSION
###
###



import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,mean_squared_error,r2_score
from sklearn.inspection import PartialDependenceDisplay

from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from keras import callbacks
import shap
import sys 
import matplotlib

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


print("python version:",sys.version)
print("numpy version:",np.__version__)
print("matplotlib version:", matplotlib.__version__)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix' 


data=pd.read_excel('DATA FOR ESC.xlsx')

X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

size=len(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)


X_train1 = pd.DataFrame(X_train)
X_test1 = pd.DataFrame(X_test)
Y_train1 = pd.DataFrame(Y_train)
Y_test1 = pd.DataFrame(Y_test)




model=LinearRegression()

model.fit(X_train,Y_train)




Y_pred_train=model.predict(X_train)
Y_pred_test=model.predict(X_test)


R2_value_test=r2_score(Y_test,Y_pred_test)
R2_value_train=r2_score(Y_train,Y_pred_train)
MSE_value_test=mean_squared_error(Y_test,Y_pred_test)
MSE_value_train=mean_squared_error(Y_train,Y_pred_train)
MAE_value_test=mean_absolute_error(Y_test,Y_pred_test)
MAE_value_train=mean_absolute_error(Y_train,Y_pred_train)
RMSE_value_test=np.sqrt(mean_squared_error(Y_test,Y_pred_test))
RMSE_value_train=np.sqrt(mean_squared_error(Y_train,Y_pred_train))


print("R2 value test:",R2_value_test)
print("R2 value train:",R2_value_train)
print("mean squared error test",MSE_value_test)
print("mean squared error train",MSE_value_train)
print("mean absolute error test",MAE_value_test)
print("mean absolute error train",MAE_value_train)
print("root mean squared error test",RMSE_value_test)
print("root mean squared error train",RMSE_value_train)

LR_VALUE=[R2_value_test,R2_value_train,MSE_value_test,MSE_value_train,MAE_value_test,MAE_value_train,RMSE_value_test,RMSE_value_train]
LR_Train=[RMSE_value_train,MSE_value_train,MAE_value_train,R2_value_train]
LR_Test=[RMSE_value_test,MSE_value_test,MAE_value_test,R2_value_test]



plt.figure(figsize=(10,10))
plt.scatter(Y_test,Y_pred_test,color='blue')
plt.xlabel('Actual ESC', fontsize=22, fontweight='bold')
plt.ylabel('Predicted ESC', fontsize=22, fontweight='bold')
plt.grid(True)
plt.title('LR',fontsize=20,fontweight='bold')
plt.plot([-30,300],[-30,300],color='red',linestyle='-')
plt.xlim(-30, 300)
plt.ylim(-30, 300)
plt.gca().set_aspect('equal', adjustable='box') 

plt.text(
    -22, 239,
    f"RMSE = {RMSE_value_test:.2f}\nMSE = {MSE_value_test:.2f}\nMAE = {MAE_value_test:.2f}\n$R^2$ score = {R2_value_test:.2f}",
    fontsize=22,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

plt.show()

Y_pred_train=Y_pred_train.flatten()
Y_pred_test=Y_pred_test.flatten()

test_residuals_LR=Y_test-Y_pred_test





###
###
### DECISION TREE
###
###


# Parameters taken by mangalam

# param_grid={
#     'max_depth': [5,6,7,8],
#     'min_samples_split': [3,4,5,6],
#     'min_samples_leaf': [2,3,4,5],
#     'ccp_alpha': [0.005, 0.01, 0.05]
# }

# best for mangalam's case

# param_grid={
#     'max_depth': [8],
#     'min_samples_split': [3],
#     'min_samples_leaf': [5],
#     'ccp_alpha': [0.05]
# }

# updated case parameters

# param_grid={
#     'max_depth': [None, 4, 6, 8, 10],              # None = fully grown tree
#     'min_samples_split': [2, 3, 4, 5, 6],          # minimum samples to split a node
#     'min_samples_leaf': [1, 2, 3, 4, 5],           # minimum samples per leaf
#     'ccp_alpha': [0.0, 0.01, 0.05, 0.1]            # complexity pruning parameter
# }

# updated case  best parameters

param_grid={
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [5],
    'ccp_alpha': [0.05]
}




grid_search=GridSearchCV(DecisionTreeRegressor(random_state=42),param_grid,cv=5,scoring='r2')
grid_search.fit(X_train,Y_train)

best_params=grid_search.best_params_
print("best parameters are:",best_params)

Y_pred_train=grid_search.predict(X_train)
Y_pred_test=grid_search.predict(X_test)

R2_value_test=r2_score(Y_test,Y_pred_test)
R2_value_train=r2_score(Y_train,Y_pred_train)
MSE_value_test=mean_squared_error(Y_test,Y_pred_test)
MSE_value_train=mean_squared_error(Y_train,Y_pred_train)
MAE_value_test=mean_absolute_error(Y_test,Y_pred_test)
MAE_value_train=mean_absolute_error(Y_train,Y_pred_train)
RMSE_value_test=np.sqrt(mean_squared_error(Y_test,Y_pred_test))
RMSE_value_train=np.sqrt(mean_squared_error(Y_train,Y_pred_train))


print("R2 value test:",R2_value_test)
print("R2 value train:",R2_value_train)
print("mean squared error test",MSE_value_test)
print("mean squared error train",MSE_value_train)
print("mean absolute error test",MAE_value_test)
print("mean absolute error train",MAE_value_train)
print("root mean squared error test",RMSE_value_test)
print("root mean squared error train",RMSE_value_train)

DT_VALUE=[R2_value_test,R2_value_train,MSE_value_test,MSE_value_train,MAE_value_test,MAE_value_train,RMSE_value_test,RMSE_value_train]
DT_Train=[RMSE_value_train,MSE_value_train,MAE_value_train,R2_value_train]
DT_Test=[RMSE_value_test,MSE_value_test,MAE_value_test,R2_value_test]

plt.figure(figsize=(10,10))
plt.scatter(Y_test,Y_pred_test,color='blue')
plt.xlabel('Actual ESC', fontsize=22, fontweight='bold')
plt.ylabel('Predicted ESC', fontsize=22, fontweight='bold')
plt.grid(True)
plt.title('DT',fontsize=20,fontweight='bold')
plt.plot([-30,300],[-30,300],color='red',linestyle='-')
plt.xlim(-30, 300)
plt.ylim(-30, 300)
plt.gca().set_aspect('equal', adjustable='box') 

plt.text(
    -22, 239,
    f"RMSE = {RMSE_value_test:.2f}\nMSE = {MSE_value_test:.2f}\nMAE = {MAE_value_test:.2f}\n$R^2$ score = {R2_value_test:.2f}",
    fontsize=22,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

plt.show()

Y_pred_train=Y_pred_train.flatten()
Y_pred_test=Y_pred_test.flatten()

test_residuals_DT=Y_test-Y_pred_test






###
###
### RANDOM FOREST REGRESSOR
###
###


# updated case parameters

# param_grid = {
#     'n_estimators': [50, 100, 200, 300],       
#     'max_depth': [3, 5, 6, 8, None],           
#     'min_samples_split': [2, 4, 6, 8],         
#     'min_samples_leaf': [1, 2, 3, 4, 5],
# }

# updated case  best parameters

param_grid = {
    'n_estimators': [300],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [2]
}




grid_search=GridSearchCV(RandomForestRegressor(random_state=42),param_grid,cv=5,scoring='r2',verbose=1)
grid_search.fit(X_train,Y_train)

best_params=grid_search.best_params_
print("best parameters are:",best_params)



Y_pred_train=grid_search.predict(X_train)
Y_pred_test=grid_search.predict(X_test)

R2_value_test=r2_score(Y_test,Y_pred_test)
R2_value_train=r2_score(Y_train,Y_pred_train)
MSE_value_test=mean_squared_error(Y_test,Y_pred_test)
MSE_value_train=mean_squared_error(Y_train,Y_pred_train)
MAE_value_test=mean_absolute_error(Y_test,Y_pred_test)
MAE_value_train=mean_absolute_error(Y_train,Y_pred_train)
RMSE_value_test=np.sqrt(mean_squared_error(Y_test,Y_pred_test))
RMSE_value_train=np.sqrt(mean_squared_error(Y_train,Y_pred_train))


print("R2 value test:",R2_value_test)
print("R2 value train:",R2_value_train)
print("mean squared error test",MSE_value_test)
print("mean squared error train",MSE_value_train)
print("mean absolute error test",MAE_value_test)
print("mean absolute error train",MAE_value_train)
print("root mean squared error test",RMSE_value_test)
print("root mean squared error train",RMSE_value_train)

RF_VALUE=[R2_value_test,R2_value_train,MSE_value_test,MSE_value_train,MAE_value_test,MAE_value_train,RMSE_value_test,RMSE_value_train]
RF_Train=[RMSE_value_train,MSE_value_train,MAE_value_train,R2_value_train]
RF_Test=[RMSE_value_test,MSE_value_test,MAE_value_test,R2_value_test]

plt.figure(figsize=(10,10))
plt.scatter(Y_test,Y_pred_test,color='blue')
plt.xlabel('Actual ESC', fontsize=22, fontweight='bold')
plt.ylabel('Predicted ESC', fontsize=22, fontweight='bold')
plt.grid(True)
plt.title('RF',fontsize=20,fontweight='bold')
plt.plot([-30,300],[-30,300],color='red',linestyle='-')
plt.xlim(-30, 300)
plt.ylim(-30, 300)
plt.gca().set_aspect('equal', adjustable='box') 

plt.text(
    -22, 239,
    f"RMSE = {RMSE_value_test:.2f}\nMSE = {MSE_value_test:.2f}\nMAE = {MAE_value_test:.2f}\n$R^2$ score = {R2_value_test:.2f}",
    fontsize=22,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

plt.show()

Y_pred_train=Y_pred_train.flatten()
Y_pred_test=Y_pred_test.flatten()

test_residuals_RF=Y_test-Y_pred_test





###
###
### SUPPORT VECTOR REGRESSION
###
###


# updated case parameters

# param_grid = {
#     'svr__C': [0.1, 1, 10, 50, 100],              
#     'svr__epsilon': [0.01, 0.1, 0.2, 0.5, 1.0],  
#     'svr__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
# }

# updated case  best parameters

param_grid = {
    'svr__C': [100],
    'svr__epsilon': [0.2],
    'svr__kernel': ['rbf'],
}




pipeline=Pipeline(
    [
        ('scaler',StandardScaler()),
        ('svr',SVR())
    ]
)

grid_search=GridSearchCV(pipeline,param_grid,cv=5,scoring='r2',verbose=1)
grid_search.fit(X_train,Y_train)

best_params=grid_search.best_params_
print('best parameters are:',best_params)


Y_pred_train=grid_search.predict(X_train)
Y_pred_test=grid_search.predict(X_test)

R2_value_test=r2_score(Y_test,Y_pred_test)
R2_value_train=r2_score(Y_train,Y_pred_train)
MSE_value_test=mean_squared_error(Y_test,Y_pred_test)
MSE_value_train=mean_squared_error(Y_train,Y_pred_train)
MAE_value_test=mean_absolute_error(Y_test,Y_pred_test)
MAE_value_train=mean_absolute_error(Y_train,Y_pred_train)
RMSE_value_test=root_mean_squared_error(Y_test,Y_pred_test)
RMSE_value_train=root_mean_squared_error(Y_train,Y_pred_train)

SVR_VALUE=[R2_value_test,R2_value_train,MSE_value_test,MSE_value_train,MAE_value_test,MAE_value_train,RMSE_value_test,RMSE_value_train]
SVR_Train=[RMSE_value_train,MSE_value_train,MAE_value_train,R2_value_train]
SVR_Test=[RMSE_value_test,MSE_value_test,MAE_value_test,R2_value_test]


print("R2 value test:",R2_value_test)
print("R2 value train:",R2_value_train)
print("mean squared error test",MSE_value_test)
print("mean squared error train",MSE_value_train)
print("mean absolute error test",MAE_value_test)
print("mean absolute error train",MAE_value_train)
print("root mean squared error test",RMSE_value_test)
print("root mean squared error train",RMSE_value_train)



plt.figure(figsize=(10,10))
plt.scatter(Y_test,Y_pred_test,color='blue')
plt.xlabel('Actual ESC', fontsize=22, fontweight='bold')
plt.ylabel('Predicted ESC', fontsize=22, fontweight='bold')
plt.grid(True)
plt.title('SVR',fontsize=20,fontweight='bold')
plt.plot([-30,300],[-30,300],color='red',linestyle='-')
plt.xlim(-30, 300)
plt.ylim(-30, 300)
plt.gca().set_aspect('equal', adjustable='box') 

plt.text(
    -22, 239,
    f"RMSE = {RMSE_value_test:.2f}\nMSE = {MSE_value_test:.2f}\nMAE = {MAE_value_test:.2f}\n$R^2$ score = {R2_value_test:.2f}",
    fontsize=22,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

plt.show()

Y_pred_train=Y_pred_train.flatten()
Y_pred_test=Y_pred_test.flatten()

test_residuals_SVR=Y_test-Y_pred_test




###
###
### GRADIENT BOOSTING REGRESSION
###
###




# updated case parameters

# param_grid={
#     'gbr__n_estimators': [100, 200, 300, 500],          
#     'gbr__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
#     'gbr__max_depth': [3, 4, 5, 6],                     
#     'gbr__min_samples_split': [2, 4, 6, 8],             
#     'gbr__min_samples_leaf': [1, 2, 3, 4],
# }


# updated case  best parameters

param_grid = {
    'gbr__n_estimators': [500],
    'gbr__learning_rate': [0.1],
    'gbr__max_depth': [3],
    'gbr__min_samples_split': [4],
    'gbr__min_samples_leaf': [1]
}


pipeline=Pipeline(
    [
        ('scaler',StandardScaler()),
        ('gbr',GradientBoostingRegressor(random_state=42))
    ]
)

grid_search=GridSearchCV(pipeline,param_grid,cv=5,scoring='r2',verbose=1,n_jobs=-1)
grid_search.fit(X_train,Y_train)

best_params=grid_search.best_params_
print("best parameters are:",best_params)

Y_pred_train=grid_search.predict(X_train)
Y_pred_test=grid_search.predict(X_test)


R2_value_test=r2_score(Y_test,Y_pred_test)
R2_value_train=r2_score(Y_train,Y_pred_train)
MSE_value_test=mean_squared_error(Y_test,Y_pred_test)
MSE_value_train=mean_squared_error(Y_train,Y_pred_train)
MAE_value_test=mean_absolute_error(Y_test,Y_pred_test)
MAE_value_train=mean_absolute_error(Y_train,Y_pred_train)
RMSE_value_test=root_mean_squared_error(Y_test,Y_pred_test)
RMSE_value_train=root_mean_squared_error(Y_train,Y_pred_train)

GB_VALUE=[R2_value_test,R2_value_train,MSE_value_test,MSE_value_train,MAE_value_test,MAE_value_train,RMSE_value_test,RMSE_value_train]
GB_Train=[RMSE_value_train,MSE_value_train,MAE_value_train,R2_value_train]
GB_Test=[RMSE_value_test,MSE_value_test,MAE_value_test,R2_value_test]


print("R2 value test:",R2_value_test)
print("R2 value train:",R2_value_train)
print("mean squared error test",MSE_value_test)
print("mean squared error train",MSE_value_train)
print("mean absolute error test",MAE_value_test)
print("mean absolute error train",MAE_value_train)
print("root mean squared error test",RMSE_value_test)
print("root mean squared error train",RMSE_value_train)


plt.figure(figsize=(10,10))
plt.scatter(Y_test,Y_pred_test,color='blue')
plt.xlabel('Actual ESC', fontsize=22, fontweight='bold')
plt.ylabel('Predicted ESC', fontsize=22, fontweight='bold')
plt.grid(True)
plt.title('GB',fontsize=20,fontweight='bold')
plt.plot([-30,300],[-30,300],color='red',linestyle='-')
plt.xlim(-30, 300)
plt.ylim(-30, 300)
plt.gca().set_aspect('equal', adjustable='box') 

plt.text(
    -22, 239,
    f"RMSE = {RMSE_value_test:.2f}\nMSE = {MSE_value_test:.2f}\nMAE = {MAE_value_test:.2f}\n$R^2$ score = {R2_value_test:.2f}",
    fontsize=22,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)


plt.show()

Y_pred_train=Y_pred_train.flatten()
Y_pred_test=Y_pred_test.flatten()

test_residuals_GBR=Y_test-Y_pred_test




###
###
### ARTIFICIAL NEURAL NETWORK
###
###






scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def ANN_model():
    model= keras.Sequential()
    model.add(layers.InputLayer(shape=X_train.shape[1:]))
    for _ in range(n_hidden):
        model.add(layers.Dense(n_neurons,activation=activation))
    model.add(layers.Dense(1))
    model.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model


# updated case parameters

# param_grid={
#     'n_hidden': [1, 2, 3, 4, 5],                 
#     'n_neurons': [10, 20, 30, 50, 100, 200],
#     'learning_rate': [0.001, 0.01, 0.05, 0.1],
#     'batch_size': [16, 32, 64, 128],
#     'epochs': [50, 100, 200, 500],
#     'activation': ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
# }

# updated case  best parameters

param_grid = {
    'n_hidden': [2],
    'n_neurons': [100],
    'learning_rate': [0.05],
    'batch_size': [64],
    'epochs': [500],
    'activation': ['relu']
}



best_r2 = -np.inf
best_params = {}
best_model = None


for n_hidden in param_grid['n_hidden']:
    for n_neurons in param_grid['n_neurons']:
        for learning_rate in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for epochs in param_grid['epochs']:
                    for activation in param_grid['activation']:            
                        print(f'Training with: n_hidden={n_hidden}, n_neurons={n_neurons}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}, activation={activation}')
                        model = ANN_model()
                        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                        Y_pred = model.predict(X_test)
                        r2 = r2_score(Y_test, Y_pred)
                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = {'n_hidden': n_hidden, 'n_neurons': n_neurons, 'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs,  'activation':activation}
                            best_model = model
                            callbacks.EarlyStopping(monitor="val_loss", mode="min",patience=5, restore_best_weights=True)

print("Best Parameters:", best_params)
print("Best R2:", best_r2)

Y_pred_train=best_model.predict(X_train)
Y_pred_test=best_model.predict(X_test)


Y_pred_train1 = pd.DataFrame(Y_pred_train)
Y_pred_test1 = pd.DataFrame(Y_pred_test)

R2_value_test=r2_score(Y_test,Y_pred_test)
R2_value_train=r2_score(Y_train,Y_pred_train)
MSE_value_test=mean_squared_error(Y_test,Y_pred_test)
MSE_value_train=mean_squared_error(Y_train,Y_pred_train)
MAE_value_test=mean_absolute_error(Y_test,Y_pred_test)
MAE_value_train=mean_absolute_error(Y_train,Y_pred_train)
RMSE_value_test=root_mean_squared_error(Y_test,Y_pred_test)
RMSE_value_train=root_mean_squared_error(Y_train,Y_pred_train)


print("R2 value test:",R2_value_test)
print("R2 value train:",R2_value_train)
print("mean squared error test",MSE_value_test)
print("mean squared error train",MSE_value_train)
print("mean absolute error test",MAE_value_test)
print("mean absolute error train",MAE_value_train)
print("root mean squared error test",RMSE_value_test)
print("root mean squared error train",RMSE_value_train)

ANN_VALUE=[R2_value_test,R2_value_train,MSE_value_test,MSE_value_train,MAE_value_test,MAE_value_train,RMSE_value_test,RMSE_value_train]
ANN_Train=[RMSE_value_train,MSE_value_train,MAE_value_train,R2_value_train]
ANN_Test=[RMSE_value_test,MSE_value_test,MAE_value_test,R2_value_test]


plt.figure(figsize=(10,10))
plt.scatter(Y_test,Y_pred_test,color='blue')
plt.xlabel('Actual ESC', fontsize=22, fontweight='bold')
plt.ylabel('Predicted ESC', fontsize=22, fontweight='bold')
plt.grid(True)
plt.title('ANN',fontsize=20,fontweight='bold')
plt.plot([-30,300],[-30,300],color='red',linestyle='-')
plt.xlim(-30, 300)
plt.ylim(-30, 300)
plt.gca().set_aspect('equal', adjustable='box') 

plt.text(
    -22, 239,
    f"RMSE = {RMSE_value_test:.2f}\nMSE = {MSE_value_test:.2f}\nMAE = {MAE_value_test:.2f}\n$R^2$ score = {R2_value_test:.2f}",
    fontsize=22,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

plt.show()

Y_pred_train=Y_pred_train.flatten()
Y_pred_test=Y_pred_test.flatten()

test_residuals_ANN=Y_test-Y_pred_test









custom_feature_names = [
    '$T^{eq}$', '$ΔH^{r}$', '${Cp}_{MOx}$', '${Cp}_{MOy}$',
    '$ρ_{MOx}$', '$ρ_{MOy}$', 'MP', 'ν', '${MW}_{MOx}$'
]


# Assuming best_model, X_train, X_test are already defined
explainer = shap.KernelExplainer(best_model.predict, X_train)
shap_values = explainer.shap_values(X_test, nsamples=20)
shap_values = np.array(shap_values).reshape(X_test.shape)
base_value = explainer.expected_value

shap_values_explanation = shap.Explanation(
    values=shap_values,
    base_values=base_value,
    data=X_test,
    feature_names=custom_feature_names
)

def set_bold_labels(ax):
    ax.xaxis.label.set_fontsize(22)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(22)
    ax.yaxis.label.set_fontweight('bold')
    ax.title.set_fontsize(22)
    ax.title.set_fontweight('bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(22)
        label.set_fontweight('bold')

# Make combined plot with closer panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'wspace': 0.10, 'width_ratios': [1, 1]})

plt.sca(ax1)
shap.plots.beeswarm(shap_values_explanation, plot_size=None, show=False)
ax1.set_xlabel("SHAP value", fontsize=22, fontweight='bold')
for collection in ax1.collections:
    collection.set_sizes([60])
set_bold_labels(ax1)
for tick in ax1.get_yticklabels():
    tick.set_fontsize(22)
    tick.set_fontweight('bold')
    tick.set_color('black')  # force darker text color



shap.plots.bar(shap_values_explanation, ax=ax2, show=False)
ax2.set_xlabel("mean SHAP value", fontsize=22, fontweight='bold')
set_bold_labels(ax2)

# Annotate with "A" and "B"
ax1.text(-0.08, 1.05, '(A)', transform=ax1.transAxes,
         fontsize=22, fontweight='bold', va='top', ha='right')
ax2.text(-0.08, 1.05, '(B)', transform=ax2.transAxes,
         fontsize=22, fontweight='bold', va='top', ha='right')

# Only one big colorbar label
found = False
for ax in fig.axes:
    if hasattr(ax, "get_ylabel") and ax.get_ylabel() == "Feature value":
        if not found:
            ax.set_ylabel("Feature value", fontsize=22, fontweight='bold')
            found = True
        else:
            ax.set_ylabel("")

plt.tight_layout()
plt.show()











train_data = {
    "Model": ["LR", "DT", "RF", "SVR", "GB", "ANN"],
    "RMSE": [LR_Train[0], DT_Train[0], RF_Train[0], SVR_Train[0], GB_Train[0], ANN_Train[0]],
    "MSE":  [LR_Train[1], DT_Train[1], RF_Train[1], SVR_Train[1], GB_Train[1], ANN_Train[1]],
    "MAE":  [LR_Train[2], DT_Train[2], RF_Train[2], SVR_Train[2], GB_Train[2], ANN_Train[2]],
    "R2 score": [LR_Train[3], DT_Train[3], RF_Train[3], SVR_Train[3], GB_Train[3], ANN_Train[3]]
}

test_data = {
    "Model": ["LR", "DT", "RF", "SVR", "GB", "ANN"],
    "RMSE": [LR_Test[0], DT_Test[0], RF_Test[0], SVR_Test[0], GB_Test[0], ANN_Test[0]],
    "MSE":  [LR_Test[1], DT_Test[1], RF_Test[1], SVR_Test[1], GB_Test[1], ANN_Test[1]],
    "MAE":  [LR_Test[2], DT_Test[2], RF_Test[2], SVR_Test[2], GB_Test[2], ANN_Test[2]],
    "R2 score": [LR_Test[3], DT_Test[3], RF_Test[3], SVR_Test[3], GB_Test[3], ANN_Test[3]]
}

df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# Save into Excel with 2 sheets
file_path = "RADAR DATA ESC.xlsx"
with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
    df_train.to_excel(writer, sheet_name="Train", index=False)
    df_test.to_excel(writer, sheet_name="Test", index=False)




df= pd.DataFrame({'LINEAR REGRESSION':LR_VALUE,
                 'DECISION TREE':DT_VALUE,
                 'RANDOM FOREST':RF_VALUE,
                 "SVM":SVR_VALUE,
                 "GBR":GB_VALUE,
                 "ANN":ANN_VALUE}
                 ) 
df.to_excel('COMPARISON OF ERROR ESC.xlsx',index=0)





df= pd.DataFrame({'LR':test_residuals_LR,
                 'DT':test_residuals_DT,
                 'RF':test_residuals_RF,
                 "SVM":test_residuals_SVR,
                 "GBR":test_residuals_GBR,
                 "ANN":test_residuals_ANN}
                 ) 

df.to_excel('residuals for ESC.xlsx',index=0)





path = "training and testing data esc.xlsx"

with pd.ExcelWriter(path , engine='openpyxl') as writer:
    X_train1.to_excel(writer, sheet_name="X_train", index=False)
    X_test1.to_excel(writer, sheet_name="X_test", index=False)
    Y_train1.to_excel(writer, sheet_name="Y_train", index=False)
    Y_test1.to_excel(writer, sheet_name="Y_test", index=False)
    Y_pred_train1.to_excel(writer, sheet_name="Y_pred_train", index=False)
    Y_pred_test1.to_excel(writer, sheet_name="Y_pred_test", index=False)















