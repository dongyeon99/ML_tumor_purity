#### Package load ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


#### Random Forest Regression model ####
def RFR(x_train, x_test, y_train, y_test):
    
    # best parameter model
    best_model = RandomForestRegressor(max_depth=20, n_estimators=500, n_jobs=-1)
    
    # Train model
    best_model.fit(x_train, y_train.values.ravel())
    
    # Prediction
    best_model_pred = best_model.predict(x_test)
        
    # Score
    MSE_test = mean_squared_error(y_test, best_model_pred)
    print("MSE_test:", MSE_test)  
    
    return best_model, best_model_pred, MSE_test


#### Using Top 1 ~ 10 miRNA for 8 Machine learning tools ####
def ML_8(top_x_train, top_x_test, y_train, y_test):
    
    #### 1. Random Forest Regression model ####
    
    RFR = RandomForestRegressor(max_depth = 20, n_estimators = 500, n_jobs = -1)
    RFR.fit(top_x_train, y_train.values.ravel())

    # Predict
    RFR_pred = RFR.predict(top_x_test)
    RFR_pred

    # score
    mse_RFR_test = mean_squared_error(y_test, RFR_pred)
    print("RFR_MSE_test: ", mse_RFR_test)
    
    #### 2. Lienar Regression model ####
    
    Linear = linear_model.LinearRegression(n_jobs= -1)
    Linear.fit(top_x_train, y_train.values.ravel())

    # Predict
    Linear_pred = Linear.predict(top_x_test)

    # Score
    mse_Linear_test = mean_squared_error(y_test, Linear_pred)
    print("Linear_MSE_test: ", mse_Linear_test)
    
    #### 3. Lasso Regression model ####
    
    Lasso = linear_model.Lasso(alpha = 0.000774263682681127, max_iter = 1000)
    Lasso.fit(top_x_train, y_train.values.ravel())

    # Predict
    Lasso_pred = Lasso.predict(top_x_test)

    # Score
    mse_Lasso_test = mean_squared_error(y_test, Lasso_pred)
    print("Lasso_MSE_test: ", mse_Lasso_test)
    
    #### 4. Ridge model ####
    
    Ridge = linear_model.Ridge(alpha = 1.0, max_iter = 1000)
    Ridge.fit(top_x_train, y_train.values.ravel())

    # Predict
    Ridge_pred = Ridge.predict(top_x_test)

    # Score
    mse_Ridge_test = mean_squared_error(y_test, Ridge_pred)
    print("Ridge_MSE_test: ", mse_Ridge_test)
    
    #### 5. ElasticNet model ####
    
    ElasticNet = linear_model.ElasticNet(alpha = 0.002154434690031882, l1_ratio = 0.55)
    ElasticNet.fit(top_x_train, y_train.values.ravel())

    # Predict
    ElasticNet_pred = ElasticNet.predict(top_x_test)

    # Score
    mse_ElasticNet_test = mean_squared_error(y_test, ElasticNet_pred)
    print("ElasticNet_MSE_test: ", mse_ElasticNet_test)
    
    #### 6. SVR model ####
    
    svr = make_pipeline(StandardScaler(), SVR(C = 0.5, gamma = 'auto', kernel = 'rbf'))
    svr.fit(top_x_train, y_train.values.ravel())

    # Predict
    svr_pred = svr.predict(top_x_test)

    # Score
    mse_svr_test = mean_squared_error(y_test, svr_pred)
    print("SVR_MSE_test: ", mse_svr_test)
    
    #### 7. KNR model ####
    
    kneighbor = KNeighborsRegressor(algorithm = 'ball_tree', leaf_size = 25,
                                   n_neighbors = 9, weights = 'distance', n_jobs = -1)
    kneighbor.fit(top_x_train, y_train.values.ravel())

    # Predict
    kneighbor_pred = kneighbor.predict(top_x_test)

    # Score
    mse_kneighbor_test = mean_squared_error(y_test, kneighbor_pred)
    print("KNR_MSE_test: ", mse_kneighbor_test)
    
    #### 8. MLP model ####
    # Train MLP model
    MLP = MLPRegressor(activation = 'logistic', solver = 'lbfgs',
                      learning_rate = 'adaptive', alpha = 1)
    MLP.fit(top_x_train, y_train.values.ravel())

    # Predict
    MLP_pred = MLP.predict(top_x_test)

    # Score
    mse_MLP_test = mean_squared_error(y_test, MLP_pred)
    print("MLP_MSE_test:",mse_MLP_test)


