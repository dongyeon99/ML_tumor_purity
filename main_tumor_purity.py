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

from utils_tumor_purity import *
from models_tumor_purity import *


#### Data prepare ###
# path
data_folder = "/mnt/c/Users/laboratory/Documents/"

# miRNA & tumor purity merge (TCGA) [except STAD]
tumor_type = ['BRCA','KIRC','UCEC','THCA','HNSC','LUAD','PRAD','LGG',
             'LUSC','OV','COAD','SKCM','BLCA','LIHC','KIRP', 'CESC',
             'SARC','ESCA','LAML','PCPG','PAAD','READ','TGCT',
             'THYM','KICH','MESO','ACC','UVM','UCS','DLBC','CHOL','GBM']

# Each tumor type data (TCGA)
for i in tumor_type:   
    globals()['{}_cpe'.format(i)] = miRNA_purity(data_folder, i)

# model dataset (TCGA) [except STAD]
model_tumor_type = ['BRCA','KIRC','UCEC','THCA','HNSC','LUAD','PRAD','LGG',
                    'LUSC','OV','COAD','SKCM','BLCA','LIHC','KIRP','CESC']

All = model_data_merge(model_tumor_type)


#### All miRNAs RFR modeling ####
print("#### All miRNAs RFR modeling ####")
print(" ")

# Data split
All_tr_te = Data_tr_te(All)
x_train, x_test, y_train, y_test = All_tr_te[0], All_tr_te[1], All_tr_te[2], All_tr_te[3]

# RFR model 
print("All miRNAs RFR model's MSE")
All_RFR = RFR(x_train, x_test, y_train, y_test)
All_RFR_best_model, All_RFR_pred, All_RFR_MSE_test = All_RFR[0], All_RFR[1], All_RFR[2]
print(" ")

# RFR model scatterplot 
print("All miRNAs RFR model Predict value & Real value correlation")
plot_name1 = "All_RFR"
All_RFR_scatter_plot = scatter_plot(data_folder, plot_name1, y_test, All_RFR_pred)
print(" ")


#### All miRNAs RFR modeling (Each Tumor type) ####
# Data split
for i in model_tumor_type:
    data = globals()['{}_cpe'.format(i)]
    tr_te = Data_tr_te(data)
    globals()['{}_x_train'.format(i)], globals()['{}_x_test'.format(i)], globals()['{}_y_train'.format(i)], globals()['{}_y_test'.format(i)] = tr_te[0], tr_te[1], tr_te[2], tr_te[3]

# RFR model
for i in model_tumor_type:
    print("TCGA tumor type:", i)
    RFR_each = RFR(globals()['{}_x_train'.format(i)], globals()['{}_x_test'.format(i)], globals()['{}_y_train'.format(i)], globals()['{}_y_test'.format(i)])
    globals()['{}_RFR_model'.format(i)], globals()['{}_RFR_pred'.format(i)], globals()['{}_RFR_MSE_test'.format(i)] = RFR_each[0], RFR_each[1], RFR_each[2]


#### Top 10 miRNAs & Tumor purity Correlation ####
# miRNA feature importance
All_feature_importance = feature_importance(All_RFR_best_model, x_train)

# Top 10 miRNA & tumor purity correlation
print(" ")
top10 = list(All_feature_importance.index[0:10])
print("Top 10 miRNAs feature importance score")
print(All_feature_importance.head(10))
print(" ")

PCC = miRNA_CPE_pcc(All[top10], All[['CPE']])
print("Top 10 miRNA & tumor purity correlation")
print(PCC)
print(" ")


#### Using Top 1 ~ 10 miRNAs 8 ML model's MSE ####
print("#### Top miRNAs RFR modeling ####")
print(" ")

# Top miRNA data processing
TOP = top_miRNA_Data_tr_ts(All, All_feature_importance)

for i in range(4):
    if i == 0:
        for j in range(10):
            globals()['top{}_x_train'.format(j+1)] = TOP[i][j]
    if i == 1:
        for j in range(10):
            globals()['top{}_x_test'.format(j+1)] = TOP[i][j]  
    if i == 2:
        for j in range(10):
            globals()['top{}_y_train'.format(j+1)] = TOP[i][j]
    if i == 3:
        for j in range(10):
            globals()['top{}_y_test'.format(j+1)] = TOP[i][j]   

# 8 ML model's MSE
for i in range(1,11):
    print("Top {} miRNAs".format(i))
    ML_8(globals()['top{}_x_train'.format(i)], globals()['top{}_x_test'.format(i)], globals()['top{}_y_train'.format(i)], globals()['top{}_y_test'.format(i)])  
    print(" ")


#### Using Top 10 miRNAs for RFR model [Each tumor type] ####
# Data processing Using Top 10 miRNA for Each tumor type
for i in model_tumor_type:
    top10 = list(All_feature_importance.index[0:10])
    tr_te = Data_tr_te(globals()['{}_cpe'.format(i)])
    x_train = tr_te[0]
    x_test = tr_te[1]
    x_train = x_train[top10]
    x_test = x_test[top10]
    globals()['{}_top10_x_train'.format(i)], globals()['{}_top10_x_test'.format(i)], globals()['{}_top10_y_train'.format(i)], globals()['{}_top10_y_test'.format(i)] = x_train, x_test, tr_te[2], tr_te[3]

# RFR model
for i in model_tumor_type:
    print("(Top 10) TCGA tumor type:", i)
    RFR_each = RFR(globals()['{}_top10_x_train'.format(i)], globals()['{}_top10_x_test'.format(i)], globals()['{}_top10_y_train'.format(i)], globals()['{}_top10_y_test'.format(i)])
    globals()['{}_top10_RFR_model'.format(i)], globals()['{}_top10_RFR_pred'.format(i)], globals()['{}_top10_RFR_MSE_test'.format(i)] = RFR_each[0], RFR_each[1], RFR_each[2]


#### Validation Test ####
print("#### Validation Test ####")
print(" ")

#### TCGA validation dataset, Top 10 miRNA for pretrained RFR model ####
# Validation datset (TCGA)
validation_tumor_type = ['SARC','ESCA','LAML','PCPG','PAAD','READ','TGCT','THYM',
                         'KICH','MESO','ACC','UVM','UCS','DLBC','CHOL','GBM']

validation = model_data_merge(validation_tumor_type)

validation_top10_x = validation[list(All_feature_importance.index[0:10])]

validation_y = validation[['CPE']]

# pretrained RFR model
print("Top 10 RFR MSE test")
Top10_RFR = RFR(top10_x_train, top10_x_test, top10_y_train, top10_y_test)
Top10_RFR_best_model= Top10_RFR[0]
print(" ")

# Predict
Top10_validation_pred = Top10_RFR_best_model.predict(validation_top10_x)

# score
Top10_RFR_MSE_validation = mean_squared_error(validation_y, Top10_validation_pred)
print("Top 10 miRNA for pretrained RFR model MSE validation: ", Top10_RFR_MSE_validation)

# Top 10 miRNA RFR model scatterplot 
print("Validation dataset, Top10 miRNAs RFR model Predict value & Real value correlation")
plot_name2 = "Top10_RFR_validation"
Top10_RFR_validation_scatter_plot=scatter_plot(data_folder, plot_name2, validation_y, Top10_validation_pred)
print("")


#### Validation PCAWG dataset, Top 10 miRNA for RFR model ####
# PCAWG dataset processing [validation]

PCAWG = PCAWG_purity(data_folder)

# Train Test data split
PCAWG_x = PCAWG.iloc[:, 1:len(PCAWG.columns)-1]
PCAWG_y = PCAWG[['purity']]

# split data
PCAWG_x_train, PCAWG_x_test, PCAWG_y_train, PCAWG_y_test = train_test_split(PCAWG_x,PCAWG_y, test_size=0.30, random_state=42)

# RFR model
# All miRNAs Train RFR model 
print("PCAWG dataset miRNA & tumor purity All miRNAs RFR model")
PCAWG_RFR = RFR(PCAWG_x_train, PCAWG_x_test, PCAWG_y_train, PCAWG_y_test)
PCAWG_RFR_best_model, PCAWG_RFR_pred, PCAWG_RFR_MSE_test = PCAWG_RFR[0], PCAWG_RFR[1], PCAWG_RFR[2]

# Top 10 miRNAs RFR model 
# feature importance list
PCAWG_feature_importance = feature_importance(PCAWG_RFR_best_model, PCAWG_x_train)

PCAWG_top10_x_train = PCAWG_x_train[list(PCAWG_feature_importance.index[0:10])]
PCAWG_top10_x_test = PCAWG_x_test[list(PCAWG_feature_importance.index[0:10])]

# Train RFR model
print("PCAWG dataset miRNA & tumor purity Top 10 miRNAs RFR model")
PCAWG_top10_RFR = RFR(PCAWG_top10_x_train, PCAWG_top10_x_test, PCAWG_y_train, PCAWG_y_test)
PCAWG_top10_RFR_best_model, PCAWG_top10_RFR_pred, PCAWG_top10_RFR_MSE_test = PCAWG_top10_RFR[0], PCAWG_top10_RFR[1], PCAWG_top10_RFR[2]

# PCAWG dataset, Top 10 miRNAs RFR model scatterplot 
print("PCAWG dataset, Top10 miRNAs RFR model Predict value & Real value correlation")
plot_name3 = "Top10_RFR_PCAWG"
Top10_RFR_PCAWG_scatter_plot=scatter_plot(data_folder, plot_name3, PCAWG_y_test, PCAWG_top10_RFR_pred)
print( "")


