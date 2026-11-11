#### Package load ####
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils_tumor_purity import *
from models_tumor_purity import *


#### Data prepare ###
# path
data_folder = "data/"

# miRNA & tumor purity merge (TCGA) [except STAD]
tumor_type = ['BRCA','KIRC','UCEC','THCA','HNSC','LUAD','PRAD','LGG',
             'LUSC','OV','COAD','SKCM','BLCA','LIHC','KIRP', 'CESC',
             'SARC','ESCA','LAML','PCPG','PAAD','READ','TGCT',
             'THYM','KICH','MESO','ACC','UVM','UCS','DLBC','CHOL','GBM']

# Each tumor type data (TCGA)
CPE = load_cpe(data_folder)
cpe = {i: miRNA_purity(data_folder, i, CPE) for i in tumor_type}

# model dataset (TCGA) [except STAD]
model_tumor_type = ['BRCA','KIRC','UCEC','THCA','HNSC','LUAD','PRAD','LGG',
                    'LUSC','OV','COAD','SKCM','BLCA','LIHC','KIRP','CESC']

All = model_data_merge(cpe, model_tumor_type)

# Test MSE of every RFR model, saved to CSV at the end
mse_results = []


#### All miRNAs RFR modeling ####
print("#### All miRNAs RFR modeling ####")
print(" ")

# Data split
x_train, x_test, y_train, y_test = Data_tr_te(All)

# RFR model
print("All miRNAs RFR model's MSE")
All_RFR_best_model, All_RFR_pred, All_RFR_MSE_test = RFR(x_train, x_test, y_train, y_test)
mse_results.append(("All miRNAs", "All", All_RFR_MSE_test))
print(" ")

# RFR model scatterplot
print("All miRNAs RFR model Predict value & Real value correlation")
scatter_plot(data_folder, "All_RFR", y_test, All_RFR_pred)
print(" ")


#### All miRNAs RFR modeling (Each Tumor type) ####
for i in model_tumor_type:
    print("TCGA tumor type:", i)
    MSE_test = RFR(*Data_tr_te(cpe[i]))[2]
    mse_results.append(("All miRNAs", i, MSE_test))


#### Top 10 miRNAs & Tumor purity Correlation ####
# miRNA feature importance
All_feature_importance = feature_importance(All_RFR_best_model, x_train)
All_feature_importance.to_csv(os.path.join(data_folder, "feature_importance.csv"), header=["importance"])

# Top 10 miRNA & tumor purity correlation
print(" ")
top10 = list(All_feature_importance.index[0:10])
print("Top 10 miRNAs feature importance score")
print(All_feature_importance.head(10))
print(" ")

PCC = miRNA_CPE_pcc(All[top10], All[['CPE']])
PCC.to_csv(os.path.join(data_folder, "top10_miRNA_CPE_pcc.csv"))
print("Top 10 miRNA & tumor purity correlation")
print(PCC)
print(" ")


#### Using Top 1 ~ 10 miRNAs 8 ML model's MSE ####
print("#### Top miRNAs RFR modeling ####")
print(" ")

# Top miRNA data processing
top_x_train, top_x_test, top_y_train, top_y_test = top_miRNA_Data_tr_ts(All, All_feature_importance)

# 8 ML model's MSE
ML_8_results = {}
for i in range(10):
    print("Top {} miRNAs".format(i+1))
    ML_8_results[i+1] = ML_8(top_x_train[i], top_x_test[i], top_y_train[i], top_y_test[i])
    print(" ")

pd.DataFrame(ML_8_results).T.rename_axis("top_n_miRNAs").to_csv(os.path.join(data_folder, "ML_8_MSE.csv"))


#### Using Top 10 miRNAs for RFR model [Each tumor type] ####
for i in model_tumor_type:
    print("(Top 10) TCGA tumor type:", i)
    x_tr, x_te, y_tr, y_te = Data_tr_te(cpe[i])
    MSE_test = RFR(x_tr[top10], x_te[top10], y_tr, y_te)[2]
    mse_results.append(("Top 10 miRNAs", i, MSE_test))


#### Validation Test ####
print("#### Validation Test ####")
print(" ")

#### TCGA validation dataset, Top 10 miRNA for pretrained RFR model ####
# Validation datset (TCGA)
validation_tumor_type = ['SARC','ESCA','LAML','PCPG','PAAD','READ','TGCT','THYM',
                         'KICH','MESO','ACC','UVM','UCS','DLBC','CHOL','GBM']

validation = model_data_merge(cpe, validation_tumor_type)

validation_top10_x = validation[top10]

validation_y = validation[['CPE']]

# pretrained RFR model
print("Top 10 RFR MSE test")
Top10_RFR_best_model, _, Top10_RFR_MSE_test = RFR(top_x_train[9], top_x_test[9], top_y_train[9], top_y_test[9])
mse_results.append(("Top 10 miRNAs", "All", Top10_RFR_MSE_test))
print(" ")

# Predict
Top10_validation_pred = Top10_RFR_best_model.predict(validation_top10_x)

# score
Top10_RFR_MSE_validation = mean_squared_error(validation_y, Top10_validation_pred)
mse_results.append(("Top 10 miRNAs", "TCGA validation", Top10_RFR_MSE_validation))
print("Top 10 miRNA for pretrained RFR model MSE validation: ", Top10_RFR_MSE_validation)

# Top 10 miRNA RFR model scatterplot
print("Validation dataset, Top10 miRNAs RFR model Predict value & Real value correlation")
scatter_plot(data_folder, "Top10_RFR_validation", validation_y, Top10_validation_pred)
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
PCAWG_RFR_best_model, _, PCAWG_RFR_MSE_test = RFR(PCAWG_x_train, PCAWG_x_test, PCAWG_y_train, PCAWG_y_test)
mse_results.append(("All miRNAs", "PCAWG", PCAWG_RFR_MSE_test))

# Top 10 miRNAs RFR model
# feature importance list
PCAWG_top10 = list(feature_importance(PCAWG_RFR_best_model, PCAWG_x_train).index[0:10])

# Train RFR model
print("PCAWG dataset miRNA & tumor purity Top 10 miRNAs RFR model")
_, PCAWG_top10_RFR_pred, PCAWG_top10_RFR_MSE_test = RFR(PCAWG_x_train[PCAWG_top10], PCAWG_x_test[PCAWG_top10], PCAWG_y_train, PCAWG_y_test)
mse_results.append(("Top 10 miRNAs", "PCAWG", PCAWG_top10_RFR_MSE_test))

# PCAWG dataset, Top 10 miRNAs RFR model scatterplot
print("PCAWG dataset, Top10 miRNAs RFR model Predict value & Real value correlation")
scatter_plot(data_folder, "Top10_RFR_PCAWG", PCAWG_y_test, PCAWG_top10_RFR_pred)
print( "")

# Save all RFR test MSEs
pd.DataFrame(mse_results, columns=["features", "dataset", "MSE"]).to_csv(os.path.join(data_folder, "RFR_MSE.csv"), index=False)
