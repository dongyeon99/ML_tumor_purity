#### Package load ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.model_selection import train_test_split


#### Data combine [miRNA expression & Tumor purity] ####
def miRNA_purity(data_folder, tumor_type):
    """
    Prepare miRNA expression and tumor purity data for a specific TCGA tumor type.
    Args:
        data_folder (str): Path to the folder containing the data files.
        tumor_type (str): The type of tumor (e.g., 'BRCA', 'LUAD').
    Returns:
        pd.DataFrame: Merged DataFrame containing miRNA expression and tumor purity data.
    """
    warnings.filterwarnings("ignore")
    # Prepare tumor purity data #
    purity = pd.read_excel(os.path.join(data_folder,"ncomms9971-s2.xlsx"), 
                      header = None)

    purity = purity.iloc[3:, :]
    purity_header = purity.iloc[0]
    purity = purity.iloc[1:]
    purity.columns = purity_header
    CPE = purity[['Sample ID', 'CPE']]

    # Remove Nan value
    CPE = CPE.dropna(axis = 0)
    CPE.reset_index(drop=True, inplace=True)

    # Prepare miRNA expression data #
    miRNA = pd.read_csv(os.path.join(data_folder,"TCGA-{}.mirna.tsv".format(tumor_type)), sep ='\t', header = None)
    
    miRNA = miRNA.transpose()
    miRNA_header = miRNA.iloc[0]
    miRNA = miRNA.iloc[1:]
    miRNA.columns = miRNA_header
    miRNA.rename(columns={'miRNA_ID':'Sample ID'}, inplace=True)
    
    # Merge miRNA & tumor purity data
    globals()['{}_cpe'.format(tumor_type)] = pd.merge(miRNA, CPE, how = 'inner',on = 'Sample ID')
    
    return globals()['{}_cpe'.format(tumor_type)]


#### Each TCGA tumor type merge ####
def model_data_merge(tumor_type):
    """
    Merge miRNA expression and tumor purity data for multiple TCGA tumor types.
    Args:
        tumor_type (list): List of tumor types (e.g., ['BRCA', 'LUAD']).
    Returns:
        pd.DataFrame: Concatenated DataFrame containing miRNA expression and tumor purity data for
        all specified tumor types.
    """
    list_tupu = []
    
    for i in tumor_type:
        list_tupu.append(globals()['{}_cpe'.format(i)])
        
    dataset = pd.concat(list_tupu)
    
    return dataset


#### Train Test Data Split ####
def Data_tr_te(data):
    """
    Split the dataset into training and testing sets.
    Args:
        data (pd.DataFrame): DataFrame containing miRNA expression and tumor purity data.
    Returns:
        tuple: Four DataFrames representing the training and testing sets for features and target values.
    """
    # miRNA
    x = data.iloc[:, 1:len(data.columns)-1]
    # target values
    y = data[['CPE']]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    
    return x_train, x_test, y_train, y_test


#### Feature importance ####
def feature_importance(best_model, x_train):
    """
    Calculate feature importance from a trained Random Forest Regressor model.
    Args:
        best_model (RandomForestRegressor): Trained Random Forest Regressor model.
        x_train (pd.DataFrame): Training features DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing feature importance scores sorted in descending order.
    """

    # get feature names
    X = x_train
    feature_names = X.columns.tolist()
    
    # Feature importance from RFR
    feature_importance = pd.DataFrame(best_model.feature_importances_, index = feature_names)
    feature_importance_sort = feature_importance.sort_values(by=0, 
                                                             axis=0, 
                                                             ascending=False)
    
    return feature_importance_sort


#### Top miRNA data processing ####
def top_miRNA_Data_tr_ts(data, feature_importance):
    """
    Process the top miRNAs based on feature importance and split the data into training and testing sets.
    Args:
        data (pd.DataFrame): DataFrame containing miRNA expression and tumor purity data.
        feature_importance (pd.DataFrame): DataFrame containing feature importance scores.
    Returns:
        tuple: Four lists containing training and testing sets for the top miRNAs and target values.
    """
    top_x_train=[] 
    top_x_test=[] 
    top_y_train=[] 
    top_y_test=[]
    y = data['CPE']
    
    for i in range(10):
        globals()['top{}'.format(i)] = list(feature_importance.index[0:i+1])
        globals()['top{}_x'.format(i)] = data[globals()['top{}'.format(i)]]
        
        # split data
        globals()['top{}_x_train'.format(i)], globals()['top{}_x_test'.format(i)], y_train, y_test = train_test_split(globals()['top{}_x'.format(i)],y, test_size=0.30, random_state=42)
        
        top_x_train.append(globals()['top{}_x_train'.format(i)]) 
        top_x_test.append(globals()['top{}_x_test'.format(i)]) 
        
        top_y_train.append(y_train) 
        top_y_test.append(y_test)
        
    return top_x_train, top_x_test, top_y_train, top_y_test


#### PCC (miRNA expression & tumor purity) ####
def miRNA_CPE_pcc(top10_x, y):
    """
    Calculate the Pearson correlation coefficient (PCC) between miRNA expression and tumor purity.
    Args:
        top10_x (pd.DataFrame): DataFrame containing the top 10 miRNAs.
        y (pd.Series): Series containing tumor purity values.
    Returns:
        pd.DataFrame: DataFrame containing the PCC values between each miRNA and tumor purity.
    """
    miRNA = top10_x
    miRNA = miRNA.astype(float)
    CPE = y
    CPE = CPE.astype(float)
    PCC_miRNA_CPE = pd.concat([miRNA, CPE], axis=1)
    
    PCC_miRNA_CPE_matrix = PCC_miRNA_CPE.corr()
    PCC = PCC_miRNA_CPE_matrix[['CPE']]
    
    return PCC


#### Scatter plot ####
def scatter_plot(data_folder, plot_name, Obes, Pred):
    """
    Create a scatter plot of observed vs predicted tumor purity and calculate the Pearson correlation coefficient (PCC).
    Args:
        data_folder (str): Path to the folder where the plot will be saved.
        plot_name (str): Name of the plot file.
        Obes (list): List of observed tumor purity values.
        Pred (list): List of predicted tumor purity values.
    Returns:
        float: Pearson correlation coefficient (PCC) between observed and predicted tumor purity.
    """
    obes = pd.DataFrame(Obes)
    pred = pd.DataFrame(Pred)
    
    obes.columns = ['CPE']
    obes['CPE'] = obes['CPE'].astype(float)
    pred.columns = ['predict']
    
    obes = obes.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    
    # obes & pred PCC
    pcc = obes['CPE'].corr(pred['predict'])
    print("PCC:", pcc)
    
    plt.scatter(obes['CPE'], pred['predict'], s=3)

    #regression line
    m, b = np.polyfit(obes['CPE'], pred['predict'], 1)
    plt.plot(obes['CPE'], m*obes['CPE']+b, color='red')

    # plot design
    plt.xlabel('Observed tumor purity')
    plt.ylabel('Predicted tumor purity')

    # plot font size
    small_size = 15
    large_size = 20

    plt.rc('axes', labelsize=small_size)   # x,y axis label font size
    plt.rc('figure', titlesize=large_size) # figure title font size
    #plt.show()

    # save plot
    plt.savefig(os.path.join(data_folder,"Scatter_plot_{}.png".format(plot_name)) , dpi=300)
    
    return pcc


#### PCAWG dataset processing [validation] ####
def PCAWG_purity(data_folder):
    """
    Prepare PCAWG miRNA expression and tumor purity data.
    Args:
        data_folder (str): Path to the folder containing the PCAWG data files.
    Returns:
        pd.DataFrame: Merged DataFrame containing PCAWG miRNA expression and tumor purity data.
    """
    warnings.filterwarnings("ignore")
    pcawg = pd.read_csv(os.path.join(data_folder,"x3t2m1.mature.UQ.mirna.matrix.log.txt"), sep ='\t', header = None)
    pcawg = pcawg.transpose()

    pcawg_header = pcawg.iloc[0]
    pcawg = pcawg.iloc[1:]
    pcawg.columns = pcawg_header
    pcawg.rename(columns={'mirna.name':'Sample ID'}, inplace=True)

    pcawg_purity = pd.read_csv(os.path.join(data_folder,"consensus.20170217.purity.ploidy_sp"), sep ='\t')
    pcawg_purity = pcawg_purity[['samplename', 'purity']]
    pcawg_purity.rename(columns={'samplename':'Sample ID'}, inplace=True)

    PCAWG = pd.merge(pcawg, pcawg_purity, how = 'inner',on = 'Sample ID')
    
    return PCAWG
