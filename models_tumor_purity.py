#### Package load ####
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Fixed seed so RF/MLP results are reproducible across runs
SEED = 42


#### Random Forest Regression model ####
def RFR(x_train, x_test, y_train, y_test):
    """
    Train a Random Forest Regressor model and evaluate its performance.
    Args:
        x_train (pd.DataFrame): Training features DataFrame.
        x_test (pd.DataFrame): Testing features DataFrame.
        y_train (pd.DataFrame): Training target values DataFrame.
        y_test (pd.DataFrame): Testing target values DataFrame.
    Returns:
        tuple: Best model, predictions on the test set, and mean squared error on the test
        set.
    """

    # best parameter model
    best_model = RandomForestRegressor(max_depth=20, n_estimators=500, n_jobs=-1, random_state=SEED)

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
    """
    Train and evaluate 8 different machine learning models using the top miRNAs.
    Args:
        top_x_train (pd.DataFrame): Training features DataFrame with top miRNAs.
        top_x_test (pd.DataFrame): Testing features DataFrame with top miRNAs.
        y_train (pd.DataFrame): Training target values DataFrame.
        y_test (pd.DataFrame): Testing target values DataFrame.
    Returns:
        dict: Model name -> mean squared error on the test set.
    """
    models = {
        "RFR": RandomForestRegressor(max_depth = 20, n_estimators = 500, n_jobs = -1, random_state = SEED),
        "Linear": linear_model.LinearRegression(n_jobs = -1),
        "Lasso": linear_model.Lasso(alpha = 0.000774263682681127, max_iter = 1000),
        "Ridge": linear_model.Ridge(alpha = 1.0, max_iter = 1000),
        "ElasticNet": linear_model.ElasticNet(alpha = 0.002154434690031882, l1_ratio = 0.55),
        "SVR": make_pipeline(StandardScaler(), SVR(C = 0.5, gamma = 'auto', kernel = 'rbf')),
        "KNR": KNeighborsRegressor(algorithm = 'ball_tree', leaf_size = 25,
                                   n_neighbors = 9, weights = 'distance', n_jobs = -1),
        "MLP": MLPRegressor(activation = 'logistic', solver = 'lbfgs',
                            learning_rate = 'adaptive', alpha = 1, random_state = SEED),
    }

    mse = {}
    for name, model in models.items():
        model.fit(top_x_train, y_train.values.ravel())
        mse[name] = mean_squared_error(y_test, model.predict(top_x_test))
        print("{}_MSE_test: ".format(name), mse[name])

    return mse
