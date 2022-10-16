from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score

def corr_matrix(df):
    
    corr_matrix = df.corr(method='pearson')
    follower_corr = corr_matrix['followers'].sort_values(key=abs)

    return follower_corr

def nmi_follower_matrix(df):
    follower_bins = df['followers'].apply(bin_function, args = (df['followers'].max(),))
    df.drop('followers', axis=1, inplace=True)
    nmi_list = []

    for column in df.columns:
        current_column = df[column].apply(bin_function, args = (df[column].max(),))
        nmi = normalized_mutual_info_score(follower_bins, 
                             current_column, 
                             average_method='min')
        
        nmi_list.append((str(column), nmi))

    nmi_df = pd.DataFrame(nmi_list, columns=['Feature', 'NMI with followers']).sort_values(by=['NMI with followers'], key=abs)
    nmi_df.reset_index(drop=True, inplace=True)

    return nmi_df
    
def train_test_split(df):
    y = df['followers'].to_numpy()
    # y = df['followers'].apply(lambda x: x/df['followers'].max()).to_numpy() # NORMALISE
    X = df.drop('followers', axis=1).to_numpy()
    k = 5 # CAN BE ADJUSTED
    kf_CV = KFold(n_splits=k, shuffle=True, random_state=43) # remove random_state if you want 
                                                             # randomness, not reproducibility
    tt_split = {}                                                            
    for train_idx, test_idx in kf_CV.split(X):
        tt_split['X_train'], tt_split['X_test'] = X[train_idx], X[test_idx]
        tt_split['y_train'], tt_split['y_test'] = y[train_idx], y[test_idx]
    return tt_split

def multiregression(tt_split): 
    X_train = tt_split['X_train']
    y_train = tt_split['y_train']
    X_test = tt_split['X_test']
    y_test = tt_split['y_test']

    # Create and fit the linear model
    # lm = Linear Model (variable name)
    lm = LinearRegression()

    # Fit to the train dataset
    lm.fit(X_train, y_train)

    print("Intercept:", lm.intercept_)
    print("Coefficients:", lm.coef_)

    y_pred = lm.predict(X_test)
    r2 = lm.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Regression R^2 score:", r2, "\n")
    print("Regression mean squared error:", mse)

    regression_plot(y_test, y_pred)

    # subtract the predicted values from the observed values
    residuals = y_test - y_pred 

    residual_plot(y_pred, residuals)
    

def regression_plot(y_test, y_pred):
    plt.scatter(y_test, y_pred, alpha=0.3)

    plt.title('Linear Regression (Log Followers)')
    plt.xlabel('Actual Value (Log 10)')
    plt.ylabel('Predicted Value (Log 10)')

    plt.show()
    plt.savefig('linear_regression.png')
    plt.close()

def residual_plot(y_pred, residuals):
        # plot residuals
    plt.scatter(y_pred, residuals, alpha=0.3)

    # plot the 0 line (we want our residuals close to 0)
    plt.plot([min(y_pred), max(y_pred)], [0,0], color='red')

    plt.title('Residual Plot (Log 10 Scale)')
    plt.xlabel('Predicted Value (Log 10)')
    plt.ylabel('Residual')

    plt.show()
    plt.savefig('residual_plot.png')

def bin_function(value, valueMax):
    x = value/valueMax * 10
    return math.ceil(x)

