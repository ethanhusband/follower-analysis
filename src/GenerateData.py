import random
import sys
import seaborn as sns
import pandas as pd
import numpy as np
from Methods import corr_matrix, multiregression, nmi_follower_matrix, train_test_split
from PartitionData import partitionData

def generateData(category):
    df = partitionData(category)
    processDf(df)


# This is ultimately the function we wanna call
# Generate all the data for your section of the dataset
def processDf(df):
    with open('output.txt', 'w') as f:
        sys.stdout = f
        print("Mean followers:", df['followers'].mean())
        print("Median followers:", df['followers'].median())
        print("\n")

        print("Correlation Matrix:\n", corr_matrix(df))
        # Correlation matrix
            # Print correlation matrix
        print("\n")

        # Mutual information matrix,
        print("NMI Matrix:\n", nmi_follower_matrix(df.copy(deep=True)))
        print("\n")

        tt_split = train_test_split(df)
        print("Train-Test Split Created.\n")
        # Train-test split
            # Output df 'tt_split' with: X_train, y_train, X_test, y_test

        print("Multiregression Data:\n")
        multiregression(tt_split)
        # Fit data to multiregression model
            # Output R^2 score and Mean Squares Error (essentially, the measure of accuracy)
            # Output graph of predicted vs actual value
            # Output Residual plot for appendix   




