import pandas as pd
import numpy as np
import math

def convert_dtype(x):
    if not x:
        return None
    try:
        return float(x)
    except:
        return None

def partitionData(category):
    df2 = pd.read_csv('~/src/data/RPI_Expertise_2016_Features.csv')
    dt = {}

    for i in df2.index:
        dt[i] = convert_dtype

    df = pd.read_csv('~/src/data/RPI_Expertise_2016_Features.csv', converters=dt)

    # print(df.columns)

    ### getting rid of a bunch of unnamed empty columns
    df.drop(df.iloc[:, 72:], inplace=True, axis=1)
    ### getting rid of last row since its empty
    df.drop(index=df.index[-1], axis=0, inplace=True)
    

    if (category == 'basic'):
        start = 'friends'
        end = 'per_rt'
        Df = df.loc[:, start:end]
    elif (category == 'language'):
        start = 'chars'
        end = 'punc_rt'
        Df = df.loc[:, start:end]
    elif (category == 'interactivity'):
        start = 'tagpermsg'
        end = 'percent_msgwithurl_rt'
        Df = df.loc[:, start:end]
    elif (category == 'coherence'):
        df[' topic_coherence'] = df[' topic_coherence'].fillna(df['Unnamed: 71']) 
        Df = df[['lexco', ' topic_coherence']]
    else :
        print("selected category does not exit")
        exit()
    
    Df['followers'] = df['followers'].dropna()
    Df.fillna(df.mean(), inplace=True)
    Df = Df[df.followers != 0] # REMOVE ZERO FOR LOG SCALE
    Df['followers'] = Df['followers'].apply(lambda x: math.log10(x)) # LOGARITHMIC SCALE

    print("Proceeding to process dataframe:\n", Df.head(5))

    return Df
