'''
Created on Dec 13, 2016

@author: Nidhalios
'''

# remove warnings
import warnings

from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectFromModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.grid_search import GridSearchCV

import seaborn as sns
import lightgbm as lgb
import pandas as pd
import numpy as np
import models.dataPrePro as prepro
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.metrics.scorer import accuracy_scorer

warnings.filterwarnings('ignore')
# ---

def calculate_vif_(X):

    '''X - pandas dataframe'''
    variables = range(X.shape[1])
    tab = []
    for i in np.arange(0, len(variables)):
        vif = variance_inflation_factor(X.values, i)
        tab.append(vif)
    return tab

def main():
    combined = prepro.get_combined_data()
    print(combined.shape)
    combined = prepro.get_titles(combined)
    combined = prepro.process_age(combined)
    combined = prepro.process_cabin(combined)
    combined = prepro.process_names(combined)
    combined = prepro.process_fares(combined)
    g = lambda x: x+2
    combined['Fare'] = stats.boxcox(combined['Fare'].apply(g))[0]
    combined = prepro.process_embarked(combined)
    combined = prepro.process_sex(combined)
    combined = prepro.process_pclass(combined)
    combined = prepro.process_family(combined)
    combined = prepro.process_ticket(combined)
    train, test, targets = prepro.recover_train_test_target(combined)

    clf = RandomForestClassifier(n_estimators=200)
    clf = clf.fit(train, targets)
    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = clf.feature_importances_
    print(features.sort('importance', ascending=False).head(10))
    model = SelectFromModel(clf, prefit=True)
    train_new = model.transform(train)
    test_new = model.transform(test)
    
    # As suspected many features are causing multicollinearity, here I am calculating the 
    # value inflation factor for the top selected features and trying to resolve the problem
    # accordingly in the data processing/engineering code (dataPrePro) 
    h = zip(features['feature'][model.get_support(indices=True)],
            calculate_vif_(pd.DataFrame(train_new)))
    for x, y in h:
        print(x," ",y)
    
    train_data = lgb.Dataset(train_new, label=targets)
    
    param_grid = {
    'boosting_type' :['gbdt','dart'],
    'num_leaves': [20,30,50,100],
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40],
    'is_unbalance': [True,False]
    }

    bst = lgb.LGBMClassifier(objective='binary',silent=False)
    gbm = GridSearchCV(bst, param_grid, scoring=accuracy_scorer)
    gbm.fit(train_new, targets)
    
    print('Best score: {}'.format(gbm.best_score_))
    print('Best parameters: {}'.format(gbm.best_params_))
    
    output = gbm.predict(test_new).astype(int)
    df_output = pd.DataFrame()
    df_output['PassengerId'] = test['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId','Survived']].to_csv('../data/output.csv',index=False)
    print(output)

if __name__ == '__main__': main()