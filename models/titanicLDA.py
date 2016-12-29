'''
Created on Dec 13, 2016

@author: Nidhalios
'''
# remove warnings
import warnings

from matplotlib import pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd
import models.dataPrePro as prepro
from sklearn.metrics.scorer import accuracy_scorer

warnings.filterwarnings('ignore')
# ---

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

def main():
    combined = prepro.get_combined_data()
    combined = prepro.get_titles(combined)
    combined = prepro.process_age(combined)
    combined = prepro.process_names(combined)
    combined = prepro.process_fares(combined)
    combined = prepro.process_embarked(combined)
    combined = prepro.process_cabin(combined)
    combined = prepro.process_sex(combined)
    combined = prepro.process_pclass(combined)
    combined = prepro.process_family(combined)
    combined = prepro.process_ticket(combined)
    combined = prepro.scale_all_features(combined)
    train,test,targets = prepro.recover_train_test_target(combined)
    clf = RandomForestClassifier(n_estimators=200)
    clf = clf.fit(train, targets)
    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = clf.feature_importances_
    print(features.sort(['importance'],ascending=False).head(10))
    model = SelectFromModel(clf, prefit=True)
    train_new = model.transform(train)
    test_new = model.transform(test)
    lda = LinearDiscriminantAnalysis()
    parameter_grid = {
                     #'tol': [0.0001,0.0005,0.001],
                     'solver': ['svd','lsqr','eigen']
                     }
    
    cross_validation = StratifiedKFold(targets, n_folds=5)
    
    grid_search = GridSearchCV(lda,
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               scoring=accuracy_scorer)
    
    grid_search.fit(train_new, targets)
    
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
    output = grid_search.predict(test_new).astype(int)
    df_output = pd.DataFrame()
    df_output['PassengerId'] = test['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId','Survived']].to_csv('../data/output.csv',index=False)

if __name__ == '__main__': main()