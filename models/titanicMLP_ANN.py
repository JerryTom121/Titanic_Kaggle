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

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

import pandas as pd
import models.dataPrePro as prepro
from sklearn.metrics.scorer import accuracy_scorer, roc_auc_scorer

warnings.filterwarnings('ignore')
# ---


def main():
    combined = prepro.get_combined_data()
    print(combined.shape)
    combined = prepro.get_titles(combined)
    grouped = combined.groupby(['Sex','Pclass','Title'])
    grouped.median()
    combined["Age"] = combined.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    prepro.status("Age")
    combined = prepro.process_names(combined)
    combined = prepro.process_fares(combined)
    combined = prepro.process_embarked(combined)
    combined = prepro.process_cabin(combined)
    combined = prepro.process_sex(combined)
    combined = prepro.process_pclass(combined)
    combined = prepro.process_family(combined)
    combined = prepro.process_ticket(combined)
    combined = prepro.scale_all_features(combined)
    train, test, targets = prepro.recover_train_test_target(combined)
    
    clf = RandomForestClassifier(n_estimators=200)
    clf = clf.fit(train, targets)
    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    train_new = model.transform(train)
    test_new = model.transform(test)
    
    # 'activiation': ['identity', 'logistic', 'tanh', 'relu']
    par_grid={   
    'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
    'solver': ['adam','lbfgs', 'sgd']}
    
    mlp = MLPClassifier(activation='tanh', early_stopping=True, hidden_layer_sizes=(50, 50), 
                        learning_rate='constant', max_iter=200, validation_fraction=0.1, 
                        warm_start=False)
    
    cross_validation = StratifiedKFold(targets, n_folds=5)
    
    grid_search = GridSearchCV(mlp,
                               param_grid=par_grid,
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