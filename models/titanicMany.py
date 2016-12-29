'''
Created on Dec 13, 2016

@author: Nidhalios
'''

# remove warnings
import warnings

from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import models.dataPrePro as prepro

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from scipy import stats

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

    sns.distplot(train['Fare'])
    plt.show()

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
    
    classifiers = [
    XGBClassifier(),
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

    log_cols = ["Classifier", "Accuracy"]
    log      = pd.DataFrame(columns=log_cols)
    
    for clf in classifiers:
        name = clf.__class__.__name__
        acc = prepro.compute_score(clf, train_new, targets, "accuracy")
        log_entry = pd.DataFrame([[name, acc]], columns=log_cols)
        log = log.append(log_entry)

    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy')
    
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
    plt.show()
    
    print(log.sort('Accuracy', ascending=False).head(3))
    estimators = []
    estimators.append(('LDA', LinearDiscriminantAnalysis()))
    estimators.append(('RF', RandomForestClassifier(max_depth=8, n_estimators=150,
                                                    criterion='entropy')))
    
    #ensemble = VotingClassifier(estimators)
    #results =  prepro.compute_score(ensemble, train_new, targets, "accuracy")
    #print("Ensemble Accuracy : ",results.mean())
    
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[x[1] for x in estimators], 
                          meta_classifier=lr)

    print('5-fold cross validation:\n')
    estimators.append(('Stack', sclf))
    for label, clf  in estimators:
    
        scores = cross_val_score(clf, train_new, targets, cv=5, scoring='accuracy')
        print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))
    
    candidate_classifier = sclf
    candidate_classifier.fit(train_new, targets)
    output = candidate_classifier.predict(test_new).astype(int)

    df_output = pd.DataFrame()
    df_output['PassengerId'] = test['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId','Survived']].to_csv('../data/output.csv',index=False)


if __name__ == '__main__': main()