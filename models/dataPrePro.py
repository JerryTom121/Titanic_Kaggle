'''
Created on Dec 25, 2016

@author: Nidhalios
'''

# remove warnings
import warnings


from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import RobustScaler

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
# ---

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

def status(feature):
    print('Processing',feature,': ok')

def get_combined_data():
    # reading train data
    train = pd.read_csv('../data/train.csv')
    
    # reading test data
    test = pd.read_csv('../data/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    
    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    
    return combined

def get_titles(combined):
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"
                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
    return combined

def process_names(combined):

    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    #combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)
    
    status('names')
    return combined

def process_fares(combined):

    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.Fare.median(),inplace=True)
    
    status('fare')
    return combined

def process_embarked(combined):

    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S',inplace=True)
    
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
    
    status('embarked')
    return combined

def process_cabin(combined):
    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    
    combined = pd.concat([combined,cabin_dummies],axis=1)
    
    combined.drop('Cabin',axis=1,inplace=True)
    
    status('cabin')
    return combined

def process_sex(combined):

    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    
    # Test drop Sex
    combined.drop('Sex',axis=1,inplace=True)
    
    status('sex')
    return combined

def process_age(combined):
    
    child_age = 18
    def get_person(passenger):
        title, age, sex, sibSp ,parch  = passenger
        if (age < child_age):
            temp = 'Child'
            if (sibSp + parch == 0): temp += '_Alone'
            else: temp += '_Not_Alone'
            return temp
        elif (sex == 'female'):
            temp = 'Female'
            if (title != 'Miss' and parch > 0): temp += '_Mother'
            else: temp += '_Not_Mother'
            return temp
        else:
            return 'male_adult'
        
    grouped = combined.groupby(['Sex','Pclass','Title'])
    grouped.median()
    combined["Age"] = combined.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    combined = pd.concat([combined, pd.DataFrame(combined[['Title', 'Age', 'Sex', 'SibSp', 'Parch']].apply(get_person, axis=1), columns=['Person'])],axis=1)
    combined = pd.concat([combined,pd.get_dummies(combined['Person'])],axis=1)
    # removing "Person" and "Age"
    combined.drop('Person',axis=1,inplace=True)
    combined.drop('Age',axis=1,inplace=True)
    
    status("Age")
    return combined

def process_pclass(combined):

    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    
    status('pclass')
    return combined

def process_family(combined):

    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
    
     # removing "FamilySize", "Parch", "SibSp"
    combined.drop('FamilySize',axis=1,inplace=True)
    combined.drop('Parch',axis=1,inplace=True)
    combined.drop('SibSp',axis=1,inplace=True)
    
    status('family')
    return combined

def process_ticket(combined):
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    
    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)
    
    status('ticket')
    return combined

def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

def recover_train_test_target(combined):

    train0 = pd.read_csv('../data/train.csv')
    targets = train0.Survived
    """
    scaler = RobustScaler()
    train = scaler.fit_transform(combined.ix[0:890], targets)
    train_df = pd.DataFrame(train, index=combined.ix[0:890].index, columns=combined.columns)
    test = scaler.transform(combined.ix[891:])
    test_df = pd.DataFrame(test, index=combined.ix[891:].index, columns=combined.columns)
    """
    train_df = combined.ix[0:890]
    test_df = combined.ix[891:]
    
    return train_df,test_df,targets