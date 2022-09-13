"""Functions used to treat the data from raw csv file to making predictions with selected model."""
import pandas as pd
import numpy as np
from collections import Counter
import random
random.seed(42)

def preprocessing(data: pd.DataFrame):
    """Return feature engineered and clean dataframe of the train/test dataset"""

    # create new features based on spending columns
    spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    data['LuxurySpend'] = data[spending_columns].sum(axis=1) # total spending
    data['ZeroSpend'] = (data['LuxurySpend']==0) # True for no spend

    # split PassengerId to create new features
    data[['PassengerGroup', 'PassengerGroupNumber']] = data['PassengerId'].str.split('_', expand=True)

    # convert PassengerGroup to int
    data['PassengerGroup'] = data['PassengerGroup'].astype(int)

    # get the size of the group by counting element's occurrence in PassengerGroup
    group_counter = np.vectorize(Counter(data.PassengerGroup).get) # initialize counter object and vectorize the function

    data['GroupSize'] = group_counter(data.PassengerGroup) # create a new column to store the group size feature

    # create a new feature, True if alone, False if not.
    data['Alone'] = (data['GroupSize']==1)

    # create three new columns from the Cabin column
    data[['CabinDeck', 'CabinNum', 'CabinSide']] = data['Cabin'].str.split('/', expand=True)

    # convert CabinNum to integer and preserve nan values
    data['CabinNum']= [int(num) if num is not np.nan else num  \
                        for num in data.CabinNum]
    
    # create new feature, CabinGroup
    data.loc[data.CabinNum <= 300, 'CabinGroup'] = 'Group 1'
    data.loc[(300 < data.CabinNum) & (data.CabinNum <= 600), 'CabinGroup'] = 'Group 2'
    data.loc[(600 < data.CabinNum) & (data.CabinNum <= 1200), 'CabinGroup'] = 'Group 3'
    data.loc[(1200 <data.CabinNum) & (data.CabinNum <= 1500), 'CabinGroup'] = 'Group 4'
    data.loc[data.CabinNum > 1500, 'CabinGroup'] = 'Group 5'

    data['LastName'] = data['Name'].str.split().str[-1]
    
    name_counter = np.vectorize(Counter(data['LastName']).get)
    data['FamilySize'] = name_counter(data['LastName'])

    # impute missing values
    data = impute_missing(data)

    # drop unnecessary columns
    cols_to_drop = ['Cabin', 'CabinNum', 'Name', 'LastName', 'PassengerGroup', 'PassengerGroupNumber', 'PassengerId']
    clean_data = data.drop(cols_to_drop, axis=1)
       
    return clean_data


def impute_missing(data: pd.DataFrame):
    """Impute missing values of the train/test dataset"""
    # impute last name

        # fill missing last name with respect to passenger group
    func = lambda x: x.mode()[0] if x.notna().any() else np.nan
    data['LastName'] = data.groupby(['PassengerGroup'])['LastName'].transform(func)

        # recount family size
    counter = np.vectorize(Counter(data['LastName']).get)
    data['FamilySize'] = counter(data['LastName'])

        # change row with missing last name to 1 family size
    data.loc[data['FamilySize']>100, 'FamilySize'] = 1

    # impute HomePlanet
        # fill missing homeplanet with respect to last name
    data['HomePlanet'] = data.groupby(['LastName'])['HomePlanet'].transform(func)

        # fill missing values with respect to cabin deck
    data.loc[(data.CabinDeck=='G')&(data.HomePlanet.isna()), 'HomePlanet'] = 'Earth'
    data.loc[(data.CabinDeck.isin(['A','B','C','T']))&(data.HomePlanet.isna()), 'HomePlanet']='Europa'

        # fill missing values with respect to spending
    data.loc[(data.LuxurySpend>6400)&(data.HomePlanet.isna()), 'HomePlanet']='Europa'

        # fill the rest with mode (Earth)
    data['HomePlanet'] = data['HomePlanet'].fillna('Earth')

    # impute Cabins
        # impute CabinSide
    data['CabinSide'] = data.groupby(['PassengerGroup'])['CabinSide'].transform(func)
        # impute remaining with random
    sides = ['S', 'P']
    data.loc[data.CabinSide.isna(), 'CabinSide'] = [random.choice(sides) for missing_value in data.loc[data.CabinSide.isna(), 'CabinSide']]

        # impute CabinDeck
    data.loc[(data.CabinDeck.isna())&(data.HomePlanet=='Earth'), 'CabinDeck'] = 'G'
    data.loc[(data.CabinDeck.isna())&(data.HomePlanet=='Mars'), 'CabinDeck'] = 'F'
    data.loc[(data.CabinDeck.isna())&(data.HomePlanet=='Europa'), 'CabinDeck'] = 'B'

        # impute CabinGroup
    data.loc[(data.CabinGroup.isna())&(data.PassengerGroup <=2000), 'CabinGroup'] = 'Group 1'
    data.loc[(data.CabinGroup.isna())&(data.PassengerGroup <=4000), 'CabinGroup'] = 'Group 2'
    data.loc[(data.CabinGroup.isna())&(data.PassengerGroup <=6000), 'CabinGroup'] = 'Group 3'
    data.loc[(data.CabinGroup.isna())&(data.PassengerGroup <=8000), 'CabinGroup'] = 'Group 4'
    data['CabinGroup'] = data['CabinGroup'].fillna('Group 5')

    # impute CryoSleep, Destination, and VIP
    data['CryoSleep'] = data['CryoSleep'].fillna(data.CryoSleep.mode()[0])
    data['Destination'] = data['Destination'].fillna(data.Destination.mode()[0])
    data['VIP'] = data['VIP'].fillna(data.VIP.mode()[0])

    # impute Age
    data['Age'] = data.groupby(['HomePlanet', 'ZeroSpend', 'Alone'])['Age'].transform(lambda x: x.fillna(x.median()))

    # impute expenditure columns
    luxury_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    for col in luxury_cols:
        data[col] = data[col].fillna(data[col].median())

    return data  




