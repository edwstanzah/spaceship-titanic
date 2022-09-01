"""Functions used to treat the data from raw csv file to making predictions with selected model."""
import pandas as pd
import numpy as np

def fill_missing_values(df):
    """Fill missing values the same way the train data is filled."""
    # drop Name
    

    # impute Age
    df['Age'] = df['Age'].fillna(27)

    # impute luxury amenities columns
    luxury_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    for col in luxury_cols:
        df[col] = df[col].fillna(0)
    
    # make a new column showing the total luxury spend
    df['LuxurySpend'] = df['RoomService']+df['FoodCourt']+df['ShoppingMall']+df['Spa']+df['VRDeck']

    # impute HomePlanet with Europa if VIP is true where it is null. 
    df.loc[(df.VIP==True) & (df.HomePlanet.isnull()), 'HomePlanet'] = 'Europa'
    
    # impute HomePlanet with Europa if LuxurySpend is more than 6400 (max Earth spending)
    df.loc[(df.HomePlanet.isnull()) & (df.LuxurySpend>6400), 'HomePlanet'] = 'Europa'

    # impute the rest with Earth
    df.loc[df.HomePlanet.isnull(), 'HomePlanet'] = 'Earth'

    # impute HomePlanet with Europa if VIP is true where it is null. 
    df.loc[(df.VIP==True) & (df.HomePlanet.isnull()), 'HomePlanet'] = 'Europa'

    # impute missing CryoSleep
    df['CryoSleep'] = df['CryoSleep'].fillna(False)

    # impute VIP with the mode aka False
    df['VIP'] = df['VIP'].fillna(False)
    
    # impute destination with mode
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])

    # create CabinDeck, Num, and Side
    df[['CabinDeck', 'CabinNum', 'CabinSide']] = df['Cabin'].str.split('/', expand=True)

    # drop Cabin
    df = df.drop('Cabin', axis=1)

    # impute missing CabinDeck
    df.loc[(df.HomePlanet=='Earth')&(df.CabinDeck.isnull()), 'CabinDeck'] = 'G'
    df.loc[(df.HomePlanet=='Mars')&(df.CabinDeck.isnull()), 'CabinDeck'] = 'F'
    df.loc[(df.HomePlanet=='Europa')&(df.CabinDeck.isnull()), 'CabinDeck'] = 'B'

    # convert cabin num to integer while preserving the nan values
    df['CabinNum']= [int(num) if num is not np.nan else num for num in df.CabinNum]

    df.loc[df.CabinNum <= 300, 'CabinGroup'] = 'Group 1'
    df.loc[(300 < df.CabinNum) & (df.CabinNum <= 600), 'CabinGroup'] = 'Group 2'
    df.loc[(600 < df.CabinNum) & (df.CabinNum <= 1200), 'CabinGroup'] = 'Group 3'
    df.loc[(1200 < df.CabinNum) & (df.CabinNum <= 1500), 'CabinGroup'] = 'Group 4'
    df.loc[df.CabinNum > 1500, 'CabinGroup'] = 'Group 5'

    # impute cabin group
    df['CabinGroup'] = df['CabinGroup'].fillna(df.CabinGroup.mode()[0])

    # impute cabin side
    df['CabinSide'] = df['CabinSide'].fillna(df.CabinSide.mode()[0])

    # drop luxuryspend
    df = df.drop('LuxurySpend',axis=1)

    # drop Name
    df = df.drop('Name', axis=1)

    # drop cabinnum
    df = df.drop('CabinNum', axis=1)

    return df


def create_features(df):
    """Create new features/columns for dataframe df"""
    def alone_or_group(group_size):
        if group_size == 1:
            return 0 # 0 for alone'
        else:
            return 1 # 1 for, yes travel in a group.
    df['GroupSize'] = df.PassengerId.str.split("_").str[1].str[1].apply(int)
    df['GroupBool'] = df['GroupSize'].apply(alone_or_group)
    
    # drop GroupSize
    df = df.drop('GroupSize', axis=1)
    
    return df


def binary_variables(df):
    """encode binary categories into binary"""
    for col in ['CryoSleep', 'VIP']:
        df[col] = df[col].map({True: 1, False: 0})
    
    return df


def create_dummies(df):
    """Create dummy variables"""
    ids_ = df['PassengerId']
    features = df.drop('PassengerId', axis=1)

    features = pd.get_dummies(features, drop_first=True)

    df = pd.concat([ids_, features], axis=1)

    return df


def full_processing(df):
    """Combine all methods to one."""
    df = fill_missing_values(df)
    df = create_features(df)
    df = binary_variables(df)
    df = create_dummies(df)

    return df


    
