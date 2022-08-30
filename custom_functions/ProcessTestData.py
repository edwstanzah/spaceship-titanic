"""Functions used to treat the data from raw csv file to making predictions with selected model."""
import pandas as pd

def fill_missing_values(df):
    """Fill missing values the same way the train data is filled."""
    # drop Name
    df = df.drop('Name', axis=1)

    # impute Age
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # impute luxury amenities columns
    luxury_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    for col in luxury_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # make a new column showing the total luxury spend
    df['LuxurySpend'] = df['RoomService']+df['FoodCourt']+df['ShoppingMall']+df['Spa']+df['VRDeck']
    
    # impute HomePlanet with Europa if VIP is true where it is null. 
    df.loc[(df.VIP==True) & (df.HomePlanet.isnull()), 'HomePlanet'] = 'Europa'

    # impute HomePlanet with Europa if LuxurySpend is more than 6400 (max Earth spending)
    df.loc[(df.HomePlanet.isnull()) & (df.LuxurySpend>6400), 'HomePlanet'] = 'Europa'

    # impute the rest with Earth
    df.loc[df.HomePlanet.isnull(), 'HomePlanet'] = 'Earth'

    # impute missing CryoSleep
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])

    # impute VIP with the mode aka False
    df['VIP'] = df['VIP'].fillna(False)
    
    # impute destination with mode
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])

    # create CabinDeck
    df['CabinDeck'] = df['Cabin'].str.split('/').str[0]

    # drop Cabin
    df = df.drop('Cabin', axis=1)

    # impute missing CabinDeck
    df.loc[(df.HomePlanet=='Earth')&(df.CabinDeck.isnull()), 'CabinDeck'] = 'G'
    df.loc[(df.HomePlanet=='Mars')&(df.CabinDeck.isnull()), 'CabinDeck'] = 'F'
    df.loc[(df.HomePlanet=='Europa')&(df.CabinDeck.isnull()), 'CabinDeck'] = 'B'

    # drop luxuryspend
    df = df.drop('LuxurySpend',axis=1)

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


    
