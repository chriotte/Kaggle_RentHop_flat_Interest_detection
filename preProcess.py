# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import collections

def price_percent_cut(df_NEW, col):
    price_low = np.percentile(df_NEW[col].values, 1)
    price_high = np.percentile(df_NEW[col].values, 99)

    df_NEW = df_NEW.drop(df_NEW[df_NEW.col < price_low].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.col > price_high].index)

    return df_NEW

# Datetime object and number of photos feature engineering
def clean_preprocess(initial_df):
    # convert created column into datetime type
    try:
        initial_df['DateTime'] = pd.to_datetime(initial_df.created)
        initial_df.drop('created', axis=1, inplace=True)

        # create feature for number of photos, features and description length
        initial_df['num_of_photos'] = initial_df.photos.map(len)
        initial_df['num_of_features'] = initial_df.features.map(len)
        initial_df['description_length'] = initial_df.description.apply(lambda x: len(x.split(" ")))
    except:
        print("Clean_Preprocessed function skipped as it can only be run once")
    return initial_df

# Remove prices outside of defined range
def price_outliers(df_NEW, price_low, price_high):
    df_NEW = df_NEW.drop(df_NEW[df_NEW.price < price_low].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.price > price_high].index)
    return df_NEW

# Remove locations outside of New York
def remove_nonNY_coords(df_NEW, ny_boundaries):
    #Removing out of bounds longitude
    df_NEW = df_NEW.drop(df_NEW[df_NEW.longitude < ny_boundaries[0]].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.longitude > ny_boundaries[1]].index)

    #Removing out of bounds latitude
    df_NEW = df_NEW.drop(df_NEW[df_NEW.latitude < ny_boundaries[2]].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.latitude > ny_boundaries[3]].index)

    return df_NEW

def buildingID_count(df):
    buildingID = df['building_id']
    count = collections.Counter()
    for buildings in buildingID:
        count[buildings] += 1
    return buildingID
        
def buildingID_interest(df):
    buildingID = df['building_id']
    interests  = df['interest_level']
    new_df = [buildingID, interests]
    
    return new_df

def buildingID_Interest_count(buildingID_interest, label):
    buildingID_Interest_low = {}
    count = collections.Counter()
    
    for i in buildingID_interest:
        if i[1] == label:
            count[buildingID_Interest_low[i]] += 1
    return buildingID_Interest_low

def buildingID_Interest_Ratio(df):
    id_ratio= {}
    uniqueID = df.building_id.unique()
    count = buildingID_count(df)
    id_interest     = buildingID_interest(df)
    id_interest_low = buildingID_Interest_count(id_interest, 'low')
    
    for items in uniqueID:
        ratio = id_interest_low[items] / count[items]
        id_ratio[items] = ratio
    
    return id_ratio
    
def euclid_dist(x, lat, long):
    return np.sqrt((x[0]-lat)**2 + (x[1]-long)**2)

def boroughs(df):
# distance from borough centres
    the_bronx     = [40.8448, -73.8648]
    manhattan     = [40.7831, -73.9712]
    queens        = [40.7282, -73.7949]
    brooklyn      = [40.6782, -73.9442]
    staten_island = [40.5795, -74.1502]
    
    borough_list = {'the_bronx': the_bronx,
                    'manhattan': manhattan,
                    'queens': queens,
                    'brooklyn': brooklyn,
                    'staten_island': staten_island}
    
    for key in borough_list:
        df[key] = df[['latitude','longitude']].apply(euclid_dist, 
                                                     args=(borough_list[key]), 
                                                     axis=1)
    return df

def main(df):
#==============================================================================
# Control panel for price and location data
#==============================================================================
    #price_low = 1000
    #price_high = 10000
    price_low = np.percentile(df['price'].values, 0.5)
    price_high = np.percentile(df['price'].values, 99.5)
    
    # Define upper and lower limits for NewYork
    long_low  = -74.1
    long_high = -73.6
    lat_low   =  35
    lat_high  =  41
    ny_boundaries = [long_low,long_high,lat_low,lat_high]
#==============================================================================
# Clean data and show how many rows of data are removed at each step
#==============================================================================
    dataCount = len(df)
    print(dataCount,"datapoints in dataset")
    
    df = clean_preprocess(df)
    newCount= len(df)
    print("cleanPreprocess removed",dataCount-newCount,"datapoints")
    dataCount=newCount
    
    df = remove_nonNY_coords(df, ny_boundaries)
    newCount= len(df)
    print("remove_nonNY_coords removed",dataCount-newCount,"datapoints")
    dataCount=newCount
    
    df = price_outliers(df, price_low, price_high)
    newCount= len(df)
    print("remove_outlier_prices removed",dataCount-newCount,"datapoints")
    
    print(newCount, "datapoints remaining")
    
    df = boroughs(df)

    return df