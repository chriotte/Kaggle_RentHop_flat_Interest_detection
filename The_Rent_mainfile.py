# coding: utf-8

# In[1]:
#==============================================================================
# Import libraries
#==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#get_ipython().magic('matplotlib inline')

mpl.rc("savefig", dpi=300)      ## What is this doing?

# In[2]:
#==============================================================================
# Read data and create dataframes
#==============================================================================
df = pd.read_json("train.json")

#Create dataframes bsaed on interest levels
df_low      = df.drop(df[df.interest_level != "low"].index)
df_medium   = df.drop(df[df.interest_level != "medium"].index)
df_high     = df.drop(df[df.interest_level != "high"].index)

# In[3]:

df.head(3)
df.describe()

# In[4]:
#==============================================================================
# Data Preprocessing and Feature Engineering
#==============================================================================
# cut data by 1st and 99th percentile
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

        # create feature for number of photos
        initial_df['num_of_photos'] = initial_df.photos.map(len)
    except:
        print("Clean_Preprocessed function skipped as it can only be run once")
    return initial_df


# Remove prices outside of defined range
def remove_outlier_prices(df_NEW):    
    df_NEW = df_NEW.drop(df_NEW[df_NEW.price < price_low].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.price > price_high].index)    
    return df_NEW

# Remove locations outside of New York
def remove_nonNY_coords(df_NEW):    
    #Removing out of bounds longitude
    df_NEW = df_NEW.drop(df_NEW[df_NEW.longitude < long_low].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.longitude > long_high].index)

    #Removing out of bounds latitude
    df_NEW = df_NEW.drop(df_NEW[df_NEW.latitude < lat_low].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.latitude > lat_high].index)

    return df_NEW

#==============================================================================
# Control panel for price and location data
#==============================================================================
price_low = 1000
#price_high = 10000
#price_low = np.percentile(df['price'].values, 0.5)
price_high = np.percentile(df['price'].values, 99)

# Define upper and lower limits for NewYork
long_low  = -74.1
long_high = -73.6
lat_low   =  35
lat_high  =  41

# In[5]:
#==============================================================================
# Clean data and show how many rows of data are removed at each step
#==============================================================================
dataCount = len(df)
print(dataCount,"datapoints in dataset")

df = clean_preprocess(df)
newCount= len(df)
print("cleanPreprocess removed",dataCount-newCount,"datapoints")
dataCount=newCount

df = remove_nonNY_coords(df)
newCount= len(df)
print("remove_nonNY_coords removed",dataCount-newCount,"datapoints")
dataCount=newCount

df = remove_outlier_prices(df)
newCount= len(df)
print("remove_outlier_prices removed",dataCount-newCount,"datapoints")

print(newCount, "datapoints remaining")


# In[6]:
#==============================================================================
# Price plotting
#==============================================================================
plt.hist(df['price'], 100)
plt.title("Distribution with top 1% removed")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# In[]:
#==============================================================================
# Location plotting
#==============================================================================
plt.hist(df['price'], 100)
plt.title("Distribution with top 1% removed")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()


# In[7]:
#==============================================================================
# Feature Creation
#==============================================================================
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

def euclid_dist(x, lat, long):
    return np.sqrt((x[0]-lat)**2 + (x[1]-long)**2)

for key in borough_list:
    df[key] = df[['latitude','longitude']].apply(euclid_dist, args=(borough_list[key]), axis=1)


# ### Description BoW - TO FINISH
import nltk
from nltk.stem import WordNetLemmatizer
import re, html


description = "A Brand New 3 Bedroom 1.5 bath ApartmentEnjoy These Following Apartment Features As You Rent Here? Modern Designed Bathroom w/ a Deep Spa Soaking Tub? Room to Room AC/Heat? Real Oak Hardwood Floors? Rain Forest Shower Head? SS steel Appliances w/ Chef Gas Cook Oven & LG Fridge? washer /dryer in the apt? Cable Internet Ready? Granite Counter Top Kitchen w/ lot of cabinet storage spaceIt's Just A Few blocks To L Train<br /><br />Don't miss out!<br /><br />We have several great apartments in the immediate area.<br /><br />For additional information 687-878-2229<p><a  website_redacted" 

wordFreqDict = {}
tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')

def makeFreqDict(description):
# takes a string, splits it up and add the occurances of each word to the dictionary
    no_tags = tag_re.sub('', description)
    description = html.escape(no_tags)   
    words = nltk.tokenize.word_tokenize(description)
    
    unimportant_words = [':', 'http', '.', ',', '?', '...', "'s", "n't", 'RT', ';', '&', ')', '``', 'u', '(', "''", '|',]
    for word in words:
        if word not in unimportant_words:
            word = WordNetLemmatizer().lemmatize(word)
    
            if word in wordFreqDict:
                wordFreqDict[word] += 1
            else:
                wordFreqDict[word] = 1
                        
makeFreqDict(description)

# In[10]:
#==============================================================================
# EDA - Column Headers
#==============================================================================
df.columns.tolist()

# In[11]:
#==============================================================================
# EDA - General
#==============================================================================
# plot of interest levels
interest_cat = df.interest_level.value_counts()
x = interest_cat.index
y = interest_cat.values

sns.barplot(x, y)
plt.ylabel("Count")
plt.xlabel("Interest Level")

print(df.interest_level.value_counts())

# In[12]:
#==============================================================================
# EDA - Geospatial - MAKE MORE FANCY PLOTS WITH THIS - THIS ONE SUCKS
#==============================================================================
#position data: longitude/latitude

sns.pairplot(df[['longitude', 'latitude', 'interest_level']], hue='interest_level')
plt.ylabel('latitude')
plt.xlabel('longitude')

# In[13]:
#==============================================================================
# EDA - bedrooms
#==============================================================================
# bedrooms plot
sns.countplot(x='bedrooms',hue='interest_level', data=df)
plt.ylabel('Occurances')
plt.xlabel('Number of bedrooms')

# In[14]:
#==============================================================================
# EDA - price
#==============================================================================

sns.violinplot(x="interest_level", y="price", data=df, palette="PRGn", order=['low','medium','high'])
sns.despine(offset=10, trim=True)
plt.ylabel('price per month USD')
plt.ylim(0,17500)
plt.title("Violin plot showing distribution of rental prices by interest level")

# plotting median lines
# plt.axhline(df.price[df.interest_level == 'low'].median(), linewidth = 0.25, c='purple')
# plt.axhline(df.price[df.interest_level == 'medium'].median(), linewidth = 0.25, c='black')
# plt.axhline(df.price[df.interest_level == 'high'].median(), linewidth = 0.25, c='green')

print("Mean price per interest level \n", df[['price','interest_level']].groupby('interest_level').mean(), "\n")
print("STD of price per interest level \n", df[['price','interest_level']].groupby('interest_level').std())

# In[15]:

sns.distplot(df.price[df.interest_level == 'low'], hist=False, label='low')
sns.distplot(df.price[df.interest_level == 'medium'], hist=False, label='medium')
sns.distplot(df.price[df.interest_level == 'high'], hist=False, label='high')

# In[16]:
#==============================================================================
# OTHER NOTES - REMOVE AT SOME POINT
#==============================================================================
"""
Considerations with the data:
    imbalanced dataset (not many high interest apartments compared to the rest)
    

Plots to produce
    barplot of interest levels - done
    map of interest levels
    price map
    

Features to use
    bathrooms
    bedrooms
    price

Additional features to create:
    Number of images
    description length
    creation year, month, day
    description word frequency - create features out of top x words
    distance to borough centres


Target:
    Interest Level
"""
# In[17]:
#==============================================================================
# Modelling
#==============================================================================
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score as cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper

# In[18]:
#==============================================================================
# Splitting Dataset
#==============================================================================

# determine features to use for modelling prior to data split
features_to_use = ['bathrooms','bedrooms','price', 'the_bronx', 'staten_island','manhattan','queens','brooklyn', 'num_of_photos']
X_all = df[features_to_use]

# convert target label into numerical (ordinal)
target_conversion = {'low':0,'medium':1,'high':2}
y_all = df.interest_level.map(target_conversion).values

X_train_val, X_test, y_train_val, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=0, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=0, stratify=y_train_val)

# mapping scaler to keep dataset in a dataframe (cannot do inverse using this function)
scaler = DataFrameMapper([(X_all.columns, StandardScaler())])
#scaler = StandardScaler()

# learn scale parameters from final training set and apply to training, val, and test sets

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)


X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
X_val_df = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)

# some algorithms require dummy values for multiclass classification
# target = pd.get_dummies(train.interest_level)

# In[19]:
#==============================================================================
# Baseline Model
#==============================================================================

# define model params
model = RandomForestClassifier(n_estimators=100)

# train model
model.fit(X_train_scaled, y_train)

# evaluation
y_hat_train = model.predict(X_train_scaled)
y_hat_val = model.predict(X_val_scaled)

# confusion matrices - predicted class along the top, actual class down the side (low, medium, high)
print("training confusion matrix \n", confusion_matrix(y_train, y_hat_train, labels=[0,1,2]), "\n")
print("validation confusion matrix \n", confusion_matrix(y_val, y_hat_val, labels=[0,1,2]), "\n")

y_hat_train = model.predict_proba(X_train_scaled)
y_hat_val = model.predict_proba(X_val_scaled)

# log loss evaluations for train, val
print("log loss - training:", log_loss(y_train, y_hat_train))
print("log loss - validation:", log_loss(y_val, y_hat_val))

# In[20]:
#==============================================================================
# Modelling TODO
#==============================================================================

# feature importance measures

#from sklearn.ensemble import ExtraTreesClassifier
#clf = ExtraTreesClassifier()
#clf = clf.fit(features, target)
#clf.feature_importances_



