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
import preProcess as pre # Created a module for preprocessing

#get_ipython().magic('matplotlib inline')

# alter dpi to change the figure resolution, 100ish for general use, 300 for report
mpl.rc("savefig", dpi=100)

# Read data, create dataframes and clean it
df_raw = pd.read_json("train.json")
df = pre.main(df_raw)

#df_low      = df_raw.drop(df_raw[df_raw.interest_level != "low"].index)
# In[3]:
#==============================================================================
# Download dataframe to excel for exploration
#==============================================================================
#df.to_excel("cleanData.xlsb")
#df.to_csv("cleanData.csv")

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

# In[]
# Assess each brokers quality 


# 1 get each broker
# 2 get number of listings that are high or medium
# 3 percentage of listings 


def makeFeatureQuality(strName):
    QualityTemp = (df.groupby(strName)['interest_level'].apply(list)).to_dict()
    
    for key in QualityTemp:
        qualList = QualityTemp[key]
        listLength = len(qualList)
        totalScore = 1
        for item in qualList:
            if item == "low":
                item = 0
            elif item == "medium":
                item = 1
            elif item == "high":
                item = 1
            else:
                item = -99999
            totalScore =+ item
        totalScore = totalScore / listLength
        QualityTemp[key] = [totalScore]
    return QualityTemp

# adding the new value to the dataframe
managerID = 'manager_id'
buildingID = 'building_id'
mangagerQuality = makeFeatureQuality(managerID)
buildingQuality = makeFeatureQuality(buildingID)

df["mangager_quality"] = ""  
df["building_quality"] = ""   
df["mangager_quality"] = df[managerID].map(mangagerQuality)
df["building_quality"] = df[buildingID].map(buildingQuality)

# In[7]:
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

# In[]
# Number of 'features' and length of description
df['num_of_features'] = df.features.map(len)
df['description_length'] = df.description.apply(lambda x: len(x.split(" ")))

# In[]:
# ### price per bedroom
# creating flag for bedrooms = 0 (studio)
df['studio'] = df.bedrooms.apply(lambda x: 1 if x==0 else 0)
# setting bedrooms/bathrooms = 0 to 1
df.bedrooms[df.bedrooms == 0] = 1
df['price_per_bedroom'] = df.price / df.bedrooms
# set bedrooms back to original value
df.bedrooms[df.studio == 1] = 0

# In[]
df['day_created'] = df.DateTime.map(lambda x: x.day)
df['month_created'] = df.DateTime.map(lambda x: x.month)
df['year_created'] = df.DateTime.map(lambda x: x.year)


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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# In[18]:
#==============================================================================
# Splitting Dataset
#==============================================================================

# determine features to use for modelling prior to data split

# baseline features
features_to_use = ['bathrooms','bedrooms','price', 'longitude', 'latitude']

# final features
# features_to_use = ['latitude','longitude','bathrooms','bedrooms',
                   'price', 'the_bronx', 'staten_island','manhattan',
                   'queens','brooklyn', 'num_of_photos', 'price_per_bedroom',
                   'studio','description_length','num_of_features',
                   'day_created','month_created']


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


# In[19]:
#==============================================================================
# Modeling and evaluation
#==============================================================================

# define model params
# model = RandomForestClassifier(n_estimators=1000, random_state=1, class_weight = 'balanced') # baseline
# model = MLPClassifier(solver = 'lbfgs', alpha = 1e-2, hidden_layer_sizes = (32,64), random_state=1,)
# model = GradientBoostingClassifier(n_estimators=1000, random_state=1)
model = LogisticRegression(class_weight = 'balanced')
# train model
model.fit(X_train_df, y_train)

# evaluation
y_hat_train = model.predict(X_train_df)
y_hat_val = model.predict(X_val_df)
y_hat_test = model.predict(X_test_df)

# confusion matrices - predicted class along the top, actual class down the side (low, medium, high)
print("training confusion matrix \n", confusion_matrix(y_train, y_hat_train, labels=[0,1,2]), "\n")
print("validation confusion matrix \n", confusion_matrix(y_val, y_hat_val, labels=[0,1,2]), "\n")
print("test confusion matrix \n", confusion_matrix(y_test, y_hat_test, labels=[0,1,2]), "\n")

y_hat_train_prob = model.predict_proba(X_train_df)
y_hat_val_prob = model.predict_proba(X_val_df)
y_hat_test_prob = model.predict_proba(X_test_df)

# log loss evaluations for train, val
print("log loss - training:", log_loss(y_train, y_hat_train_prob))
print("log loss - validation:", log_loss(y_val, y_hat_val_prob))
print("log loss - test:", log_loss(y_test, y_hat_test_prob))

print(classification_report(y_test, y_hat_test))

# In[]
#==============================================================================
# Plot confusion matrix
#==============================================================================
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


test_cm = confusion_matrix(y_val, y_hat_val, labels=[0,1,2])

plot_confusion_matrix(test_cm, classes = ['low','medium','high'], normalize=False)

# In[20]:
#==============================================================================
# feature importance - TODO
#==============================================================================
# feature importance measures

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=250, random_state = 0)
clf = clf.fit(X_train_df, y_train)

features = pd.DataFrame()
features['feature'] = X_train_df.columns
features['importance'] = clf.feature_importances_

features.sort(['importance'],ascending=False)

sns.barplot(y = 'feature', x = 'importance', data=features.sort_values(by='importance', ascending=False))

# In[]
# hyperparam optimisation for random forest
forest = RandomForestClassifier(max_features='sqrt', verbose=1, class_weight = 'balanced')

parameter_grid = {
                 'max_depth' : [7,8],
                 'n_estimators': [250,500,1000],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(n_splits=3)
cross_validation.get_n_splits(X_train_df, y_train)

grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation, n_jobs=4, scoring='neg_log_loss')
grid_search.fit(X_train_df, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# In[]
# hyperparam optimisation for neural net
clf_nn = MLPClassifier(solver='lbfgs', random_state=1)
parameter_grid = {
                  'alpha': [1e-6, 1e-5],
                  'activation': ['tanh', 'relu', 'logistic'],
                  'hidden_layer_sizes': [(10, 30, 5), (30, 30, 5)]
                 }

cross_validation = StratifiedKFold(n_splits=3)
cross_validation.get_n_splits(X_train_df, y_train)

gs_nn = GridSearchCV(clf_nn, param_grid=parameter_grid, scoring='neg_log_loss', n_jobs=-1, cv=cross_validation, verbose=2, refit=True)
gs_nn.fit(X_train_df, y_train)
print('- Best score: %.4f' % gs_nn.best_score_)
print('- Best params: %s' % gs_nn.best_params_)

# In[]
# hyperparam optimisation for logistic regression
clf_log = LogisticRegression(verbose=1, class_weight = 'balanced', random_state=1)

parameter_grid = {
                 'C' : [1, 0.1, 0.001]
                 }

cross_validation = StratifiedKFold(n_splits=3)
cross_validation.get_n_splits(X_train_df, y_train)

grid_search = GridSearchCV(clf_log, param_grid=parameter_grid, cv=cross_validation, n_jobs=-1, scoring='neg_log_loss')
grid_search.fit(X_train_df, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))