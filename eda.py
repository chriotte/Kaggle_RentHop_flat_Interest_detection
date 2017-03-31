# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import preProcess as pre # Created a module for preprocessing
from sklearn.model_selection import train_test_split

from bokeh.io import output_notebook
from bokeh.layouts import gridplot,row,column
from bokeh.plotting import figure,show
from bokeh.plotting import figure, show, output_file
import matplotlib.cm as cm
output_notebook()
terrain = sns.color_palette(palette='terrain',n_colors=10)
plasma = sns.color_palette(palette='plasma',n_colors=10)

# In[]
# Read data, create dataframes and clean it
df = pd.read_json("train.json")
df, temp1, temp2 = pre.main(df,True)

# In[11]:
    
df = pd.read_json("train.json")

# In[]
#==============================================================================
#==============================================================================
#==============================================================================
# # #                              EDA - General
#==============================================================================
#==============================================================================
#==============================================================================
# Price plotting
#==============================================================================
plt.scatter(range(df.shape[0]), np.sort(df.price.values))
plt.title("Price with outliers present",fontsize = 18)
plt.xlabel('Price', fontsize = 15)
plt.ylabel('Number of apartment listings', fontsize = 15)
plt.show()

# In[]
sns.distplot(df.price.values, bins=25, kde=False)
plt.xlabel('Price', fontsize = 15)
plt.ylabel('Number of apartment listings', fontsize = 15)
plt.title("Price data after removing outliers", fontsize = 18)
plt.show()

# In[]
sns.distplot(df.price.values, bins=25, kde=False)
plt.xlabel('Price', fontsize = 15)
plt.ylabel('Number of apartment listings', fontsize = 15)
plt.title("Price data after removing outliers", fontsize = 18)
plt.show()

# In[11]:
#==============================================================================
# Location plotting
#==============================================================================
long_low  = -74.1
long_high = -73.7
lat_low   =  40.5
lat_high  =  41
x = [long_low,long_high]
y = [lat_low,lat_high]
plt.xlim(x)
plt.ylim(y)


long = df['longitude']
lat  = df['latitude']
plt.title("Distribution of apartments", fontsize = 18)
plt.ylabel('Longitude', fontsize = 15)
plt.xlabel('Latitude', fontsize = 15)
plt.scatter(long, lat)

# In[]
long_low  = -74.1
long_high = -73.7
lat_low   =  40.5
lat_high  =  41
x = [long_low,long_high]
y = [lat_low,lat_high]
plt.xlim(x)
plt.ylim(y)

plt.title("Distribution of apartments", fontsize = 18)
plt.ylabel('Longitude', fontsize = 15)
plt.xlabel('Latitude', fontsize = 15)

lowLat=df['latitude'][df['interest_level']=='low']
lowLong=df['longitude'][df['interest_level']=='low']
medLat=df['latitude'][df['interest_level']=='medium']
medLong=df['longitude'][df['interest_level']=='medium']
highLat=df['latitude'][df['interest_level']=='high']
highLong=df['longitude'][df['interest_level']=='high']

longLat = [[lowLat,lowLong],[medLat,medLong],[highLat,highLong]]

colors = iter(cm.rainbow(np.linspace(0, 1, len(longLat))))
for int_level in longLat:
    plt.scatter(int_level[1], int_level[0], color=next(colors))


# In[]

p = figure(title="interest level based on geography",y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p.xaxis.axis_label = 'latitude'
p.yaxis.axis_label = 'longitude'
lowLat=df['latitude'][df['interest_level']=='low']
lowLong=df['longitude'][df['interest_level']=='low']
medLat=df['latitude'][df['interest_level']=='medium']
medLong=df['longitude'][df['interest_level']=='medium']
highLat=df['latitude'][df['interest_level']=='high']
highLong=df['longitude'][df['interest_level']=='high']
p.circle(lowLong,lowLat,size=3,color=terrain.as_hex()[1],fill_alpha=0.1,line_alpha=0.1,legend='low')
p.circle(medLong,medLat,size=3,color=plasma.as_hex()[9],fill_alpha=0.1,line_alpha=0.1,legend='med')
p.circle(highLong,highLat,size=3,color=plasma.as_hex()[5],fill_alpha=0.1,line_alpha=0.1,legend='high')
show(p, notebook_handle=True)

p1 = figure(width=500, height=500, title=None,y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p1.circle(lowLong,lowLat,size=3,color=terrain.as_hex()[1],fill_alpha=0.1,line_alpha=0.1,legend='low')
p2 = figure(width=500, height=500, title=None,y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p2.circle(medLong,medLat,size=3,color=plasma.as_hex()[9],fill_alpha=0.1,line_alpha=0.1,legend='med')
p3 = figure(width=500, height=500, title=None,y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p3.circle(highLong,highLat,size=3,color=plasma.as_hex()[5],fill_alpha=0.1,line_alpha=0.1,legend='high')
show(column(p1,p2,p3), notebook_handle=True)


# In[]
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

# In[]
sns.countplot(x='hour_created', hue='interest_level', data=df)
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Hour of the day posted', fontsize=15)
plt.show()

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