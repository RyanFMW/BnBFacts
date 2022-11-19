import os
import glob
import pathlib
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LassoCV

# define function pre_process to load and prepare data

def pre_process(city):
  
    path = pathlib.WindowsPath('C:/Users/user/Documents/working_dir')
  
    gz_files = glob.glob(os.path.join(path, "*.gz"))    
    listings2 = list(filter(lambda gz_files: 'listings' in gz_files and city in gz_files and '2022' in gz_files, gz_files))
    
    geojson_files = glob.glob(os.path.join(path, "*.geojson"))    
    neighbourhoods_geojson = list(filter(lambda geojson_files: city in geojson_files, geojson_files))
    neighbourhoods_geojson = ''.join(neighbourhoods_geojson)
    
    csv_files = glob.glob(os.path.join(path, "*.csv"))    
    walkscore_path = filter(lambda csv_files: city in csv_files, csv_files)
    walkscore_path = ''.join(walkscore_path)
            
    li = []
    
    for filename in listings2:
        df = pd.read_csv(filename, index_col=None, header=0, parse_dates=['last_review', 'calendar_last_scraped'])
        df['filedate'] = filename
        li.append(df)

    listings2  = pd.concat(li, axis=0, ignore_index=True)
    listings2['filedate'] = listings2['filedate'].str.replace(r'[^0-9]+', '')
    listings2['filedate'] = pd.to_datetime(listings2['filedate'], format='%Y%m%d')
    listings2['review_scores_rating'] = np.where(listings2['review_scores_rating']>5, listings2['review_scores_rating']/20, listings2['review_scores_rating'])
    listings2['host_response_rate'] = listings2['host_response_rate'].str.replace('%', '').astype(float)
    listings2['host_acceptance_rate'] = listings2['host_acceptance_rate'].str.replace('%', '').astype(float)
    listings2['days_since_last_review'] = (listings2['calendar_last_scraped'] - listings2['last_review']).dt.days
    listings2['price']  = listings2['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    listings2['price_per_accommodates'] = listings2['price']/listings2['accommodates']    
    listings2 = listings2[listings2['number_of_reviews']>3]
    
    neighbourhood_stats2 = listings2[['id','filedate','neighbourhood_cleansed','price_per_accommodates']].groupby(['filedate', 'neighbourhood_cleansed'], as_index=False).agg({'id' : ['count'], 'price_per_accommodates' :['mean']})
    neighbourhood_stats2 = neighbourhood_stats2.droplevel(1, axis=1)
    neighbourhood_stats2 = neighbourhood_stats2.rename(columns={"id": "listing_count","price_per_accommodates" : "price_per_accommodates_avg"})
    
    neighbourhoods = gpd.read_file(neighbourhoods_geojson)
    neighbourhoods['sq_km'] = neighbourhoods['geometry'].to_crs({'proj':'cea'}).area / 10**6
    neighbourhood_stats2 = neighbourhood_stats2.merge(neighbourhoods, left_on = 'neighbourhood_cleansed', right_on = 'neighbourhood')
    neighbourhood_stats2['listings_per_sq_km'] = neighbourhood_stats2['listing_count']/neighbourhood_stats2['sq_km']
    walkscore = pd.read_csv(walkscore_path, index_col=None, header=0)
    neighbourhood_stats2 = neighbourhood_stats2.merge(walkscore, left_on = 'neighbourhood', right_on = 'neighbourhood')

    amenities = listings2[['id', 'filedate', 'neighbourhood_cleansed','amenities']]
    amenities = amenities.dropna(axis=0)
    amenities['amenities'] = amenities['amenities'].str.replace('"', '')
    amenities['amenities'] = amenities['amenities'].str.replace('[', '')
    amenities['amenities'] = amenities['amenities'].str.replace(']', '')
    amenities['amenities_count'] = amenities['amenities'].str.count(',')
    amenities['amenities'] = amenities['amenities'].str.split(',') 
    amenities= amenities.explode('amenities')
    amenities = amenities[~amenities.amenities.isin(['translation missing: en.hosting_amenity_49',
                                                     'translation missing: en.hosting_amenity_50',
                                                     'Pets live on this property'
                                                     'First aid kit'])] 
    amenities['amenities'] = amenities['amenities'].str.strip()
    amenities['count'] = 1   
    amenities = amenities.pivot_table(index=['id', 'filedate', 'amenities_count', 'neighbourhood_cleansed'], columns = 'amenities', values = 'count')
    amenities.reset_index(inplace=True)
    amenities = amenities.fillna(0)

    amenities_exclude_df = pd.DataFrame(amenities.astype(bool).sum(axis=0), columns =['listing_count'])
    amenities_exclude_df.reset_index(inplace=True)
    amenities_exclude_df = amenities_exclude_df[amenities_exclude_df['listing_count']<100]
    amenities_exclude =  amenities_exclude_df['amenities'].tolist()
    amenities_include = list(set(amenities.columns) - set(amenities_exclude))
    amenities = amenities[amenities_include]
    
    data = listings2.merge(neighbourhood_stats2, left_on = ['filedate', 'neighbourhood_cleansed'], right_on = ['filedate', 'neighbourhood_cleansed']) 
    data['bedrooms_per_accommodates'] = data['bedrooms']/data['accommodates']
    
    return data, amenities, neighbourhoods_geojson, walkscore

# loop each city through pre_process function and lasso regression; save output in lists

li2 = []
li3 = []
li4 = []
dic1 = {}

cities = ['denver', 'new_york', 'san_diego', 'washington', 'portland', 'chicago', 'toronto', 'new_orleans']

for city in cities: 
    
    data, amenities, neighbourhoods_geojson, walkscore = pre_process(city)
              
    data2 = data[[
        'id',
        'filedate',
        'room_type',
        'accommodates',
        'bedrooms',
        'beds',
        'price',
        'minimum_nights_avg_ntm',
        'maximum_nights_avg_ntm',
        'review_scores_rating',
        'instant_bookable',
        'listings_per_sq_km',
        'walk_score',
        'host_is_superhost',
        'review_scores_accuracy',
        'review_scores_cleanliness',
        'review_scores_checkin',
        'review_scores_communication',
        'review_scores_location',
        'review_scores_value'
        ]]
                        
    
    encoder = OneHotEncoder() 
    
    encoder_df = pd.DataFrame(encoder.fit_transform(data2[['host_is_superhost','room_type','instant_bookable']]).toarray())
    encoder_df.columns = encoder.get_feature_names_out()
    
    data2 = data2.join(encoder_df)
    data2.drop(['host_is_superhost','room_type','instant_bookable'], axis=1, inplace=True)
    data2 = data2.merge(amenities, left_on = ['id','filedate'], right_on = ['id', 'filedate'])
    data2.replace([np.inf, -np.inf], np.nan, inplace=True)
    data2 = data2.dropna(axis=0) 
    data2 = data2[data2['price'] <1000]
    data2 = data2[data2['price'] >0]
    data2 = data2.set_index('id')  
    data2['city'] = city
    
    li4.append(data2)
    
    # Separating X and y
    X = data2.drop(['filedate', 'price', 'neighbourhood_cleansed', 'amenities_count', 'city'], axis=1)
    y = data2.price
    
    
    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) 
    
    # Scaling
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=list(X_train.columns))
    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=list(X_test.columns))   
    
    # LASSO REGRESSOR
    
    lasso = LassoCV()
    
    # Fit the regressor to the data
    lasso_fit = lasso.fit(X_train, y_train)
    
    # Compute and print the coefficients
    lasso_coef = lasso_fit.coef_
    
    df_columns = X.columns
    
    lasso_df = pd.DataFrame({'coef':lasso_coef, 'variable':df_columns})
    lasso_df['city'] = city
    
    li3.append(lasso.score(X_test, y_test))
    
    li2.append(lasso_df)
    
    dic1[city] = [neighbourhoods_geojson, walkscore] 
    

    
# r2 for case study cities

df3  = pd.DataFrame({'city':cities,'r2':li3})
df3.loc['average'] = df3.mean().replace(np.nan, 'average')

# summarize case study cities and normalize coefficients 

df2  = pd.concat(li2, axis=0, ignore_index=False).rename(columns={0: "coef"})
df2.reset_index(inplace=True)

# all case study data 

df4 = pd.concat(li4, axis=0, ignore_index=False)
df4['price_per_accommodates']=df4['price']/df4['accommodates']

# average price per city to normalize coefficients 

df5 = df4[['city','price']].groupby('city').agg({'city' : ['count'], 'price' :['mean']})
df5 = df5.droplevel(1, axis=1)
df5 = df5.rename(columns={"city": "listing_count","price" : "avg_price"})

df2 = df2.join(df5, on='city', lsuffix='_l', rsuffix='_r')
df2['normalized_coef'] = df2['coef']/df2['avg_price']

df2 = df2[['city','variable','normalized_coef']].pivot_table(index=['variable'], columns = 'city', values = 'normalized_coef')
df2['average'] = df2.mean(axis=1)
df2['count'] = df2.count(axis=1)
df2 = df2.sort_values(by="average",ascending=False)
df2 = df2[df2['count']>4]

# plot coefficients 

sns.set(rc={"figure.dpi":150, 'savefig.dpi':150, 'figure.figsize':(10,30)})
fig, ax = plt.subplots(1, 1)
sns.barplot (x="average", y=df2.index, data=df2).set_title('Price Influence Index of Airbnb Listing Features',fontsize=20)
ax.set(xlabel='Price Influence Index',ylabel='')
plt.show()

# box plots accommodates vs price by city

sns.catplot(
    data=df4[df4['accommodates'].isin([2,4,6,8])], x='accommodates', y='price',
    col='city', kind='box', col_wrap=2
)

# scatter plots overall rating vs price per accommodates by city

sns.relplot(
    data=df4[(df4['review_scores_rating']>=4) & (df4['price_per_accommodates']<=150)][['city','price_per_accommodates','review_scores_rating']].groupby(['city','review_scores_rating']).mean('price_per_accommodates'), x='review_scores_rating', y='price_per_accommodates',
    col="city", kind="scatter", col_wrap=2
)

# maps showing walkscore versus price per accommodates (some outliers removed)

colors = ['Purples', 'Reds', 'Greens', 'Oranges', 'Blues', 'Purples', 'Reds', 'Greens']

for city, color in zip(cities, colors): 

    citymap = gpd.read_file(dic1[city][0])
    citymap = citymap.merge(dic1[city][1])
    citymap = citymap.merge(df4[(df4['city']==city) & (df4['neighbourhood_cleansed'].isin(['Indian Creek', 'Argay','Read Blvd East']) ==False)][['neighbourhood_cleansed','price_per_accommodates']].groupby('neighbourhood_cleansed').mean('price_per_accommodates'), left_on = 'neighbourhood', right_on='neighbourhood_cleansed')
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    citymap.plot(column='walk_score', cmap=color, ax=ax1, legend=True,legend_kwds={'orientation': 'horizontal', 'pad': 0.01, 'label': "Walk Score", "ticks" :[]})
    ax1.axis('off')
    ax1.set_title('Walk Score by neighbourhood for ' + city, fontsize=14)
    citymap.plot(column='price_per_accommodates', cmap=color, ax=ax2, legend=True,legend_kwds={'orientation': 'horizontal', 'pad': 0.01, 'label': "Price per accommodates", "ticks" :[]})
    ax2.axis('off')
    ax2.set_title('Price per Accommodates by neighbourhood for ' + city, fontsize=14)
    plt.show()

# plot rating features only

rating_features = ['review_scores_rating','review_scores_location','review_scores_cleanliness','review_scores_accuracy','review_scores_communication','review_scores_checkin','review_scores_value']

sns.set(rc={"figure.dpi":150, 'savefig.dpi':150, 'figure.figsize':(10,5)})
fig, ax = plt.subplots(1, 1)
sns.barplot (x=df2[df2.index.isin(rating_features)].index, y='average', data=df2[df2.index.isin(rating_features)]).set_title('Price Influence Index of Airbnb Ratings',fontsize=20)
ax.set(xlabel='Price Influence Index', ylabel='')
plt.xticks(rotation=90)
ax.set_xticks(range(len(df2[df2.index.isin(rating_features)])), labels=(['Overall','Location','Cleanliness','Accuracy','Communication','Check-in','Value']))
plt.show()

# reproduce plots for instagram;

sub_df2 = df2[(df2['average']>0.02) | (df2['average']<-0.02)]

fig, ax = plt.subplots(figsize=(1080/96, 1080/96))
sns.barplot (x="average", y=sub_df2.index, data=sub_df2,ax=ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_yticklabels(sub_df2.index,size = 18)
ax.text(x=0.5, y=1.05, s='Price Influence Index of Airbnb Listing Features', fontsize=32, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
ax.text(x=0.5, y=1.02, s='Features with positive values increase price, negative values decrease price', fontsize=22, ha='center', va='bottom', transform=ax.transAxes)
plt.show()


# individual boxplots 

city_caps = ['Denver', 'New York', 'San Diego', 'Washington D.C.', 'Portland', 'Chicago', 'Toronto', 'New Orleans']

for city, city_caps in zip(cities, city_caps): 
    
    fig, ax = plt.subplots(figsize=(11.25,11.25))
    sns.boxplot(data=df4[(df4['accommodates'].isin([2,4,6,8])) & (df4['city']==city)], x='accommodates', y='price', ax=ax)
    ax.text(x=0.5, y=1.05,s='Price Distribution by No. Guests for '+city_caps,fontsize=24, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=1.015, s='Price increases with number of guests accommodated', fontsize=18, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
    ax.set_yticklabels(ax.get_yticks(), size = 14)
    ax.set_xticklabels([2,4,6,8], size = 14)
    ax.set_xlabel('Number of guests accommodated', size = 16)
    ax.set_ylabel('Price', size = 16)
    ax.yaxis.set_major_formatter('${x:1.0f}')
    plt.show()
    fig.savefig('insta_box_'+city)

# individual scatterplots  

colors2 = ['Purple', 'Red', 'Green', 'Orange', 'Blue', 'Purple', 'Red', 'Green']

for city, color, city_caps in zip(cities, colors2, city_caps): 
    
    fig, ax = plt.subplots(figsize=(1080/96, 1080/96), dpi=80)
    sns.scatterplot(data=df4[(df4['review_scores_rating']>=4) & (df4['price_per_accommodates']<=150) & (df4['city']==city)][['city','price_per_accommodates','review_scores_rating']].groupby(['city','review_scores_rating']).mean('price_per_accommodates'), x='review_scores_rating', y='price_per_accommodates', color =color, s = 200, alpha=0.75, ax=ax)
    ax.text(x=0.5, y=1.05,s='Price per Guest vs Overall Rating for '+city_caps,fontsize=24, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=1.015, s='Price per guest increases with the overall rating', fontsize=18, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
    ax.set_yticklabels(ax.get_yticks(), size = 14)
    ax.set_xticklabels(ax.get_xticks(), size = 14)
    ax.set_xlabel('Overall star rating', size = 16)
    ax.set_ylabel('Price per guest', size = 16)
    ax.yaxis.set_major_formatter('${x:1.0f}')
    ax.xaxis.set_major_formatter('{x:1.1f}')
    ax.set_box_aspect(0.935)
    plt.show()

# individual maps  
cities = ['denver', 'new_york', 'san_diego', 'washington', 'portland', 'chicago', 'toronto', 'new_orleans']
colors = ['Purples', 'Reds', 'Greens', 'Oranges', 'Blues', 'Purples', 'Reds', 'Greens']
city_caps = ['Denver', 'New York', 'San Diego', 'Washington D.C.', 'Portland', 'Chicago', 'Toronto', 'New Orleans']

for city, color, city_caps in zip(cities, colors, city_caps): 

    citymap = gpd.read_file(dic1[city][0])
    citymap = citymap.merge(dic1[city][1])
    citymap = citymap.merge(df4[(df4['city']==city) & (df4['neighbourhood_cleansed'].isin(['Indian Creek', 'Argay','Read Blvd East','Lake Catherine','Village De Lest', 'Viavant - Venetian Isles', 'DIA']) ==False)][['neighbourhood_cleansed','price_per_accommodates']].groupby('neighbourhood_cleansed').mean('price_per_accommodates'), left_on = 'neighbourhood', right_on='neighbourhood_cleansed')
    fig, ax1 = plt.subplots(1,1)
    citymap.plot(column='walk_score', edgecolors='black', cmap=color,  ax=ax1, legend=True,legend_kwds={'orientation': 'horizontal', 'pad': 0.01, "shrink":0.7, 'label': "Walk Score", "ticks" :[]})
    ax1.axis('off')
    ax1.set_title('Walk Score by Neighbourhood for ' + city_caps+'\n (Darker Colors = More Walkable)', fontsize=24, weight='bold')
    fig1 = ax1.figure
    cb_ax = fig1.axes[1] 
    cb_ax.patch.set_edgecolor('black')  
    cb_ax.patch.set_linewidth('5')
    ax1.set_box_aspect(0.84)
    plt.show()
    
    fig, ax2 = plt.subplots(1,1)
    citymap.plot(column='price_per_accommodates',  edgecolors='black', cmap=color, ax=ax2, legend=True,legend_kwds={'orientation': 'horizontal', 'pad': 0.01, "shrink":0.7, 'label': "Price per Guest", "ticks" :[]})
    ax2.axis('off')
    ax2.set_title('Price per Guest by Neighbourhood for ' + city_caps+'\n (Darker Colors = Higher Prices)', fontsize=24, weight='bold')
    fig2 = ax2.figure
    cb_ax2 = fig2.axes[1] 
    cb_ax2.patch.set_edgecolor('black')  
    cb_ax2.patch.set_linewidth('5')
    ax2.set_box_aspect(0.84)
    plt.show()

# individual category plot  

sns.set(rc = {'figure.figsize':(1080/96, 1080/96)}) 
fig, ax = plt.subplots(1, 1)
sns.barplot (x=df2[df2.index.isin(rating_features)].index, y='average', data=df2[df2.index.isin(rating_features)]).set_title('Price Influence Index of Airbnb Ratings',fontsize=28)
ax.set(xlabel='', ylabel='')
plt.xticks(rotation=45)
ax.set_xticks(range(len(df2[df2.index.isin(rating_features)])), labels=(['Overall','Location','Cleanliness','Accuracy','Communication','Check-in','Value']), fontsize=18)
ax.text(x=0.5, y=1.05, s='The size of each point corresponds to sepal width', fontsize=8, ha='center', va='bottom', transform=ax.transAxes)
plt.show()












