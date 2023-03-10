import os
import glob
import pathlib
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import xgboost as xgb
from xgboost import plot_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import shap

path = pathlib.WindowsPath('C:/Users/user/Documents/AirbnbData/older_newyork')
gz_files = glob.glob(os.path.join(path, "*.gz"))

geojson_files = glob.glob(os.path.join(path, "*.geojson"))    
neighbourhoods_geojson = list(filter(lambda geojson_files: 'york' in geojson_files, geojson_files))
neighbourhoods_geojson = ''.join(neighbourhoods_geojson)

csv_files = glob.glob(os.path.join(path, "*.csv"))    
walkscore_path = filter(lambda csv_files: 'york' in csv_files, csv_files)
walkscore_path = ''.join(walkscore_path)

listings2 = list(filter(lambda gz_files: 'listings' in gz_files and 'york' in gz_files, gz_files))

li = []

for filename in listings2:
    df = pd.read_csv(filename, header=0, parse_dates=['last_review', 'calendar_last_scraped'])
    df['filedate'] = filename
    li.append(df)

listings  = pd.concat(li, axis=0)

# clean listings data

listings['filedate'] = listings['filedate'].str.replace(r'[^0-9]+', '')
listings['filedate'] = pd.to_datetime(listings['filedate'], format='%Y%m%d')
listings['review_scores_rating'] = np.where(listings['review_scores_rating']>5, listings['review_scores_rating']/20, listings['review_scores_rating'])
listings['price']  = listings['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
listings['cleaning_fee']  = listings['cleaning_fee'].replace({'\$': '', ',': ''}, regex=True).astype(float)
listings['security_deposit']  = listings['security_deposit'].replace({'\$': '', ',': ''}, regex=True).astype(float)
listings['extra_people']  = listings['extra_people'].replace({'\$': '', ',': ''}, regex=True).astype(float)
listings['price_per_accommodates'] = listings['price']/listings['accommodates']
listings['month'] = listings['filedate'].dt.month
listings['bedrooms'] = listings['bedrooms'].fillna(0)
listings['cleaning_fee'] = listings['cleaning_fee'].fillna(0)
listings['security_deposit'] = listings['security_deposit'].fillna(0)
listings['days_booked_n30'] = 30-listings['availability_30'] #30 less availability_30 is an approximation of next 30 day bookings
listings['days_booked_n30_cat'] = 0
listings['days_booked_n30_cat'] = np.where((listings['days_booked_n30']>=10) & (listings['days_booked_n30']<20), 1, listings['days_booked_n30_cat'])
listings['days_booked_n30_cat'] = np.where(listings['days_booked_n30']>=20, 2, listings['days_booked_n30_cat'])
listings['cancellation_policy'] = np.where(listings['cancellation_policy']=="super_strict_60", "strict", listings['cancellation_policy'])
listings['cancellation_policy'] = np.where(listings['cancellation_policy']=="super_strict_30", "strict", listings['cancellation_policy'])
listings['cancellation_policy'] = np.where(listings['cancellation_policy']=="strict_14_with_grace_period", "strict", listings['cancellation_policy'])
listings = listings[~listings.property_type.isin(['Hotel', 'Boutique hotel', 'Hostel'])]
listings['property_type'] = np.where(~listings['property_type'].isin(['Apartment', 'House']), "Other", listings['property_type'])

listings = listings.reset_index(drop=True)

# create neighborhood data set w/ walkscore

neighbourhood_stats = listings[['id','filedate','neighbourhood_cleansed','price_per_accommodates']].groupby(['filedate', 'neighbourhood_cleansed'], as_index=False).agg({'id' : ['count'], 'price_per_accommodates' :['mean']})
neighbourhood_stats = neighbourhood_stats.droplevel(1, axis=1)
neighbourhood_stats = neighbourhood_stats.rename(columns={"id": "listing_count","price_per_accommodates" : "price_per_accommodates_avg"})

neighbourhoods_geojson = os.path.join(path,neighbourhoods_geojson)
neighbourhoods = gpd.read_file(neighbourhoods_geojson)
neighbourhoods['sq_km'] = neighbourhoods['geometry'].to_crs({'proj':'cea'}).area / 10**6
neighbourhood_stats = neighbourhood_stats.merge(neighbourhoods, left_on = 'neighbourhood_cleansed', right_on = 'neighbourhood')
neighbourhood_stats['listings_per_sq_km'] = neighbourhood_stats['listing_count']/neighbourhood_stats['sq_km']
walkscore_path = os.path.join(path,walkscore_path)
walkscore = pd.read_csv(walkscore_path, index_col=None, header=0)
neighbourhood_stats = neighbourhood_stats.merge(walkscore, left_on = 'neighbourhood', right_on = 'neighbourhood')
neighbourhood_stats['population_per_sq_km'] = neighbourhood_stats['population']/neighbourhood_stats['sq_km']

# create amenities data set from listings. Exclude infrequent amenities 

amenities = listings[['id', 'filedate', 'neighbourhood_cleansed','amenities']]
amenities = amenities.dropna(axis=0)
amenities['amenities'] = amenities['amenities'].str.replace('"', '')
amenities['amenities'] = amenities['amenities'].str.replace('}', '')
amenities['amenities'] = amenities['amenities'].str.replace('{', '')
amenities['amenities_count'] = amenities['amenities'].str.count(',')
amenities['amenities'] = amenities['amenities'].str.split(',') 
amenities= amenities.explode('amenities')
amenities['amenities'] = amenities['amenities'].str.strip()
amenities['count'] = 1

amenities = amenities.pivot_table(index=['id', 'filedate', 'amenities_count', 'neighbourhood_cleansed'], columns = 'amenities', values = 'count')
amenities.reset_index(inplace=True)
amenities = amenities.fillna(0)

amenities_exclude_df = pd.DataFrame(amenities.astype(bool).sum(axis=0), columns =['listing_count'])
amenities_exclude_df.reset_index(inplace=True)
amenities_exclude_df = amenities_exclude_df[amenities_exclude_df['listing_count']<500]
amenities_exclude =  amenities_exclude_df['amenities'].tolist()
amenities_include = list(set(amenities.columns) - set(amenities_exclude))
amenities = amenities[amenities_include]

# create and constrain dataset for modeling 

data = listings.merge(neighbourhood_stats, left_on = ['filedate', 'neighbourhood_cleansed'], right_on = ['filedate', 'neighbourhood_cleansed'])

data = data[data.calendar_updated.isin(['a week ago',
'today',
'yesterday',
'1 week ago',
'2 weeks ago',
'3 weeks ago',
'4 weeks ago',
'5 weeks ago',
'6 weeks ago',
'7 weeks ago',
'2 days ago',
'3 days ago',
'4 days ago',
'5 days ago',
'6 days ago',
'2 months ago'])]

data = data[data['minimum_nights'] <=3]
data = data[data['accommodates'] <=4]
data = data[(data['days_booked_n30']>0) & (data['days_booked_n30'] <30)]  
data = data[data['number_of_reviews'] >3]

data = data.reset_index(drop=True)

data = data[['id', 
             'filedate',
             'host_is_superhost',
             'host_total_listings_count', 
             'room_type',
             'security_deposit',
             'cleaning_fee',
             'extra_people',
             'instant_bookable',
             'cancellation_policy',
             'month',
             'listings_per_sq_km', 
             'walk_score', 
             'property_type',
             'review_scores_value',
             'review_scores_rating', 
             'review_scores_accuracy',
             'review_scores_cleanliness',
             'review_scores_location',
             'review_scores_checkin',
             'review_scores_communication',
             'price_per_accommodates',
             'days_booked_n30',
             'days_booked_n30_cat'
             ]]
 
temp=data.head(100) 

# One hot encode categorical features

encoder = OneHotEncoder() 

encoder_df = pd.DataFrame(encoder.fit_transform(data[['month','host_is_superhost','room_type','instant_bookable','cancellation_policy', 'property_type']]).toarray())
encoder_df.columns = encoder.get_feature_names_out()

data = data.join(encoder_df)
data.drop(['month','host_is_superhost','room_type','instant_bookable','cancellation_policy', 'property_type'], axis=1, inplace=True)

data = data.merge(amenities, left_on = ['id','filedate'], right_on = ['id', 'filedate'])

data.replace([np.inf, -np.inf], np.nan, inplace=True)

data = data.dropna(axis=0)

#summarize data for explanatory charts

neighbourhood_summary = data[['id', 'days_booked_n30_cat', 'neighbourhood_cleansed', 'walk_score', 'listings_per_sq_km']]
neighbourhood_summary = neighbourhood_summary.reset_index(drop=True)
neighbourhood_summary['high_occupancy'] = np.where(neighbourhood_summary['days_booked_n30_cat'] == 2, 1,0)
neighbourhood_summary = neighbourhood_summary.groupby(['neighbourhood_cleansed','walk_score'], as_index=False).agg({'id' : ['count'], 'high_occupancy' :['sum'], 'listings_per_sq_km' :['mean']})
neighbourhood_summary = neighbourhood_summary.droplevel(1, axis=1)
neighbourhood_summary['high_occupancy_per_avg'] = neighbourhood_summary['high_occupancy']/neighbourhood_summary['id']

data2 = data[['id', 'filedate', 'price_per_accommodates', 'host_total_listings_count','extra_people','days_booked_n30_cat']]
data2 = data2.reset_index(drop=True)
data2['high_occupancy'] = np.where(data2['days_booked_n30_cat'] == 2, 1,0)

# finalize dataset for modeling 

data = data.set_index('id')
data = data.drop(['days_booked_n30', 'filedate', 'neighbourhood_cleansed'], axis=1)

# scaler for numeric features 

scaler = Pipeline(steps=[('standard', StandardScaler())])

preprocessor = ColumnTransformer(
remainder='passthrough', #passthough features not listed
transformers=[('std', scaler , ['host_total_listings_count', 
  'security_deposit',
  'cleaning_fee',
  'extra_people',
  'listings_per_sq_km',
  'walk_score',
  'review_scores_value',
  'review_scores_rating', 
  'review_scores_accuracy',
  'review_scores_cleanliness',
  'review_scores_location',
  'review_scores_checkin',
  'review_scores_communication',
  'price_per_accommodates'])])

# Separating X and y
X = data.drop(['days_booked_n30_cat'], axis=1)
y = data.days_booked_n30_cat

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) 

# Scaling
scaled_X_train = preprocessor.fit_transform(X_train)     
X_train = pd.DataFrame(scaled_X_train, index=X_train.index, columns=X_train.columns)    

scaled_X_test = preprocessor.fit_transform(X_test)     
X_test = pd.DataFrame(scaled_X_test, index=X_test.index, columns=X_test.columns)    

# Gradient Boost Classifier 

xgbc = xgb.XGBClassifier(min_child_weight=200,label_encoder=False)

df_columns = X.columns

xgbc.fit(X_train, y_train)

y_pred = xgbc.predict(X_train) # Predictions

print(xgbc.score(X_test, y_test)) 

# Confusion matrix

class_names = ['low','mid','high']
disp = plot_confusion_matrix(xgbc, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, xticks_rotation='vertical')

# Feature importance 

xbgc_features = pd.DataFrame({'feature_importance':xgbc.feature_importances_, 'variable':df_columns})

plot_importance(xgbc, max_num_features=25)

# Shapley Value plots

explainer = shap.TreeExplainer(xgbc)
shap_values = shap.TreeExplainer(xgbc).shap_values(X_train)

shap.summary_plot(shap_values, X_train,class_names=['Low', 'Mid','High'], show=False)
plt.title('Shapley Value Summary Plot', fontsize = 20)
plt.show()

shap.summary_plot(shap_values[0], X_train,show=False)
plt.title("Low Occupancy Rate",fontsize = 20)
plt.show()

shap.summary_plot(shap_values[1], X_train,show=False)
plt.title("Mid",fontsize = 30)
plt.show()

shap.summary_plot(shap_values[2], X_train,show=False)
plt.title("High Occupancy Rate",fontsize = 20)
plt.show()


# Individual Shapley Value waterfall plot 

shap.waterfall_plot(shap.Explanation(values=shap_values[2][0], 
                                              base_values=explainer.expected_value[2], data=X_train.iloc[0],  
                                              feature_names=X_train.columns.tolist()))


# map high occupancy vs walkscore 

citymap = gpd.read_file(neighbourhoods_geojson)
citymap = citymap.merge(neighbourhood_summary[neighbourhood_summary['id']>100], how='left', left_on = 'neighbourhood', right_on='neighbourhood_cleansed')
  
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
citymap.plot(column='walk_score', cmap='Oranges', ax=ax1, legend=True,legend_kwds={'orientation': 'horizontal', 'pad': 0.01, 'label': "Walk Score", "ticks" :[]})
ax1.axis('off')
ax1.set_title('Walk Score', fontsize=14)
citymap.plot(column='high_occupancy_per_avg', cmap='Oranges', ax=ax2, legend=True,legend_kwds={'orientation': 'horizontal', 'pad': 0.01, 'label': "Avg % Listings with High Occupancy", "ticks" :[]})
ax2.axis('off')
ax2.set_title('Avg % High Occupancy', fontsize=14)
plt.show()

# map high occupancy vs listings per sq km

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
citymap.plot(column='listings_per_sq_km', cmap='Blues', ax=ax1, vmax = 200, legend=True,legend_kwds={'orientation': 'horizontal', 'pad': 0.01, 'label': "Listings per Square Kilometer", "ticks" :[]})
ax1.axis('off')
ax1.set_title('Avg Listings per Sq Km', fontsize=14)
citymap.plot(column='high_occupancy_per_avg', cmap='Blues', ax=ax2, legend=True,legend_kwds={'orientation': 'horizontal', 'pad': 0.01, 'label': "Avg % Listings with High Occupancy", "ticks" :[]})
ax2.axis('off')
ax2.set_title('Avg % High Occupancy', fontsize=14)
plt.show()

# seasonality (high occupancy by month)

data2_by_month = data2.groupby(['filedate'], as_index=False).agg({'id' : ['count'], 'high_occupancy' :['sum']})
data2_by_month = data2_by_month.droplevel(1, axis=1)
data2_by_month['high_occupancy_per_avg'] = 100*data2_by_month['high_occupancy']/data2_by_month['id']

fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.barplot(x = "filedate", y = "high_occupancy_per_avg", data = data2_by_month, 
                  estimator = sum, ci=None, ax=ax, color='tab:purple').set_title('% Listings with High Occupancy by Month',fontsize=20)
ax.set(xlabel='',ylabel='% high occupancy')
x_dates = data2_by_month['filedate'].dt.strftime('%Y-%m').sort_values().unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()

# price per accommodates vs high occupancy

sns.histplot(data=data2[data2["price_per_accommodates"]<200], x="price_per_accommodates") # check distribution of price

data2['price_bin'] = pd.cut(data2['price_per_accommodates'],[0, 25, 50, 75, 100, 125, 150, 175, 200])
data2_by_price = data2.groupby("price_bin", as_index=False).agg({'id' : ['count'], 'high_occupancy' :['sum']})
data2_by_price = data2_by_price.droplevel(1, axis=1)
data2_by_price['high_occupancy_per_avg'] = 100*data2_by_price['high_occupancy']/data2_by_price['id']

fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.barplot(x = "price_bin", y = "high_occupancy_per_avg", data = data2_by_price, 
                  estimator = sum, ci=None, ax=ax, color='tab:green').set_title('% Listings with High Occupancy by Price',fontsize=20)
ax.set(xlabel='Price per Guest Bin',ylabel='% High Occupancy')
ax.set(ylim=(30, 60))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()

# extra guest fee vs high occupancy

sns.histplot(data=data2, x="extra_people") # check distribution of extra guest fee

data2_by_extra_guest = data2[data2['extra_people'].isin([0.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0])].groupby("extra_people", as_index=False).agg({'id' : ['count'], 'high_occupancy' :['sum']})
data2_by_extra_guest = data2_by_extra_guest.droplevel(1, axis=1)
data2_by_extra_guest['high_occupancy_per_avg'] = 100*data2_by_extra_guest['high_occupancy']/data2_by_extra_guest['id']

fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.barplot(x = "extra_people", y = "high_occupancy_per_avg", data = data2_by_extra_guest, 
                  estimator = sum, ci=None, ax=ax).set_title('% Listings with High Occupancy by Extra Guest Fee',fontsize=20)
ax.set(xlabel='Extra Guest Fee',ylabel='% High Occupancy')
ax.set(ylim=(40, 60))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
for bar in ax.patches:
    if bar.get_height() > 56:
        bar.set_color('tab:red')    
    else:
        bar.set_color('tab:gray')
plt.show()

# number of listings under management vs high occupancy

data2['num_listings_bin'] = np.where(data2['host_total_listings_count']>=10, 10, data2['host_total_listings_count'])
data2_by_num_listings = data2[data2['host_total_listings_count']>0].groupby("num_listings_bin", as_index=False).agg({'id' : ['count'], 'high_occupancy' :['sum']})
data2_by_num_listings = data2_by_num_listings.droplevel(1, axis=1)
data2_by_num_listings['high_occupancy_per_avg'] = 100*data2_by_num_listings['high_occupancy']/data2_by_num_listings['id']

fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.barplot(x = "num_listings_bin", y = "high_occupancy_per_avg", data = data2_by_num_listings, 
                  estimator = sum, ci=None, ax=ax, color='tab:cyan').set_title('% Listings with High Occupancy by Listings under Management',fontsize=20)
ax.set(xlabel='Number of Listings under Management',ylabel='% High Occupancy')
ax.set(ylim=(30, 60))
x_labels = ['1','2','3','4','5','6','7','8','9','10+']
ax.set_xticklabels(labels=x_labels)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()
