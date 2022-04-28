import os
import glob
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import geopandas as gpd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Import data. Data collected from Inside Airbnb from April 2020

path = pathlib.WindowsPath('C:/Users/user/Documents/Full_Listings')
path2 = pathlib.WindowsPath('C:/Users/user/Documents/AirbnbData')
  
gz_files = glob.glob(os.path.join(path, "*.gz"))    
listings2 = list(filter(lambda gz_files: 'listings' in gz_files and 'united-state' in gz_files and '2020-04' in gz_files, gz_files))

li = []

for filename in listings2:
    df = pd.read_csv(filename, index_col=None, header=0, parse_dates=['last_review', 'calendar_last_scraped'])
    df['filedate'] = filename
    li.append(df)

# Format dataset. Category ratings were out of a scale 1-10. 10 used as target variable. 
    
listings  = pd.concat(li, axis=0, ignore_index=True)
listings['filecity'] = listings['filedate'].str[82:].str.split(' ').str[0]
listings['filedate'] = listings['filedate'].str.replace(r'[^0-9]+', '')
listings['filedate'] = pd.to_datetime(listings['filedate'], format='%Y%m%d')
listings = listings[listings['number_of_reviews']>=5]
listings = listings.drop_duplicates(subset=['id'], keep='last')
listings['host_response_rate'] = listings['host_response_rate'].str.replace('%', '').astype(float)
listings['host_acceptance_rate'] = listings['host_acceptance_rate'].str.replace('%', '').astype(float)
listings['price']  = listings['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
listings['cleaning_fee']  = listings['cleaning_fee'].replace({'\$': '', ',': ''}, regex=True).astype(float)
listings['security_deposit']  = listings['security_deposit'].replace({'\$': '', ',': ''}, regex=True).astype(float)
listings['extra_people']  = listings['extra_people'].replace({'\$': '', ',': ''}, regex=True).astype(float)
listings['has_value'] = np.where(listings['review_scores_value']==10, 1, 0)
listings['pets_allowed'] = np.where(listings['amenities'].str.contains('Pets allowed', regex=False),1,0)

# Import neighbourhood data also from Inside Airbnb

geojson_files = glob.glob(os.path.join(path2, "*.geojson"))    
neighbourhoods_geojson = list(filter(lambda geojson_files: 'united-states' in geojson_files and '12' in geojson_files, geojson_files))

li = []

for filename in neighbourhoods_geojson:
    df = gpd.read_file(os.path.join(path,filename))
    df['city'] = filename
    li.append(df)

# Calculate neighborhood area to get listings per sq km 

neighbourhoods = pd.concat(li, axis=0, ignore_index=True)
neighbourhoods['city'] = neighbourhoods['city'].str[79:].str.split(' ').str[0]
neighbourhoods['sq_km'] = neighbourhoods['geometry'].to_crs({'proj':'cea'}).area / 10**6
neighbourhoods = neighbourhoods.groupby(['neighbourhood', 'city']).sum(['sq_km'])

neighbourhood_stats = listings[['id','filedate','filecity','neighbourhood_cleansed']].groupby(['filedate','filecity', 'neighbourhood_cleansed'], as_index=False).agg({'id' : ['count']})
neighbourhood_stats = neighbourhood_stats.droplevel(1, axis=1)
neighbourhood_stats = neighbourhood_stats.rename(columns={"id": "listing_count"})

neighbourhood_stats = neighbourhood_stats.merge(neighbourhoods, left_on = ['filecity','neighbourhood_cleansed'], right_on = ['city','neighbourhood'])
neighbourhood_stats['listings_per_sq_km'] = neighbourhood_stats['listing_count']/neighbourhood_stats['sq_km']
neighbourhoods = neighbourhood_stats[['filecity','neighbourhood_cleansed', 'listing_count', 'sq_km','listings_per_sq_km']]

listings = listings.merge(neighbourhoods, left_on = ['filecity','neighbourhood_cleansed'], right_on = ['filecity','neighbourhood_cleansed'])

# Other adjustments to property type and cancellation polict

dict_property_type = {"House" : 'House', "Apartment" : 'Apartment/Condo',  "Condominium" : 'Apartment/Condo'}
listings['property_type'] = listings['property_type'].map(dict_property_type)
listings['property_type'] = listings['property_type'].fillna('Other')
 
dict_cancel = {"moderate" : 'moderate', "flexible" : 'flexible'}
listings['cancellation_policy'] = listings['cancellation_policy'].map(dict_cancel)
listings['cancellation_policy'] = listings['cancellation_policy'].fillna('strict')  
 
listings = listings[(listings['room_type'] != 'Hotel room') & (listings['room_type'] != 'Shared room')]
listings['bedrooms_per_accommodates'] = listings['bedrooms']/listings['accommodates']


# Prepare data

data = listings[['id', 
        'filedate',
        'host_is_superhost',
        'host_total_listings_count', 
        'room_type',
        'bedrooms_per_accommodates',
        'security_deposit',
        'cleaning_fee',
        'extra_people',
        'instant_bookable',
        'cancellation_policy',
        'listings_per_sq_km', 
        'property_type',
        'has_value',
        'pets_allowed']]

# Identify top 5% threshold for numerical features to exclude

data[['host_total_listings_count', 
  'bedrooms_per_accommodates',
  'security_deposit',
  'cleaning_fee',
  'extra_people',
  'listings_per_sq_km']].quantile(q=0.95)

data = data[data['host_total_listings_count'] < 100]
data = data[data['security_deposit'] < 1000]
data = data[data['cleaning_fee'] < 250]
data = data[data['extra_people'] < 50]
data = data[data['listings_per_sq_km'] < 250]

# One hot encode categorical features

encoder = OneHotEncoder() 

encoder_df = pd.DataFrame(encoder.fit_transform(data[['host_is_superhost','room_type','instant_bookable','cancellation_policy', 'property_type']]).toarray())
encoder_df.columns = encoder.get_feature_names_out()

data = data.join(encoder_df)
data.drop(['host_is_superhost','room_type','instant_bookable','cancellation_policy', 'property_type'], axis=1, inplace=True)


data.replace([np.inf, -np.inf], np.nan, inplace=True)
data['cleaning_fee'] = data['cleaning_fee'].fillna(0)
data['security_deposit'] = data['security_deposit'].fillna(0)

data = data.dropna(axis=0)
data = data.set_index('id')


# Separating X and y
X = data.drop(['has_value', 'filedate'], axis=1)
y = data.has_value

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) 

# Create Pipeline to scale only numeric variables 

scaler = Pipeline(steps=[('standard', StandardScaler())])

preprocessor = ColumnTransformer(
remainder='passthrough', #passthough features not listed
transformers=[('std', scaler , ['host_total_listings_count', 
  'bedrooms_per_accommodates',
  'security_deposit',
  'cleaning_fee',
  'extra_people',
  'listings_per_sq_km'])])

clf = Pipeline(
steps=[("preprocessor", preprocessor), ("classifier", LogisticRegressionCV(max_iter =1000))])
          
# Fit the regressor to the data
log_fit = clf.fit(X_train, y_train)

# Compute and print the coefficients
log_coef = clf.named_steps['classifier'].coef_
log_coef = log_coef.reshape(-1, 1)

df_columns = X.columns

log_df = pd.DataFrame(log_coef, index=df_columns)

# Assess accuracy and performance 

clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
confusion_matrix = print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))   
   
probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Standard deviations of numeric variables 

data[['host_total_listings_count', 
  'bedrooms_per_accommodates',
  'security_deposit',
  'cleaning_fee',
  'extra_people',
  'listings_per_sq_km']].std()
 

# plot coefficients
log_df['feat_probability'] = (np.exp(log_df[0])-1)*100
log_df = log_df.sort_values(by = 'feat_probability', ascending = False)

sns.set(rc = {'figure.figsize':(10,8)})

fig, ax = plt.subplots(1, 1)

sns.barplot(x="feat_probability", y=log_df.index , data=log_df).set_title('Probability of Feature Adding Value to Airbnb Listing',fontsize=20)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()

