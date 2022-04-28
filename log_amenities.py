import os
import glob
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report

# Import data. Data collected from Inside Airbnb from december 2021

path = pathlib.WindowsPath('C:/Users/user/Documents/AirbnbData')
csv_files = glob.glob(os.path.join(path, "*.csv"))
gz_files = glob.glob(os.path.join(path, "*.gz"))

listings = list(filter(lambda gz_files: 'listings' in gz_files and '2021-12' in gz_files, gz_files))

li = []

for filename in listings:
    df = pd.read_csv(filename, index_col=None, header=0, parse_dates=['last_review', 'calendar_last_scraped'])
    df['filedate'] = filename
    li.append(df)

# Format dataset including target variable of value >= 4.8

listings  = pd.concat(li, axis=0, ignore_index=True)
listings['filedate'] = listings['filedate'].str.replace(r'[^0-9]+', '')
listings['filedate'] = pd.to_datetime(listings['filedate'], format='%Y%m%d')
listings = listings[(listings['review_scores_rating']<=5) & (listings['review_scores_value']<=5)]
listings = listings[listings['number_of_reviews']>=5]
listings = listings.dropna(subset = ['review_scores_value'])
listings = listings.drop_duplicates(subset=['id'], keep='last')
listings['has_value'] = np.where(listings['review_scores_value']>=4.8, 1, 0)

# Unlist and one hot encode amenities

amenities = listings[['id', 'filedate', 'amenities']]
amenities = amenities.dropna(axis=0)
amenities['amenities'] = amenities['amenities'].str.replace('"', '')
amenities['amenities'] = amenities['amenities'].str.replace('[', '')
amenities['amenities'] = amenities['amenities'].str.replace(']', '')
amenities['amenities'] = amenities['amenities'].str.split(',') 
amenities= amenities.explode('amenities')
amenities['amenities'] = amenities['amenities'].str.strip()
amenities['count'] = 1

amenity_count = amenities.groupby('amenities').sum('count').sort_values(by=['count'], ascending=False)
amenity_count = amenity_count.drop(['id'], axis=1)
amenity_count = amenity_count[amenity_count['count'] > 1000]
amenity_count.reset_index(inplace=True)

amenities = amenities.merge(amenity_count, left_on = ['amenities'], right_on = ['amenities'], how = 'inner')
amenities = amenities.drop(['count_x', 'count_y'], axis=1)
amenities['count'] = 1

amenities = amenities.pivot_table(index=['id', 'filedate'], columns = 'amenities', values = 'count')
amenities.reset_index(inplace=True)
amenities = amenities.fillna(0)


# prepare data for modeling 

data = listings[['id', 'filedate', 'has_value', 'price']].merge(amenities, left_on = ['id','filedate'], right_on = ['id', 'filedate'])

data = data.set_index('id')

# Separating X and y
X = data.drop(['has_value', 'filedate', 'price'], axis=1)
y = data.has_value

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) 
   
# LOGISTIC REGRESSOR

log = LogisticRegressionCV(max_iter =1000)

# Fit the regressor to the data
log_fit = log.fit(X_train, y_train)

# Compute and print the coefficients
log_coef = log_fit.coef_
log_coef = log_coef.reshape(-1, 1)

df_columns = X.columns

log_df = pd.DataFrame(log_coef, index=df_columns)
log_df = log_df.merge(amenity_count, left_on = log_df.index, right_on = ['amenities'])

# Assess accuracy and performance 

log.score(X_test, y_test)

y_pred = log.predict(X_test)
confusion_matrix = print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot coefficients
log_df['feat_probability'] = (np.exp(log_df[0])-1)*100
log_df = log_df.sort_values(by = 'feat_probability', ascending = False)

sns.set(rc={"figure.dpi":150, 'savefig.dpi':150, 'figure.figsize':(10,30)})
fig, ax = plt.subplots(1, 1)
sns.barplot (x="feat_probability", y='amenities', data=log_df).set_title('Probability of Amenity Adding Value to Airbnb Listing',fontsize=20)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()





