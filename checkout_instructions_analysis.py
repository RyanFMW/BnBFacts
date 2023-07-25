import os
import glob
import pathlib
import pandas as pd

path = pathlib.WindowsPath('C:/Users/user/Documents/AirbnbData/Summer2023')

csv_files = glob.glob(os.path.join(path, "*.csv"))    

li = []

for filename in csv_files:
    df = pd.read_csv(filename, header=0, parse_dates=['checkin', 'checkout','date_sourced'])
    df['filedate'] = filename
    li.append(df)

listings  = pd.concat(li, axis=0)

# clean listings data

listings['total_rate'] = listings['total_rate'].str.extract('(\d+)', expand=False).astype(float)
listings['cleaning_fee'] = listings['cleaning_fee'].str.extract('(\d+)', expand=False).astype(float)
listings['airbnb_service_fee'] = listings['airbnb_service_fee'].str.extract('(\d+)', expand=False).astype(float)
listings['taxes'] = listings['taxes'].str.extract('(\d+)', expand=False).astype(float)
listings['days_quoted'] = (df['checkout'] - df['checkin']).dt.days
listings['nightly_rate_quoted'] = listings['total_rate']/listings['days_quoted']

# subset features of interest

listings_subset = listings[['id','nightly_rate_quoted', 'cleaning_fee', 'rating_value','rating_cleanliness', 'rating_overall', 'before_you_leave']]

# listings with chores vs listings without overall 

listings_subset['id'].count() 
listings_subset['before_you_leave'].count() 


# average stats of listings with instructions vs no instructions

listings_subset[listings_subset['before_you_leave'].isna()].mean()
listings_subset[listings_subset['before_you_leave'].notna()].mean()

# count of cleaning fee listings with instructions vs no instructions

listings_subset['has_cleaning_fee'] = ~listings_subset['cleaning_fee'].isna()
listings_subset['has_instructions'] = ~listings_subset['before_you_leave'].isna()

listings_subset[['id','has_cleaning_fee','has_instructions']].groupby(['has_instructions','has_cleaning_fee']).count()


# Breaking out individual chores

listings_subset = listings_subset.dropna(subset=['before_you_leave'])

listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('\'', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace(']', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('[', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.split(',') 
listings_subset= listings_subset.explode('before_you_leave')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.strip()
listings_subset.loc[listings_subset['before_you_leave'].str.contains('Additional requests'), 'before_you_leave'] = 'Additional requests'

summarized_chores_stats = listings_subset[listings_subset['before_you_leave'].isin(['Turn things off','Gather used towels', 'Lock up', 'Additional requests', 'Throw trash away', 'Return keys'])].groupby('before_you_leave').mean()

summarized_chores_count = listings_subset[listings_subset['before_you_leave'].isin(['Turn things off','Gather used towels', 'Lock up', 'Additional requests', 'Throw trash away', 'Return keys'])][['id','before_you_leave']].groupby('before_you_leave').count()
summarized_chores_count['percent']=summarized_chores_count['id']/6058


# Isolate only additional requests w/ dishes or dishwasher

listings_subset = listings[['id','nightly_rate_quoted', 'cleaning_fee', 'rating_value','rating_cleanliness', 'rating_overall', 'before_you_leave']]
listings_subset['has_cleaning_fee'] = ~listings_subset['cleaning_fee'].isna()
listings_subset['has_instructions'] = ~listings_subset['before_you_leave'].isna()
listings_subset = listings_subset.dropna(subset=['before_you_leave'])

listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('Turn things off', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('Gather used towels', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('Lock up',  '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('Additional requests', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('Throw trash away', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('Return keys', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('\'', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace(']', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('[', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace(',', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.replace('\n', '')
listings_subset['before_you_leave'] = listings_subset['before_you_leave'].str.strip()


listings_subset['dishes'] = listings_subset['before_you_leave'].str.contains('dishes|dishwasher')

# count and stats for listings with dishes

print(listings_subset[['id','dishes']].groupby('dishes').count())

dishes_stats = listings_subset[listings_subset['dishes'] == True].mean()


listings_subset = listings[['id','nightly_rate_quoted', 'cleaning_fee', 'rating_value','rating_cleanliness', 'rating_overall', 'before_you_leave']]

