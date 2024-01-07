#!/usr/bin/env python
# coding: utf-8

# # Package Importings

# In[81]:


import pandas as pd


# In[82]:


from geopy.distance import geodesic


# In[266]:


import numpy as np


# In[83]:


import calendar


# In[341]:


from sklearn.linear_model import LinearRegression


# In[343]:


from sklearn.model_selection import train_test_split


# In[348]:


from sklearn.metrics import mean_squared_error, r2_score
import math


# # Data Gathering

# In[84]:


df_train = pd.read_csv("train.csv")
df_train.head(10)


# In[85]:


df_test = pd.read_csv("test.csv")
df_test.head(10)


# # Data Preprocessing

# ## Exploration of data

# In[86]:


df_train.shape


# In[87]:


df_test.shape


# In[88]:


df_train.info()


# In[89]:


df_test.info()


# In[90]:


df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"])
df_train["dropoff_datetime"] = pd.to_datetime(df_train["dropoff_datetime"])


# In[91]:


df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"])


# ### NUll VALUES

# In[92]:


df_train.isnull().sum()


# In[93]:


df_test.isnull().sum()


# In[94]:


df_test.passenger_count.unique()


# In[95]:


df_test.store_and_fwd_flag.unique()


# ### Categorical variables

# In[96]:


from sklearn.preprocessing import LabelEncoder

# Assuming you have a DataFrame named df_test with a categorical column 'store_and_fwd_flag'

# Perform label encoding
label_encoder = LabelEncoder()
df_test['store_and_fwd_flag'] = label_encoder.fit_transform(df_test['store_and_fwd_flag'])


# In[97]:


df_test.head()


# In[98]:


from sklearn.preprocessing import LabelEncoder

# Assuming you have a DataFrame named df_test with a categorical column 'store_and_fwd_flag'

# Perform label encoding
label_encoder = LabelEncoder()
df_train['store_and_fwd_flag'] = label_encoder.fit_transform(df_train['store_and_fwd_flag'])


# In[99]:


df_train.head()


# In[100]:


df_test.describe()


# In[101]:


#40.690889 = lower fence(test)
#40.814897 = higher fence(test)
#abv higher fence below lower fence are outliers(test)(pickup latitude)


# In[102]:


df_train.describe()


# In[ ]:





# ### Feature adding

# In[104]:


def calculate_distance(row):
    pickup_coords = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coords = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(pickup_coords, dropoff_coords).miles

# Calculate distance between pickup and dropoff coordinates using the defined function
df_train['distance'] = df_train.apply(calculate_distance, axis=1)


# In[105]:


df_train.head()


# In[106]:


def calculate_distance(row):
    pickup_coords = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coords = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(pickup_coords, dropoff_coords).miles

# Calculate distance between pickup and dropoff coordinates using the defined function
df_test['distance'] = df_test.apply(calculate_distance, axis=1)


# In[107]:


df_test.head()


# In[108]:


df_train['pickup_date'] = df_train['pickup_datetime'].dt.date
df_train['pickup_day'] = df_train['pickup_datetime'].apply(lambda x: x.day)
df_train['pickup_hour'] = df_train['pickup_datetime'].apply(lambda x: x.hour)
df_train['pickup_day_of_the_week'] = df_train['pickup_datetime'].apply(lambda x: calendar.day_name[x.weekday()])
df_train['pickup_month'] = df_train['pickup_datetime'].apply(lambda x: x.month)
df_train['pickup_year'] = df_train['pickup_datetime'].apply(lambda x: x.year)
df_train.head()


# In[248]:


df_train['dropoff_hour'] = df_train['dropoff_datetime'].apply(lambda x: x.hour)
df_train['dropoff_month'] = df_train['dropoff_datetime'].apply(lambda x: x.month)


# In[109]:


df_test['pickup_date'] = df_test['pickup_datetime'].dt.date
df_test['pickup_day'] = df_test['pickup_datetime'].apply(lambda x: x.day)
df_test['pickup_hour'] = df_train['pickup_datetime'].apply(lambda x: x.hour)
df_test['pickup_day_of_the_week'] = df_test['pickup_datetime'].apply(lambda x: calendar.day_name[x.weekday()])
df_test['pickup_month'] = df_test['pickup_datetime'].apply(lambda x: x.month)
df_test['pickup_year'] = df_test['pickup_datetime'].apply(lambda x: x.year)
df_test.head()


# In[110]:


pd.set_option('display.float_format', '{:.2f}'.format)
df_train.describe()


# In[111]:


df_train = df_train[(df_train['pickup_longitude'] >= -74.3) & (df_train['pickup_longitude'] <= -73.7)].copy()
df_train


# In[112]:


df_test = df_test[(df_train['pickup_longitude'] >= -74.3) & (df_test['pickup_longitude'] <= -73.7)].copy()
df_test


# In[113]:


df_test.shape


# In[116]:


#latitude from 40.4961° N (southernmost point) to 40.9156°(pickup)
df_test = df_test[(df_test['pickup_latitude'] >= 40.4961) & (df_test['pickup_latitude'] <= 40.9156)].copy()
df_train = df_train[(df_train['pickup_latitude'] >= 40.4961) & (df_train['pickup_latitude'] <= 40.9156)].copy()


# In[118]:


#longitude for dropoff
df_test = df_test[(df_train['dropoff_longitude'].ge(-74.3)) & (df_test['dropoff_longitude'].le(-73.7))].copy()
df_train = df_train[(df_train['dropoff_longitude'].ge(-74.3)) & (df_train['dropoff_longitude'].le(-73.7))].copy()


# In[119]:


#latitude for dropoff
df_test = df_test[(df_test['dropoff_latitude'] >= 40.4961) & (df_test['dropoff_latitude'] <= 40.9156)].copy()
df_train = df_train[(df_train['dropoff_latitude'] >= 40.4961) & (df_train['dropoff_latitude'] <= 40.9156)].copy()


# In[121]:


df_train.shape


# In[122]:


df_test.shape


# In[145]:


# Conversion factor from miles to kilometers
conversion_factor = 1.60934
df_train['distance_km'] = df_train['distance'] * conversion_factor
df_test['distance_km'] =  df_test['distance'] * conversion_factor


# ### Data Visualization

# In[143]:


df_train['pickup_day_of_the_week'].value_counts().sort_values(ascending=False)


# In[ ]:


#Friday has the highest and monday has the least.


# In[144]:


df_test['pickup_day_of_the_week'].value_counts().sort_values(ascending=False)


# In[ ]:


#Friday has the highest and monday has the least.


# #### Vendor_ID

# In[149]:


vendor_counts = df_train['vendor_id'].value_counts()

vendor_counts.plot(kind='bar')

plt.title('Vendor ID Frequency')
plt.xlabel('Vendor ID')
plt.ylabel('Frequency')

plt.show()


# In[ ]:


#In this plot you can see that vendor 2 has the highest count than vendor 1. So most of the trip taken by the vendor id 2


# In[152]:


#test
vendor_counts = df_test['vendor_id'].value_counts()

vendor_counts.plot(kind='bar')

plt.title('Vendor ID Frequency')
plt.xlabel('Vendor ID')
plt.ylabel('Frequency')

plt.show()


# In[ ]:


#(test)In this plot you can see that vendor 2 has the highest count than vendor 1. So most of the trip taken by the vendor id 2


# #### store_and_fwd_flag

# In[158]:


store_forward_count = df_train['store_and_fwd_flag'].value_counts()
plt.pie(store_forward_count, labels=store_forward_count.index, autopct='%1.1f%%')

plt.title('store_forward Distribution')

plt.show()


# In[ ]:


#you can see only 0.6 percent that were forwarded


# In[159]:


store_forward_count = df_test['store_and_fwd_flag'].value_counts()
plt.pie(store_forward_count, labels=store_forward_count.index, autopct='%1.1f%%')

plt.title('store_forward Distribution')


plt.show()


# In[ ]:


#we can see only 0.5 percent that were forwarded


# In[162]:


df_train['dropoff_day_of_the_week'] = df_train['dropoff_datetime'].apply(lambda x: calendar.day_name[x.weekday()])


# In[163]:


figure,ax= plt.subplots(nrows=1,ncols=2,figsize=(15,5))
sns.countplot(x='pickup_day_of_the_week',data=df_train,ax=ax[0])
ax[0].set_title("No of pickups done on each day")
sns.countplot(x='dropoff_day_of_the_week',data=df_train,ax=ax[1])
ax[1].set_title('No of dropoffs done on each day')


# In[ ]:


#Most of the taxis are booked on weekends


# In[166]:


df_train


# #### Pickup and Dropdown timezone

# In[167]:


figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
sns.countplot(x='pickup_timezone',data=df_train,ax=ax[0])
ax[0].set_title('The distribution of number of pickups on each part of the day')
sns.countplot(x='dropoff_timezone',data=df_train,ax=ax[1])
ax[1].set_title('The distribution of number of dropoffs on each part of the day')
plt.tight_layout()


# In[168]:


#most of the taxi are booked on eveninngs


# #### trip_duration by month

# In[169]:


sns.lineplot(x='pickup_month',y='trip_duration',data=df_train,color='green')
plt.show()


# In[ ]:


#After feb there is a rise in the trip duration every month


# #### pickup hour

# In[170]:


plt.figure(figsize=(10,6))
sns.countplot(x=df_train["pickup_hour"])
plt.title("ditribution of pickup during 24 hours")
plt.show()


# In[ ]:


Distribution of pickup and dropoff hours follows same pattern, it shows that most of the pickups and dropoffs are in the evening. We can see that people often use taxi services to get to their workplaces in the mornings after 10AM. and busiet time is 6PM to 7PM.


# #### Monthly pickup by vendor

# In[176]:


monthly_pickup_by_vendor=df_train.groupby(["pickup_month","vendor_id"]).size()
monthly_pickup_by_vendor = monthly_pickup_by_vendor.unstack()#pivot table


# In[180]:


monthly_pickup_by_vendor.plot(kind = 'line', figsize = (8,4))
plt.title('Vendor trip per month')
plt.xlabel('Pickup Months')
plt.ylabel('Trips count')
plt.show()


# In[ ]:


#the trip counts are at the highest in the month of march for both vendors and least in the month of january


# #### passenger count

# In[184]:


passenger_c = df_train.passenger_count.value_counts()
passenger_c


# In[188]:


plt.figure(figsize=(10,5))
sns.countplot(x=df_train["passenger_count"])
plt.title('Distribution of passenger count')
plt.show()


# In[189]:


#Most of the rides are done by one person.That means people are prefering solo rides.


# #### Trip Duration in minutes

# In[196]:


bins = [0, 1, 10, 30, 60, 1440, 1440*2, 50000]
labels = ['less than 1 min', 'within 10 mins', 'within 30 mins', 'within 1 hour', 'within 1 day', 'within 2 days', 'more than 2 days']


df_train['trip_duration_bins'] = pd.cut(df_train['trip_in_minutes'], bins=bins, labels=labels)

df_train.groupby('trip_duration_bins').size().plot(kind='bar')

plt.title('Bar Plot for Trip Duration')
plt.xlabel('Trip Duration')
plt.ylabel('Trip Counts')
plt.xticks(rotation=45)

plt.show()


# In[ ]:


#within a day most of the trip duration is for 30 min and some are for 1 hour and within a day is rare.


# #### distribution of differnt features

# In[198]:


ax = df_train['passenger_count'].plot.box()


# In[199]:


fig, ax = plt.subplots(figsize=(8, 6))

box_props = dict(color='steelblue', linewidth=2)
whisker_props = dict(color='black', linewidth=1.5)
median_props = dict(color='red', linewidth=2)
ax.boxplot(df_train['passenger_count'], showfliers=False, boxprops=box_props, whiskerprops=whisker_props, medianprops=median_props)

ax.set_xlabel('Passenger Count')
ax.set_ylabel('Number of Trips')
ax.set_title('Box Plot of Passenger Count')

plt.show()


# In[201]:


#using the both box plots visualizations we can say that median = 1 and there are outliers in the passenger count.


# In[215]:


ax = df_train['passenger_count'].plot.hist(figsize=(4,4))


# In[216]:


fig, ax = plt.subplots(figsize=(8, 6))

ax.hist(df_train['passenger_count'], bins=10, color='steelblue', edgecolor='black')

ax.set_xlabel('Passenger Count')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Passenger Count')

ax.set_xticks(range(1, df_train['passenger_count'].max() + 1))

plt.show()


# In[210]:


#distance_km


# In[209]:


ax = df_train['distance_km'].plot.box()


# In[206]:


fig, ax = plt.subplots(figsize=(8, 6))

box_props = dict(color='steelblue', linewidth=2)
whisker_props = dict(color='black', linewidth=1.5)
median_props = dict(color='red', linewidth=2)
ax.boxplot(df_train['distance_km'], showfliers=False, boxprops=box_props, whiskerprops=whisker_props, medianprops=median_props)

ax.set_xlabel('Distance (km)')
ax.set_ylabel('Number of Trips')
ax.set_title('Box Plot of Trip Distance')

plt.show()


# In[213]:


#By these two visualizations of box plot we can say that median around 2 and there are outliers in the distance


# In[217]:


ax = df_train['distance_km'].plot.hist(figsize=(4,4))


# In[224]:


fig, ax = plt.subplots(figsize=(8, 6))

ax.hist(df_train['distance_km'], bins=10, color='steelblue', edgecolor='black')

ax.set_xlabel('Distance (km)')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Trip Distance')

x_ticks = [0, 5, 10, 15, 20, 25, 30, 35, 40]
x_labels = ['0', '5', '10', '15', '20', '25', '30', '35', '40']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()


# In[225]:


#you can see there is skewness in this histogram plot for distance.


# In[226]:


#trip_duration_in_minute


# In[229]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(df_train['trip_in_minutes'], bins=10, color='steelblue', edgecolor='black')
ax1.set_xlabel('Trip Duration (minutes)')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Trip Duration')

ax2.boxplot(df_train['trip_in_minutes'], vert=False)
ax2.set_xlabel('Trip Duration (minutes)')
ax2.set_title('Box Plot of Trip Duration')

plt.tight_layout()

plt.show()


# In[234]:


#in this visualization we can see that there is skewness in hsitogram and outliers in box plot.


# In[235]:


#pickup_hour


# In[236]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(df_train['pickup_hour'], bins=24, color='steelblue', edgecolor='black')
ax1.set_xlabel('Pickup Hour')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Pickup Hour')

ax2.boxplot(df_train['pickup_hour'], vert=False)
ax2.set_xlabel('Pickup Hour')
ax2.set_title('Box Plot of Pickup Hour')

plt.tight_layout()

plt.show()


# In[237]:


#drop_off hour


# In[242]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(df_train['dropoff_hour'], bins=24, color='steelblue', edgecolor='black')
ax1.set_xlabel('Dropoff Hour')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Dropoff Hour')

ax2.boxplot(df_train['dropoff_hour'], vert=False)
ax2.set_xlabel('Dropoff Hour')
ax2.set_title('Box Plot of Dropoff Hour')

plt.tight_layout()

plt.show()


# ### **(histogram) distance and trip_duration graphs are highly skewed.**
# 
# ### **(boxplot) distance and trip_duration columns have a lot outliers as well**

# In[312]:


#Lets check this for df_test also


# In[315]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.hist(df_test['passenger_count'], bins=10, color='purple', edgecolor='black')
ax1.set_xlabel('Passenger Count')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Passenger Count')

ax2.boxplot(df_test['passenger_count'], vert=False)
ax2.set_xlabel('Passenger Count')
ax2.set_title('Box Plot of Passenger Count')

plt.tight_layout()

plt.show()


# In[317]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(df_test['distance'], bins=50, color='green', edgecolor='black')
ax1.set_xlabel('Distance (in km)')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Distance')

ax2.boxplot(df_test['distance'], vert=False)
ax2.set_xlabel('Distance (in km)')
ax2.set_title('Box Plot of Distance')

plt.tight_layout()

plt.show()


# In[321]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(df_test['pickup_hour'], bins=24, color='blue', edgecolor='black')
ax1.set_xlabel('Pickup Hour')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Pickup Hour')

ax2.boxplot(df_test['pickup_hour'], vert=False)
ax2.set_xlabel('Pickup Hour')
ax2.set_title('Box Plot of Pickup Hour')

plt.tight_layout()

plt.show()


# ### (histplot) distance and passenger count graph are highly skewed.
# ### (boxplot) distance and passenger count columns have a lot outliers as well

# ## Multicollinearity and correlation check

# ### **Heatmap**

# In[260]:


# correlation matrix calculation
correlation_matrix = df_train.corr()

# filteration of highly correlated features
highly_correlated = correlation_matrix.abs() > 0.8
correlated_features = set()

# names of the highly correlated features
for i in range(len(highly_correlated.columns)):
    for j in range(i):
        if highly_correlated.iloc[i, j]:
            feature_i = highly_correlated.columns[i]
            feature_j = highly_correlated.columns[j]
            correlated_features.add(feature_i)
            correlated_features.add(feature_j)

# conversion into a list
correlated_features = list(correlated_features)

# Print the highly correlated features
print("Highly correlated features:")
print(correlated_features)


# In[324]:


#these are the highly correlated features. so we can remove it to red


# In[257]:


#heatmap
plt.figure(figsize=(18, 12))
correlation = df_train.corr()
sns.heatmap(correlation, annot=True,linewidths=0.5)
plt.show()


# In[323]:


#df_test


# In[322]:


#heatmap
plt.figure(figsize=(18, 12))
correlation = df_test.corr()
sns.heatmap(correlation, annot=True,linewidths=0.5)
plt.show()


# In[325]:


correlation_matrix_test = df_test.corr()

highly_correlated_test = correlation_matrix_test.abs() > 0.8
correlated_features_test = set()

for i in range(len(highly_correlated_test.columns)):
    for j in range(i):
        if highly_correlated_test.iloc[i, j]:
            feature_i = highly_correlated_test.columns[i]
            feature_j = highly_correlated_test.columns[j]
            correlated_features_test.add(feature_i)
            correlated_features_test.add(feature_j)
            
correlated_features_test = list(correlated_features_test)

print("Highly correlated features in df_test:")
print(correlated_features_test)


# ## Checking skewness

# In[267]:


# Create a side-by-side comparison of two distribution plots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot the original distribution of trip_duration
sns.distplot(df_train["trip_duration"], color="red", ax=ax[0], hist=False, kde_kws={"shade": True, "linewidth": 2})
ax[0].set_title("Original Distribution")

# Plot the distribution after applying log transformation
sns.distplot(np.log10(df_train["trip_duration"]), color="green", ax=ax[1], hist=False, kde=True, kde_kws={"shade": True, "linewidth": 2})
ax[1].set_title("Distribution after Log Transformation")

# Show the plots
plt.show()


# In[268]:


#By above distribution we can see that target variable is higly right skewed .to remove the skewness we apply log transformation.after transformation we found normal distribution of targer variable.


# # **Outlier Removal (Quartile Method)**

# In[273]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

sns.boxplot(x=df_train["trip_duration"], ax=ax[0])
ax[0].set_title("Trip Duration Box Plot")

sns.boxplot(x=df_train["distance"], ax=ax[1])
ax[1].set_title("Distance Box Plot")

sns.boxplot(x=df_train["passenger_count"], ax=ax[2])
ax[2].set_title("Passenger Count Box Plot")

plt.show()


# In[275]:


Q1_trip_duration = df_train["trip_duration"].quantile(0.25)
print('first quartile value ie 25th percentile of trip duration:',Q1_trip_duration)
Q3_trip_duration = df_train["trip_duration"].quantile(0.75)
print('third quartile value ie 75th percentile of trip duration:',Q3_trip_duration)


# In[277]:


IQR_trip_duration = Q3_trip_duration - Q1_trip_duration

lower_bound_trip_duration = Q1_trip_duration - 1.5 * IQR_trip_duration
upper_bound_trip_duration = Q3_trip_duration + 1.5 * IQR_trip_duration
print('The lower limit of trip duration:',lower_bound_trip_duration)
print('The upper limit of trip duration:',upper_bound_trip_duration)


# In[278]:


df_train.shape


# In[279]:


df_train = df_train[(df_train["trip_duration"] >= lower_bound_trip_duration) & (df_train["trip_duration"] <= upper_bound_trip_duration)]


# In[280]:


df_train.shape


# In[281]:


Q1_distance = df_train["distance"].quantile(0.25)
print('first quartile value ie 25th percentile of distance:',Q1_distance)

Q3_distance = df_train["distance"].quantile(0.75)
print('third quartile value ie 75th percentile of distance:',Q3_distance)


# In[283]:


IQR_distance = Q3_distance - Q1_distance

lower_bound_distance = Q1_distance - 1.5 * IQR_distance
upper_bound_distance = Q3_distance + 1.5 * IQR_distance

print('The lower limit of distance:',lower_bound_distance)
print('The upper limit of distance:',upper_bound_distance)


# In[284]:


df_train = df_train[(df_train["distance"] >= lower_bound_distance) & (df_train["distance"] <= upper_bound_distance)]


# In[285]:


df_train.shape


# In[287]:


Q1_passenger_count = df_train["passenger_count"].quantile(0.25)
print('first quartile value ie 25th percentile of distance:',Q1_distance)

Q3_passenger_count = df_train["passenger_count"].quantile(0.75)
print('third quartile value ie 75th percentile of distance:',Q3_distance)


# In[288]:


IQR_passenger_count = Q3_passenger_count - Q1_passenger_count

lower_bound_passenger_count = Q1_passenger_count - 1.5 * IQR_passenger_count
upper_bound_passenger_count = Q3_passenger_count + 1.5 * IQR_passenger_count

print('The lower limit of distance:',lower_bound_distance)
print('The upper limit of distance:',upper_bound_distance)


# In[289]:


df_train = df_train[(df_train["passenger_count"] >= lower_bound_passenger_count) & (df_train["passenger_count"] <= upper_bound_passenger_count)]


# In[290]:


df_train.shape


# In[329]:


#df_test


# In[331]:


Q1_distance = df_test['distance'].quantile(0.25)
Q3_distance = df_test['distance'].quantile(0.75)

Q1_passenger_count = df_test['passenger_count'].quantile(0.25)
Q3_passenger_count = df_test['passenger_count'].quantile(0.75)

IQR_distance = Q3_distance - Q1_distance
IQR_passenger_count = Q3_passenger_count - Q1_passenger_count

lower_bound_distance = Q1_distance - 1.5 * IQR_distance
upper_bound_distance = Q3_distance + 1.5 * IQR_distance

lower_bound_passenger_count = Q1_passenger_count - 1.5 * IQR_passenger_count
upper_bound_passenger_count = Q3_passenger_count + 1.5 * IQR_passenger_count

df_test = df_test[(df_test['distance'] >= lower_bound_distance) & (df_test['distance'] <= upper_bound_distance)]
df_test = df_test[(df_test['passenger_count'] >= lower_bound_passenger_count) & (df_test['passenger_count'] <= upper_bound_passenger_count)]


# In[292]:


figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
sns.distplot(df_train['distance'], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 2}, color="green", ax=ax[0])
sns.distplot(df_train['trip_duration'], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 2}, color="green", ax=ax[1])

ax[0].set_title("Distribution of Distance")
ax[0].set_xlabel("Distance (in km)")
ax[0].set_ylabel("Density")

ax[1].set_title("Distribution of Trip Duration")
ax[1].set_xlabel("Trip Duration (in seconds)")
ax[1].set_ylabel("Density")

plt.show()


# In[293]:


#both distance and trip duration are now nearly to normal distribution


# In[332]:


#df_test(distance)


# In[336]:


figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
sns.distplot(df_test['distance'], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 2}, color="green", ax=ax)

ax.set_title("Distribution of Distance")
ax.set_xlabel("Distance (in km)")
ax.set_ylabel("Density")


# In[337]:


#this is almost near to normal distribution


# ## **Categorical variable conversion**

# ### ONE HOT ENCODING

# In[297]:


#adding dummy variable to convert categorical data to numerical data through one hot encoding
df_train=pd.get_dummies(df_train,columns=['pickup_day_of_the_week', 'dropoff_day_of_the_week'],drop_first=True)


# In[327]:


df_test = pd.get_dummies(df_test,columns=['pickup_day_of_the_week'],drop_first=True)


# In[328]:


df_test.shape


# In[298]:


df_train.shape


# #### now i am going to remove the features which are correlated based on the heatmap

# In[302]:


#Highly correlated features:
#['dropoff_month', 
#'pickup_month',
#'trip_duration', 'distance_km', 'pickup_hour', 'distance', 'trip_in_minutes', 'dropoff_hour']


# In[307]:


features=['vendor_id', 'passenger_count', 'distance', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag','pickup_day_of_the_week_Monday',
       'pickup_day_of_the_week_Saturday', 'pickup_day_of_the_week_Sunday',
       'pickup_day_of_the_week_Thursday', 'pickup_day_of_the_week_Tuesday',
       'pickup_day_of_the_week_Wednesday']


# In[308]:


final_df=df_train[features]
final_df.shape


# # Supervised Machine Learning of NYC taxi trip duration

# In[354]:


# define a  function to calculate evaluation metrics
def evaluation_metrics (x_train,y_train,y_predicted):
    
#calculation of mean_squared_error(MSE) using mean square function in the sckit_learn package.
  MSE=round(mean_squared_error(y_true=y_train, y_pred=y_predicted),4)

#calculation of root mean square error by square rooting the mean square error
  RMSE=math.sqrt(MSE)

#using r2_score function in the sckit learn package we calculated the R-squared score (coefficient of determination).
  R2_score=r2_score(y_true=y_train, y_pred=y_predicted)

#Adjusted the R-squared score for the number of features in the model.
  Adjusted_R2_score=1-((1-( R2_score))*(x_train.shape[0]-1)/(x_train.shape[0]-x_train.shape[1]-1))
    
#Print the calculated evaluation metrics
  print("Mean Squared Error:",MSE,"Root Mean Squared Error:", RMSE)
  print("R2 Score :",R2_score,"Adjusted R2 Score :",Adjusted_R2_score)

#Plotting Actual and Predicted Values(plotted first 100 points)
  plt.figure(figsize=(18,6))
  plt.plot((y_predicted)[:100], color='red') 
  plt.plot(np.array(y_train)[:100], color='green')
  plt.legend(["Predicted","Actual"])
  plt.title('Actual and Predicted Time Duration')

  #return(MSE,RMSE,R2_score,Adjusted_R2_score)
     


# ## Linear Regression

# In[355]:


x=final_df[features]
y=df_train["trip_in_minutes"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[356]:


lr=LinearRegression()
lr.fit(x_train,y_train)
a=lr.score(x_train, y_train)
y_pred_train = lr.predict(x_train)
y_pred_test = lr.predict(x_test)

evaluation_metrics(x_train,y_train,y_pred_train)


# In[357]:


evaluation_metrics(x_test,y_test,y_pred_test)

