#!/usr/bin/env python
# coding: utf-8

# **Project description**
# To study (about Yandex.Afisha):
#     How people use the product
#     
#     When they start to buy
#     
#     How much money each customer brings
#     
#     When they pay off
#     
# From:
#     
#     Server logs with data on Yandex.Afisha visits from January 2017 through
#     
#     December 2018
#     
#     Dump file with all orders for the period
#     
#     Marketing expenses statistics

# ### Step 1. Download the data and prepare it for analysis

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt


# In[2]:


visits=pd.read_csv('/datasets/visits_log_us.csv')
orders=pd.read_csv('/datasets/orders_log_us.csv')
costs=pd.read_csv('/datasets/costs_us.csv')


# In[3]:


visits.info()
print(visits.head(3))


# In[4]:


orders.info()
print(orders.head())


# In[5]:


costs.info()
print(costs.head(3))


# In[6]:


visits.columns= visits.columns.str.replace(' ','_').str.lower()
orders.columns= orders.columns.str.replace(' ','_').str.lower()
visits.start_ts= pd.to_datetime(visits.start_ts, dayfirst= True)
orders.buy_ts= pd.to_datetime(orders.buy_ts, dayfirst= True)


# > _start_ts_ and _buy_ts_ columns are changed to datetime format

# <div class="alert alert-success" role="alert">
# Reviewer's comment v. 1:
#     
# It's great that you brought start_ts and buy_ts columns to the datetime format.
# </div>

# orders.info()

# In[7]:


visits['date']= visits['start_ts'].dt.date
orders['date']= orders['buy_ts'].dt.date


# In[8]:


print(visits.head(3))
print(orders.head(3))
visits.info()


# In[9]:


visits['start_ts']= pd.to_datetime(visits['start_ts'], format='%Y-%m-%d %H:%M:%S') 
visits['day']= pd.DatetimeIndex(visits['start_ts']).dayofweek
visits['week']= pd.DatetimeIndex(visits['start_ts']).week
visits['month']= pd.DatetimeIndex(visits['start_ts']).month

orders['buy_ts']= pd.to_datetime(orders['buy_ts'], format='%Y-%m-%d %H:%M:%S') 
orders['day']= pd.DatetimeIndex(orders['buy_ts']).dayofyear
orders['week']= pd.DatetimeIndex(orders['buy_ts']).week
orders['month']= pd.DatetimeIndex(orders['buy_ts']).month

costs['dt']= pd.to_datetime(costs['dt'], format='%Y-%m-%d')

visits['end_ts']= pd.to_datetime(visits['end_ts'], format='%Y-%m-%d %H:%M:%S') 


# In[10]:


visits.isnull().sum()


# In[11]:


orders.isnull().sum()


# In[12]:


costs.isnull().sum()


# > No missing values in given data

# In[13]:


visits.duplicated().sum()


# In[14]:


orders.info()


# In[15]:


print(orders.duplicated().sum())


# In[16]:


print(costs.duplicated().sum())


# > No duplicated data.

# **Conclusion**

# Data are downloaded and date columns are converted to datetime format

# ### Step 2. Make reports and calculate metrics:

# #### Product
# 

# **How many people use it every day, week, and month?**

# In[17]:


print('Average DAU: ', visits.groupby(visits['start_ts'].dt.date)['uid'].nunique().mean().round(2))


# In[18]:


print('Average WAU: ',visits.groupby(visits['week'])['uid'].nunique().mean().round(2))


# In[19]:


print('Average MAU: ',visits.groupby(visits['month'])['uid'].nunique().mean().round(2))


# >The average MAU is less than four times WAU
# 
# >The average WAU is less than seven times DAU

# In[20]:


fig=px.line(visits.groupby(visits['start_ts'].dt.date)['uid'].nunique().reset_index(), x='start_ts', y='uid',
            labels={
                     "start_ts": "Starting time",
                     "uid": "uid",
                     
                 },title='Daily Active Users')
                         
fig.show()


# >In general, there are more daily users from october to april compared to rest of the year. The spike on the november suggest
# users for black friday. Likewise dip in number of users in april may be due to some technical problem as server crash.

# In[21]:



#import datetime
#d = "2013-W26"
#r = datetime.datetime.strptime(d, "%Y-W%W")
#print(r)


# In[22]:


visits['week_start']=visits['start_ts'].astype('datetime64[W]')
visits['month_start']=visits['start_ts'].astype('datetime64[M]')


# In[23]:


figure_weak=px.line(visits.groupby(visits['week_start'])['uid'].nunique().reset_index(),
                    x='week_start', y='uid',
                   labels={
                     "day_start": "Weeks  of months",
                     "uid": "The number of visitors",
                    
                 },
                title="Weekly active users")
figure_weak.show()

fig.update_layout(
    title="Weekly active users",
    xaxis_title="start_ts",
    yaxis_title="uid",    
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ))


# Weekly active users have similar graph as in daily active users. The only difference is that graph line has less fluctuation, otherwise,
# The trend is similar. The effect of black friday and server crash can still be identified in this graph too.

# In[24]:


figure_month=px.line(visits.groupby(visits['month_start'])['uid'].nunique().reset_index(),
                     x='month_start', y='uid',
                     labels={
                     "month_start": "Months of year",
                     "uid": "The number of visitors",
                    
                 },
                title="The number of website visitors for Month  from June 1 2017 to May 31 2018")
figure_month.show()


# The general trend of higher number  of visitor remain similar but the  rise and fall in number of daily active users are average out.
# 

# **How many sessions are there per day? One user might have more than
# one session.**
# 

# In[ ]:





# In[25]:


dau=visits.groupby(visits['start_ts'].dt.date)['uid'].nunique().reset_index()
daily_sessions= visits.groupby(visits['start_ts'].dt.date)['uid'].count().reset_index()
                         
daily_sessions.rename({'uid':'uid_session'},axis=1, inplace=True)
total2=pd.merge(left = dau , right = daily_sessions, how='outer',on=['start_ts']).fillna(0)
total2.plot(figsize= (17,8))

plt.title('Daily active user and session comparision');
plt.ylabel('total number');
plt.ylabel('Day of the year');
#plt.figure(figsize= (17,8));

                         

        


# The number of session follow the trend of number users. There are almost same number of session as number of daily users during the
# the first 100 days starting from  june 2017.

# In[26]:


print(dau.head())
daily_sessions.head()


# In[27]:


print('Average number of session per user: ', ((daily_sessions['uid_session']/dau['uid']).mean()).round(2))


# **What is the length of each session?**
# 

# In[28]:


visits['session_length']= ((visits['end_ts']-visits['start_ts'])/np.timedelta64(1, 'm')).round().astype(int)
#print('average session length:', visits['duration'].mean() )
print(visits.sort_values(by=['session_length'], ascending= False)['session_length'])
#print(visits.sample(10))


# In[29]:


visits=visits[visits['end_ts'] > visits['start_ts']]


# In[30]:


visits['session_length'].mean().round(2)


# Session are longer up to 711 minutes which could be due to user visit the site and did not log out unless he shut his/her device.
# The negetive sessions are not possible, there are bugs in the data!!

# In[136]:


#session_length['length'] =visits['session_length'].quantile(0.75)+np.percentile(visits['session_length'], 90, axis=0)-np.percentile(visits['session_length'], 10, axis=0)
sns.distplot(visits[visits['session_length']<100]['session_length'], bins=50);
plt.title('Session length');
plt.xlabel('Session length (minutes)');
plt.ylabel('number of sessions');
plt.grid();


# > The session length has half normal distribution as duration could not be negative and mean session length lies about 0 to 2 minutes.

# In[138]:


sns.distplot(visits[visits['session_length']<20]['session_length'], bins=50);
plt.title('Session length');
plt.xlabel('Session length (minutes)');
plt.ylabel('number of sessions');


# By far the maximum session lengths are less than two minutes. There are less number of session as session length increases. 
#  

# In[33]:


fig= px.line(visits.groupby([visits['start_ts'].dt.date, 'device'])['session_length'].mean().reset_index(),
             x='start_ts', y='session_length', color='device')
fig.show()


# The sessions are longer in desktop devices compared to touch devices. Likewise, the dip in number of visitors was due to seb server 
# for desktop devices.

# In[34]:


print('Average session length: ', visits['session_length'].mean())
print("Session length's median: ", visits['session_length'].median())
print("Session length's mode: ", visits['session_length'].mode())


# The session length mean is about 10.7 minutes as there is tail up to 60 minutes. Number of session higher than 5 minutes and lower than 5 minutes are equal.
# Session with lenght having 0 minutes and 1 minutes are by far the maximum. This indicates many visitors are not intrested in product or they mistakenly visited the site.

# **How often do users come back?**

# In[35]:





#creating a cohort with the month of first visit and calculating the cohort lifetime in terms of month
#visits['Start Ts'] = pd.to_datetime(visits['Start Ts'])
first_visit_date = visits.groupby(['uid'])['start_ts'].min()
first_visit_date.name = 'first_visit_date'
visits = visits.merge(first_visit_date,on='uid', how='left')
visits.head()


# In[36]:


#creating a cohort with the month of first visit and calculating the cohort lifetime in terms of month
visits['visit_month'] = visits['start_ts'].astype('datetime64[M]')
visits['first_visit_month'] = visits['first_visit_date'].astype('datetime64[M]')
visits.head()


# In[37]:


#calculating the cohort lifetime
visits['cohort_visits_lifetime'] = visits['visit_month'] - visits['first_visit_month']
visits['cohort_visits_lifetime'] = visits['cohort_visits_lifetime'] / np.timedelta64(1,'M')
visits['cohort_visits_lifetime'] = visits['cohort_visits_lifetime'].round().astype(int)

visits.head()


# In[38]:


#calculating retention rate
cohort_visits = visits.groupby(['first_visit_month', 'cohort_visits_lifetime']).agg({'uid' :'nunique'}).reset_index()
# Build the data frame with cohorts here
initial_users_count = cohort_visits[cohort_visits['cohort_visits_lifetime'] == 0][['first_visit_month','uid']]
# Build the data frame here
initial_users_count = initial_users_count.rename(columns={'uid':'cohort_visits_users'})
# Rename the data frame column
cohort_visits = cohort_visits.merge(initial_users_count,on='first_visit_month')
# Join the data frames cohorts and initital_users_count
cohort_visits['retention'] = cohort_visits['uid']/cohort_visits['cohort_visits_users']
# Calculate retention rate
retention_pivot = cohort_visits.pivot_table(index='first_visit_month',columns=['cohort_visits_lifetime'],values='retention',aggfunc='sum')
retention_pivot


# In[39]:


retention_pivot.mean(axis=0)


# In[40]:


#plotting a heatmap to visualize the retention rate
sns.set(style='white')
plt.figure(figsize=(13, 9))
plt.title('Cohorts: User Retention')
sns.heatmap(
    retention_pivot, cmap="PiYG", annot=True, fmt='.1%', linewidths=1, linecolor='gray', vmin=0.01, vmax=0.09
) ;


# Normally, the retention rate has decreased as month from first visit increases. But, it can be noticed that in november, more users 
# returns back. Likewise, the retention rate for first month until december is higher compared to retention rate for first month after december to june.

# In[41]:


cohort_visits.info()


# In[42]:


retention_second= retention_pivot.rename(columns = {'0': '1', '1': '2','2': '3', '3': '4','4': '5', '5': '6','6': '7', '7': '8','8': '9', '9': '10','10': '11', '11': '12',}, inplace = False)
retention_second.iloc[:, 1].mean().round(4)


# > Average retention rate for the second month of cohort life is 6.5%

#  #### Sales

# In[43]:


pivot_order= pd.pivot_table(orders , index='uid', values='buy_ts',aggfunc={'min'})
pivot_order.rename(columns={ 'uid': 'uid', "min": "first_order_date"})
pivot_visit= pd.pivot_table(visits , index='uid', values='start_ts',aggfunc={'min'})
pivot_visit.rename(columns={ 'uid': 'uid', "min": "first_visit_date"})
conversion=pd.merge(left = pivot_order , right = pivot_visit, how='inner',on=['uid']).fillna(0).rename(columns={ 'min_x': 'first_order_date', "min_y": "first_visit_date"})
conversion['days']=((conversion['first_order_date']-conversion['first_visit_date'])/np.timedelta64(1, 'D')).round().astype(int)
conversion['cohort']= conversion['first_visit_date'].apply(lambda x: x.strftime('%Y-%m'))


print(conversion)


# In[44]:


#conversion_rate=conversion.groupby('cohort')['days','index'].aggfunc={'days':np.mean, 'index':'nunique'}

print(conversion.sort_values(by=['days'], ascending= False))


# In[45]:


plt.hist(conversion['days'],bins= 10, range=(0,10))
plt.title('Conversion days')
plt.xlabel('days')
plt.ylabel('number')
plt.show()


# The users who buy the product mostly buy on the same day they visited the website.

# In[46]:


min_order=orders.groupby(['uid'])['buy_ts'].min().reset_index()
min_order


# In[47]:


min_order=orders.groupby(['uid'])['buy_ts'].min().reset_index()
min_visit=visits.groupby(['uid'])['start_ts'].min().reset_index()
actual_first_visit=min_visit.merge(min_order,on=['uid'],how='left')
print(min_order.head(3))
actual_first_visit.head(3)


# In[48]:


actual_first_visit[actual_first_visit['buy_ts']<actual_first_visit['start_ts']]
actual_first_visit['first_visit']=actual_first_visit.apply(lambda row: row.start_ts if row.start_ts<row.buy_ts                                                          or isinstance(row.buy_ts,pd._libs.tslibs.nattype.NaTType)                                                          else row.buy_ts,axis=1)


# In[49]:


conversion_data=min_order.merge(actual_first_visit[['uid','first_visit']],on=['uid'],how='left')
conversion_data


# In[50]:


conversion_data['conversion_date']=((conversion_data['buy_ts']-conversion_data['first_visit'])/np.timedelta64(1,'D'))


# In[51]:


conversion_data.head()


# In[52]:


conversion_data['Conversion_0d']=conversion_data['conversion_date'].apply(lambda x: x==0)
conversion_data['Conversion_7d']=conversion_data['conversion_date'].apply(lambda x: x<=7)
conversion_data['Conversion_14d']=conversion_data['conversion_date'].apply(lambda x: x<=14)
conversion_data['Conversion_30d']=conversion_data['conversion_date'].apply(lambda x: x<=30)


# In[53]:


conversion_data.head()


# In[54]:


conversion_data['conversion_date'].max().round(3)


# In[55]:


print(conversion_data.sort_values(by=['conversion_date'], ascending= False)['conversion_date'].head(10))


# In[56]:


conversion_time= conversion_data.sort_values(by=['conversion_date'], ascending= False)['conversion_date']
conversion_time.mean()


# In[57]:


conversion_data[conversion_data['Conversion_0d']==True]                           .groupby(['Conversion_0d'])['uid'].nunique().reset_index()['uid'].loc[0]


# In[58]:


conversion_table=[]
for i in ['Conversion_0d','Conversion_7d','Conversion_14d','Conversion_30d']:
    conversion_table.append((i,conversion_data[conversion_data[i]==True]                           .groupby([i])['uid'].nunique().reset_index()['uid'].loc[0]/visits.uid.nunique()*100))


# In[59]:


conversion_table


# In[60]:


from pandas import DataFrame
conversion_table=DataFrame(conversion_table,columns=['Conversion','Rate'])
conversion_table


# In[61]:


fig=px.line(conversion_table,x='Conversion',y='Rate',title='Conversion')
fig.show()


# Conversion rate is increased slightly more than 2 percent in 30 days as it rise from about 11.5 on the same day of 
# visit to approximately 14 percent in 30's day.

# **How many orders do they make during a given period of time?**

# In[62]:


visits['cohort']= visits['start_ts'].apply(lambda x: x.strftime('%Y-%m'))


# In[63]:


order_new=orders.merge(visits[['cohort', 'uid',]], on='uid',how='inner').reset_index(drop=True).drop_duplicates().reset_index(drop=True)
order_new.sample(10)


# In[64]:


number_orders=px.line(order_new.groupby(order_new['cohort'])['buy_ts'].count().reset_index(),
                      x='cohort', y='buy_ts',
                      labels={
                     "month_start": "Months of year",
                     "buy_ts": "The number of sales ",
                    
                 },
                title="The number of sales per cohort")
number_orders.show()


# The number of sales follow the similar trend as the number of visitors over the given period though. 
# Though, conversion rate is about 14% till one month, the number of sales shows user made multiple orders.

# **What is the average purchase size?**

# In[65]:


first_order_date_by_customers = orders.groupby('uid')['buy_ts'].min()
first_order_date_by_customers.name = 'first_buy_date'
orders = orders.merge(first_order_date_by_customers, on='uid')
orders.head()


# In[66]:


print(orders['revenue'].mean())


# The average purchase size is about $5.

# **How much money do they bring? (LTV)**

# In[67]:


orders['first_buy_date'] = orders['first_buy_date'].astype('datetime64[M]')
orders['buy_ts'] = orders['buy_ts'].astype('datetime64[M]')
orders.head()
#orders_grouped_by_cohorts = order.groupby(['first_buy_date', 'buy_ts']).agg({'revenue': 'sum', 'uid': 'nunique'}).reset_index()


# In[68]:


orders_grouped_by_cohorts = orders.groupby(['first_buy_date', 'buy_ts']).agg({'revenue': 'sum', 'uid': 'nunique'}).reset_index()


# In[69]:


orders_grouped_by_cohorts['revenue_per_user'] = ( orders_grouped_by_cohorts['revenue'] / orders_grouped_by_cohorts['uid'])


# In[70]:


orders_grouped_by_cohorts.head()


# In[71]:


print(orders_grouped_by_cohorts['revenue_per_user'].mean())


# Each buyer make purchase of $ 14.73 in average for each cohort.

# In[72]:


orders_grouped_by_cohorts.pivot_table( index='first_buy_date', columns='buy_ts', values='revenue_per_user', aggfunc='mean').head()


# In[73]:


orders_grouped_by_cohorts=orders_grouped_by_cohorts.reset_index()


# In[74]:


orders_grouped_by_cohorts['cohort_lifetime'] = (orders_grouped_by_cohorts['buy_ts']- orders_grouped_by_cohorts['first_buy_date'])
orders_grouped_by_cohorts['cohort_lifetime'] = orders_grouped_by_cohorts['cohort_lifetime'] / np.timedelta64(1, 'M')
orders_grouped_by_cohorts['cohort_lifetime'] = (orders_grouped_by_cohorts['cohort_lifetime'].round().astype('int'))


# In[75]:


orders_grouped_by_cohorts.info()


# In[76]:


orders_grouped_by_cohorts['first_buy']=orders_grouped_by_cohorts['first_buy_date'].astype('datetime64[M]')
orders_grouped_by_cohorts['first_buy_date'] = pd.to_datetime(orders_grouped_by_cohorts['first_buy_date']).dt.strftime('%Y-%m') 


# In[77]:


orders_grouped_by_cohorts


# In[78]:


revenue_per_user_pivot = orders_grouped_by_cohorts.pivot_table( index='first_buy_date',columns='cohort_lifetime', values='revenue_per_user', aggfunc='mean',)
plt.figure(figsize=(13, 9))
plt.title('revenue_per_user')
sns.heatmap(
    revenue_per_user_pivot,
    cmap="PiYG",
    annot=True,
    fmt='.1f',
    linewidths=1,
    linecolor='gray',);


# <div class="alert alert-success" role="alert">
# Reviewer's comment v. 1:
#     
# Yes, a correct values :)
# </div>

# Revenue from each user per month generally decreases over the lifetime but there are so many exceptions. In first month,there are all 
# time low revenue for all cohorts suggest that once buyers buy the product, they buy more next time. In june, september and August
# 2017, there are significant rise in revenue per user.

# In[79]:


purchases_per_user_pivot = orders_grouped_by_cohorts.pivot_table( index='uid',columns='cohort_lifetime', values='revenue_per_user', aggfunc='count',)

purchases_for_six_month=purchases_per_user_pivot.iloc[:, 6]
print('The average purchase for six month:  {}'.format(purchases_for_six_month.mean()))
purchases_for_six_month


# In[80]:


cohort_size_order=orders.groupby('first_buy_date').agg({'uid':'nunique'}).rename(columns={'uid':'n_buyers'}).reset_index()
cohort_revenue_order=orders.groupby(['first_buy_date', 'buy_ts']).agg({'revenue':'sum'}).reset_index()


# In[81]:


ltv_merged = pd.merge(cohort_size_order, cohort_revenue_order, on='first_buy_date')


# In[82]:


ltv_merged


# In[83]:


margin_rate = 1
ltv_merged['gp'] = ltv_merged['revenue'] * margin_rate
ltv_merged['age'] = (ltv_merged['buy_ts'] - ltv_merged['first_buy_date'])/np.timedelta64(1, 'M')
ltv_merged['age'] = ltv_merged['age'].round().astype('int')
ltv_merged['ltv'] = ltv_merged['revenue'] / ltv_merged['n_buyers']
ltv_merged['first_buy_date']=ltv_merged['first_buy_date'].dt.strftime('%Y-%m')


# In[84]:


ltv_merged.head(3)


# In[85]:


pivot_ltv=ltv_merged.pivot_table(index='first_buy_date', columns='age', aggfunc = 'mean', values='ltv').cumsum(axis=1)
plt.figure(figsize=(15, 12))
plt.title('Ltv')
sns.heatmap(
    pivot_ltv,
    cmap="Greens",
    annot=True,
    fmt='.2f',
    linewidths=1,
    linecolor='gray');


# Lifetime value is increasing consistently as users are continiously buying over the period.

# In[86]:


orders_grouped_by_cohorts.head(3)


# In[87]:


revenue_per_user_sum=pd.pivot_table(orders_grouped_by_cohorts, index='first_buy_date', columns='cohort_lifetime', values='revenue_per_user', aggfunc='mean').cumsum(axis=1).round(2)
plt.figure(figsize=(15, 12))
plt.title('Total cumsum Ltv')
sns.heatmap(
    revenue_per_user_sum,
    cmap="BuPu",
    annot=True,
    fmt='.1f',
    linewidths=1,
    linecolor='gray');


# In[88]:


orders_grouped_by_cohorts.head()


# In[89]:


total_revenue=pd.pivot_table(orders_grouped_by_cohorts, index='first_buy_date', columns='cohort_lifetime', values='revenue', aggfunc='sum')
plt.figure(figsize=(15, 12))
plt.title('Total purchase size')
sns.heatmap(
    total_revenue,
    cmap="PiYG",
    annot=True,
    fmt='.1f',
    linewidths=1,
    linecolor='gray');


# In october, november and december the size of cohort purchase size is over 20000.

# In[90]:


cumsum_revenue=pd.pivot_table(orders_grouped_by_cohorts, index='first_buy_date', columns='cohort_lifetime', values='revenue', aggfunc='sum').cumsum(axis=1).round(2)
plt.figure(figsize=(15, 12))
plt.title('Total cumsum purchase size')
sns.heatmap(
    cumsum_revenue,
    annot=True,
    fmt='.1f',
    linewidths=1,
    linecolor='gray');


# In[91]:


orders_grouped_by_cohorts.head()


# In[92]:


pivot_ltv.head()


# In[93]:


print(pivot_ltv.iloc[:, 5].mean().round(3))
#pivot_ltv_six_month_mean['mean']=pivot_ltv_six_month.mean()
#print(pivot_ltv_six_month_mean)

#pivot_ltv_six_month_mean['mean'].mean().round(3)


# #### Marketing
# 

# In[94]:


costs.head()


# In[95]:


costs['costs'].sum()


# In[96]:


costs['cohort'] = costs['dt'].astype('datetime64[M]')


# In[97]:


monthly_costs = costs.groupby(['cohort'])['costs'].sum().reset_index()
#monthly_costs = cost.groupby(['source_id', 'cohort'])['costs'].sum().reset_index()


# In[98]:


monthly_costs.head(2)


# In[99]:


report = pd.merge(orders_grouped_by_cohorts, monthly_costs, left_on='first_buy', right_on='cohort')


# In[100]:


report['cac'] = report['costs'] / report['uid']


# In[101]:


report.sample(5)


# In[102]:


number_orders=px.line(report.groupby(report['cohort'])['cac'].sum().reset_index(),
                      x='cohort', y='cac',
                      labels={
                     "month_start": "Months of year",
                     "buy_ts": "The number of sales ",
                    
                 },
                title="Cost per users ")
number_orders.show()


# Customer aquisition cost has decreased as time elapsed. This indicates more customer are acquired for the same cost on later times.

# In[103]:


cost_chort=px.line(costs.groupby(costs['cohort'])['costs'].sum().reset_index(), 
                   x='cohort', y='costs', 
                  labels={
                     "cohort": "Cohort interval",
                     "costs": "Total cost",
                    
                 },
                title="The total amount of cost for each cohort   from June 1 2017 to May 31 2018")
cost_chort.show()


# In[104]:


costs.head()


# In[105]:


source_cost=px.line(costs.groupby(costs['source_id'])['costs'].sum().reset_index(), 
                   x='source_id', y='costs', 
                  labels={
                     "source_id": "source",
                     "costs": "Total cost",
                    
                 },
                title="The total amount of cost for each source")
source_cost.show()


# In[106]:


total2=pd.pivot_table(costs, index=['cohort'], columns=['source_id'], values=['costs'], aggfunc=['sum'])
total2.plot(figsize=(10,8))

plt.title('Distribution of cost of each source during the time period');
plt.ylabel('costs ');


# > Source 3 has by far the highest cost of marketing

# Costs pattern follow the sales pattern suggest that on increasing marketing cost, the sales also increased.

# In[107]:


report['ROI'] = report['revenue_per_user'] / report['cac']


# In[108]:


report.head()


# Positive ROI indicates shows that investments are paid off. For first coort, ROI is already 53%.

# In[109]:


output = report.pivot_table(
    index='first_buy_date', columns='cohort_lifetime', values='ROI', aggfunc='mean'
)

plt.figure(figsize=(15, 12))
plt.title(' Romi for lifetime')
sns.heatmap(
  output,
    cmap="PiYG",
    annot=True,
    fmt='.2f',
    linewidths=1,
    linecolor='gray');


# Marketing investment is paying over the period for each cohort.

# In[110]:


output.head()


# In[111]:



plt.figure(figsize=(15, 12))
plt.title('Cumsum Romi for lifetime')
sns.heatmap(
  output.cumsum(axis=1).round(2) ,
    annot=True,
    cmap="PiYG",
    fmt='.2f',
    linewidths=1,
    linecolor='gray');


# Return on marketing invesment is upto 142%

# In[112]:


report.head()


# In[113]:


output = report.pivot_table(
    index='first_buy_date', columns='cohort_lifetime', values='ROI', aggfunc='mean'
)

plt.figure(figsize=(15, 12))
plt.title(' Romi for lifetime')
sns.heatmap(
  output,
    cmap="PiYG",
    annot=True,
    fmt='.2f',
    linewidths=1,
    linecolor='gray');


# In[114]:


output.head()


# In[115]:


plt.figure(figsize=(15, 12))
plt.title('Cumsum Romi for lifetime')
sns.heatmap(
  output.cumsum(axis=1).round(2) ,
    cmap="PiYG",
    annot=True,
    fmt='.2f',
    linewidths=1,
    linecolor='gray');


# The return on marketing investment is  142% in september cohort and 133 % in june cohort over the given period.

# In[116]:


source_chort=costs.groupby(costs['source_id'])['costs'].sum()


# In[117]:


source_chort.plot(kind='bar', grid=True, figsize=(20,10), color='#12ee33');
plt.xlabel('Name of Source')
plt.ylabel('Total amount of cost ');
plt.title('The distribtuin of the source and the money spend on them ');


# The cost is by far the highest on source 3.

# In[118]:


groupby_uid=visits.sort_values('start_ts').groupby('uid').agg({ 'source_id':'first'}).reset_index()


# In[119]:


groupby_uid.head()


# In[120]:


revenue_sum=orders.groupby('uid').agg({'revenue':'sum'}).reset_index()
revenue_sum.head()


# In[121]:


buyers=revenue_sum.merge(groupby_uid, on='uid',how='inner')
buyers.head()


# In[122]:


buyers.duplicated().sum()


# In[123]:


revnue_by_source=buyers.groupby('source_id').agg({'uid':'nunique', 'revenue':'sum'}).reset_index()
revnue_by_source


# In[124]:


source_revenue=buyers.groupby('source_id').agg({'uid':'nunique', 'revenue':'sum'})
source_revenue['revenue'].plot(kind='bar', grid=True, figsize=(20,10), color='#088e33');
plt.xlabel('Name of Source')
plt.ylabel('Revenue ');
plt.title('The distribtuin of the source and revenue');


# The revenue collected from third source is not significantly high as investment done on it. It should be futuristic investment on 
# new source otherwise the investment on that source can be dropped.

# In[125]:


revnue_by_source.duplicated().sum()


# In[126]:


costs


# In[127]:


cost_by_source=costs.groupby('source_id')['costs'].sum().reset_index()

cost_by_source


# In[128]:


merged_cost_revenue=revnue_by_source.merge(cost_by_source, on='source_id', how='inner')
merged_cost_revenue


# In[129]:


merged_cost_revenue['cac']=merged_cost_revenue['costs']/merged_cost_revenue['uid']
merged_cost_revenue


# In[130]:


merged_cost_revenue['LTV']=merged_cost_revenue['revenue']/merged_cost_revenue['uid']


# In[131]:


merged_cost_revenue


# In[132]:


merged_cost_revenue['ROI']=merged_cost_revenue['LTV']/merged_cost_revenue['cac']
merged_cost_revenue['net_revenu']=merged_cost_revenue['revenue']-merged_cost_revenue['costs']


# In[133]:


merged_cost_revenue


# In[134]:


output_1 = merged_cost_revenue.pivot_table( index='source_id', values=['cac', 'LTV', 'ROI'], aggfunc='mean'
)

plt.figure(figsize=(15, 12))
plt.title('cac, LTV  and Roi for sources')
sns.heatmap(
  output_1,
    annot=True,
    cmap="Greens",
    fmt='.3f',
    linewidths=1,
    linecolor='gray');


# The return on investment and lifetime values are high in source 1 and 2. On these source more invesment can be done. The source 3 has 
# only 30 % return on investment over the period. So, rather than investing on source 3, its better to invest on other sources, source 5 and source 9 for example. These sources
# have more than 100 % return on investment over the period.

# ### Step 3. Write a conclusion: advise marketing experts how much money to invest and where.

# Its better ko keep investing on source 5 and 9. In source 3 ROI is less and have highest customer acquisition cost. So , its better to 
# shift this investment on source 1 and 2 as they have high ROI and LTV.

# **Product**

# >  DAU:907.99, WAU: 5825.28, MAU:23228.41

# >Average number of sessions per user 1.08

# > Average session length: 11.9 minutes

# > Normally, the retention rate has decreased as month from first visit increases. But, it can be noticed that in november, more users returns back. Likewise, the retention rate for first month until december is higher compared to retention rate for first month after december to june.

# **Sales**

# > Most of the buyers buy on the same day they visited the cite, in 30 days the conversion rate increased by about 2 %

# > After conversion average purchase is only 1.2 for 6 months

# >The average purchase size is about $5.

# > average LTV for six month is $ 7.969 

# **Marketing**

# > The overall cost is highest on december and this is true for each source. In december the cost is by far the highest at about 38k. at that time the highest cost was on source 3 at about 17k . IN total the cost in source 3 is about 140k.

# >Customer aquisition cost has decreased as time elapsed. This indicates more customer are acquired for the same cost on later times.
# The cac start at about 3500 and reach about 5000 in first and second month of the period.

# > The roi is positive in every source so return is satisfactory in every source. However some sources are out

# In[ ]:




