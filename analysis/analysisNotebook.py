#!/usr/bin/env python
# coding: utf-8

# # Homework 2 - IEEE Fraud Detection

# For all parts below, answer all parts as shown in the Google document for Homework 2. Be sure to include both code that justifies your answer as well as text to answer the questions. We also ask that code be commented to make it easier to follow.

# ## Part 1 - Fraudulent vs Non-Fraudulent Transaction

# # TODO: code and runtime results
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# In[119]:


# Taking all imports for processing
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import xgboost as xgb

from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids


# ### Loading train-data in dataframes
# ### Since Part 1 to 5 would only involve following fields
# TransactionID, DeviceType, DeviceInfo, TransactionDT, TransactionAmt, ProductCD, card4, card6, P_emaildomain, R_emaildomain, addr1, addr2, dist1, dist2
# ### Loading above necessary fields into base dataframe

# In[2]:


# loading train-data in dataframes
train_identity_df = pd.read_csv("../data/train_identity.csv")
train_transaction_df = pd.read_csv("../data/train_transaction.csv")
global_df = pd.merge(left=train_transaction_df, right=train_identity_df, left_on=["TransactionID"],
                                       right_on="TransactionID", how="left")
# Since Part 1 to 5 would only involve following fields
# TransactionID,DeviceType,DeviceInfo,TransactionDT,TransactionAmt,ProductCD,card4,card6,P_emaildomain,R_emaildomain,addr1,addr2,dist1,dist2
# Loading above necessary fields into base dataframe
# And performing EDA on these data
basic_fields_df = global_df[["TransactionID", "TransactionDT", "TransactionAmt",
                                              "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", 
                                              "addr1", "addr2", "dist1", "dist2", "isFraud", "DeviceType", "DeviceInfo"]]


# In[4]:


fraudulent_df = basic_fields_df[basic_fields_df.isFraud == 1]
non_fraudulent_df = basic_fields_df[basic_fields_df.isFraud == 0]
rows_in_fraud = fraudulent_df.shape[0]
rows_in_non_fraud = non_fraudulent_df.shape[0]
total_rows = basic_fields_df.shape[0]
print("Fraudulent instances: {0} | Non Fraudulent instances: {1}".format(str(rows_in_fraud), str(rows_in_non_fraud)))


# ### Citation:
# 1. https://stackoverflow.com/questions/41681693/pandas-isnull-sum-with-column-headers (to aggregate over nulls and get their frequency)

# ### [Part 1] Comparing nulls between fraudulent and non-fraudulent data-set ###

# In[5]:


# isnull() returns boolean array for each row
# sum() on isnull() groups null values and take their sum returning a Series
# the Series returned has null value for each column (which can be seen as index for that value in Series)
# In the end, we take transpose() to convert index to column and corresponding row in Series to cell Value for that column
non_fraudulent_null_stats = non_fraudulent_df.isnull().sum().sort_values(ascending=False)
percent_non_fraud_nulls = non_fraudulent_null_stats * 100.0 / rows_in_non_fraud
percent_non_fraud_nulls = percent_non_fraud_nulls.reset_index()
percent_non_fraud_nulls = percent_non_fraud_nulls.rename(columns={'index': 'features', 0: 'null_percent'})

fraudulent_null_stats = fraudulent_df.isnull().sum().sort_values(ascending=False)
percent_fraud_nulls = fraudulent_null_stats * 100.0/ rows_in_fraud
percent_fraud_nulls = percent_fraud_nulls.reset_index()
percent_fraud_nulls = percent_fraud_nulls.rename(columns={'index': 'features', 0: 'null_percent'})

total_null_stats = pd.merge(percent_fraud_nulls, percent_non_fraud_nulls, on='features', suffixes=['_in_fraud', '_in_non_fraud'])

total_null_stats.plot.bar(x='features')


# In[7]:


total_null_stats.head(20)


# ### Observations ###
# 1. Just like the non-fraud transactions, majority of null values in fraud transactions are from distances, devices and email addresses.
# 2. The key difference is in the field of `DeviceInfo` (Windows/Mac OS), `DeviceType` (mobile, desktop), `R_emaildomain` (receipient email).
#   
#     a. For fraud transaction `null` values for DeviceInfo is around 58% of its set, whereas for non-fraud transactions its around 80%. 
#   
#     b. For fraud transaction `null` values for DeviceType is around 45% of its set, whereas for non-fraud transaction its around 77%.
#   
#     c. Also receipent email address are `null` for 45% of its set, whereas for non-fraud transaction its 77%.
#   
#   ### Conclusion (from visualization around null values in Fraud transactions and Non-Fraud transactions) ###
#  One possible explanation for above discrepancy could be: the values for above details like email address are usually not provided or deferred by benevolent transactions, but a fraudster may have provided fake values instead of ignoring like a regular transactor. This in-turn reduced `null` values for fraud transactions.

# ### [Part 1] Exploring card-issuer field (card4) between fraudulent vs non-fraudluent transactions

# In[8]:


fraud_card_issuer_freq = fraudulent_df.card4.value_counts().sort_values(ascending=True)
fraud_card_issuer_freq = fraud_card_issuer_freq.rename_axis('card4').reset_index(name='freq')

non_fraud_card_issuer_freq = non_fraudulent_df.card4.value_counts().sort_values(ascending=True)
non_fraud_card_issuer_freq = non_fraud_card_issuer_freq.rename_axis('card4').reset_index(name='freq')

total_fraud_card_issuer_freq = fraud_card_issuer_freq.freq.sum()
total_non_fraud_card_issuer_freq = non_fraud_card_issuer_freq.freq.sum()

fraud_card_issuer_freq['percent'] = fraud_card_issuer_freq['freq'] * 100.0 / total_fraud_card_issuer_freq
non_fraud_card_issuer_freq['percent'] = non_fraud_card_issuer_freq['freq'] * 100.0 / total_non_fraud_card_issuer_freq

combined_card_issuer_freq = pd.merge(fraud_card_issuer_freq, non_fraud_card_issuer_freq, on='card4', how='outer', suffixes=['_in_fraud', '_in_non_fraud'])

combined_card_issuer_freq[['card4', 'percent_in_fraud', 'percent_in_non_fraud']].plot.bar(x='card4')
plt.tight_layout()


# ### Conclusion (card4):
# 1.  Except for minor bump for card-issuer 'discover' the usage for all types of card-issuers (american express, visa, etc.) is similar between fraudulent vs non-fraudulent transactions
# 2.  The minor bump in 'discover' category which is 2 % for fraud transactions vs 1 % for non-fraud transactions, may indicate that fraudster are twice as likely to use 'discover' than their non-fraudster counterparts

# In[9]:


combined_card_issuer_freq


# ### [Part 1] Exploring card-type values (card6 field) between fraudulent vs non-fraudulent transactions ###

# In[10]:


fraud_card_freq = fraudulent_df.card6.value_counts().sort_values(ascending=True)
fraud_card_freq = fraud_card_freq.rename_axis("card6").reset_index(name="freq")

non_fraud_card_freq = non_fraudulent_df.card6.value_counts().sort_values(ascending=True)
non_fraud_card_freq = non_fraud_card_freq.rename_axis("card6").reset_index(name="freq")

total_non_fraud_freq = non_fraud_card_freq.freq.sum()
total_fraud_freq = fraud_card_freq.freq.sum()

fraud_card_freq['percent'] = fraud_card_freq['freq'] * 100.0 / total_fraud_freq
non_fraud_card_freq['percent'] = non_fraud_card_freq['freq'] * 100.0 / total_non_fraud_freq

combined_card_freq = pd.merge(fraud_card_freq[['card6', 'percent']], non_fraud_card_freq[['card6', 'percent']], on='card6', how='outer', suffixes=["_in_fraud", '_in_non_fraud'])
# Outer join results in NAN for non-existent values in opposite sets, so filling with 0, since we are dealing with percentages
combined_card_freq = combined_card_freq.fillna(0.0) 
combined_card_freq['percent_exp_fraud'] = np.cbrt(combined_card_freq['percent_in_fraud'])
combined_card_freq['percent_exp_non_fraud'] = np.cbrt(combined_card_freq['percent_in_non_fraud'])

combined_card_freq


# In[11]:


figures, axes = plt.subplots(1,2)
combined_card_freq.plot.bar(x='card6', ax=axes[0], y=['percent_in_fraud', 'percent_in_non_fraud'])
combined_card_freq.plot.bar(x='card6', ax=axes[1], y=['percent_exp_fraud', 'percent_exp_non_fraud'])
axes[0].set_title("normal scale")
axes[1].set_title("exponential scale")
plt.tight_layout()


# ### Conclusion (card6: card-type) ###
# 1. Almost twice the proportional times fraud transactions are carried out through credit card than non-fraud transactions (viz., 48%, 24%)
# 2. Fraudulent transactions are carried out relatively less amount of times through debit card than non-fraud transactions (viz., 51%, 75%)

# ### [Part 1] Exploring DeviceInfo between fraudulent and non-fraudulent transactins ###

# In[12]:


non_null_device_info_count_fraud = fraudulent_df.DeviceInfo.count()
non_null_device_info_count_non_fraud = non_fraudulent_df.DeviceInfo.count()

figure, axes = plt.subplots(1,2)

top_10_fraud_device_counts = fraudulent_df.DeviceInfo.value_counts().to_frame()[0:10]
top_10_fraud_device_counts = top_10_fraud_device_counts.rename(columns={"DeviceInfo": "count"})

top_10_nonfraud_device_counts = non_fraudulent_df.DeviceInfo.value_counts().to_frame()[0:10]
top_10_nonfraud_device_counts = top_10_nonfraud_device_counts.rename(columns={"DeviceInfo": "count"})

# Converting to percentages
# top10FraudDeviceInfos.count = (top10FraudDeviceInfos['count'] * 100) / non_null_device_info_count_fraud
top_10_fraud_device_counts['percent'] = top_10_fraud_device_counts['count'].apply(lambda x: (x * 100)/non_null_device_info_count_fraud)
top_10_nonfraud_device_counts['percent'] = top_10_nonfraud_device_counts['count'].apply(lambda x: (x * 100)/non_null_device_info_count_non_fraud)

top_10_fraud_device_counts.plot.bar(ax=axes[0], y='percent')
top_10_nonfraud_device_counts.plot.bar(ax=axes[1], y='percent')

axes[0].set_title("Fraud device-infos")
axes[1].set_title("Non Fraud device-infos")

plt.tight_layout()


# ### [Part 1] Exploring DeviceType field between fraudulent vs non-fraudulent transactions ###

# In[13]:


fraud_device_type_freq = fraudulent_df.DeviceType.value_counts().sort_values(ascending=True)
fraud_device_type_freq = fraud_device_type_freq.rename_axis('device_type').reset_index(name='freq')

non_fraud_device_type_freq = non_fraudulent_df.DeviceType.value_counts().sort_values(ascending=True)
non_fraud_device_type_freq = non_fraud_device_type_freq.rename_axis('device_type').reset_index(name='freq')

total_fraud_device_type = fraud_device_type_freq.freq.sum()
total_non_fraud_device_type = non_fraud_device_type_freq.freq.sum()

fraud_device_type_freq['percent'] = fraud_device_type_freq['freq'] * 100.0 / total_fraud_device_type
non_fraud_device_type_freq['percent'] = non_fraud_device_type_freq['freq'] * 100.0 / total_non_fraud_device_type

combined_device_type = pd.merge(fraud_device_type_freq, non_fraud_device_type_freq, on='device_type', how='outer', suffixes=['_in_fraud', '_in_non_fraud'])

combined_device_type[['device_type', 'percent_in_fraud', 'percent_in_non_fraud']].plot.bar(x='device_type')
plt.tight_layout()


# ### Conclusion (DeviceType): ###
# 1. Desktop devices are used less in fraudulent transactions, while mobile devices are used heavily for fraudulent transactions
# 

# ### [Part 1] Exploring DeviceInfo field between fraudulent vs non-fraudulent transacstions ###

# In[14]:


suspicious_device_agents = ['hi6210sft Build/MRA58K', 'SM-A300H Build/LRX22G', 'LG-D320 Build/KOT49I.V10a']
percents_in_innocent_transaction = [non_fraudulent_df[non_fraudulent_df.DeviceInfo == a].shape[0] for a in suspicious_device_agents]
percents_in_innocent_transaction = [p * 100.0 / non_null_device_info_count_non_fraud  for p in percents_in_innocent_transaction]
percents_in_innocent_transaction = [round(p, 3) for p in percents_in_innocent_transaction]

percents_in_fraud_transaction = [fraudulent_df[fraudulent_df.DeviceInfo == a].shape[0] for a in suspicious_device_agents]
percents_in_fraud_transaction = [p * 100.0 / non_null_device_info_count_fraud  for p in percents_in_fraud_transaction]
percents_in_fraud_transaction = [round(p, 3) for p in percents_in_fraud_transaction]

i = 0
df_dic = {}
df_dic['transaction_type'] = ['fraud', 'non_fraud']
for agent in suspicious_device_agents:
    df_dic[agent] = [percents_in_fraud_transaction[i], percents_in_innocent_transaction[i]]
    print("Percentage for agent: " + agent +
          ", fraud: " + str(percents_in_fraud_transaction[i]) + ", non-fraud: " + str(percents_in_innocent_transaction[i]))
    i += 1
suspicious_device_infos = pd.DataFrame(df_dic)


for agent in suspicious_device_agents:
#   Amplifying values to understand the relative scale
    suspicious_device_infos[agent] = np.sqrt(suspicious_device_infos[agent])
suspicious_device_infos.plot.bar(x='transaction_type')


# ### Conclusion (DeviceInfo): ###
# 1.  We can observe that for both fraudulent and non-fraudulent transactions, device agents: Windows/iOS/MacOS have been used most frequently
# 2.  But, for some special devices ('hi6210sft Build/MRA58K', 'SM-A300H Build/LRX22G', 'LG-D320 Build/KOT49I.V10a') have frequency in the range of [0.7, 2] % which may not seem significant at first glance. But it is at least 70~ times more frequent than their non fraudulent counterparts [0.009, 0.031] %
# 3.  A quick search on this agents reveals that these are agents of mobile devices.
# 4.  In summary, it seems that fraudulent transactions are done more on mobile devices

# ### [Part 1] Exploring values of addr1 (field) for fraud vs non-fraud transactions ###

# ### Citation:
#  https://stackoverflow.com/questions/51749208/plotting-two-histograms-from-a-pandas-dataframe-in-one-subplot-using-matplotlib

# In[15]:


fraud_addr1_freq = fraudulent_df.addr1.value_counts().sort_values(ascending=False)
fraud_addr1_freq = fraud_addr1_freq.rename_axis('addr1').reset_index(name='freq')
fraud_addr1_freq['percents'] = fraud_addr1_freq['freq'] * 100.0 / fraud_addr1_freq.freq.sum()

non_fraud_addr1_freq = non_fraudulent_df.addr1.value_counts().sort_values(ascending=False)
non_fraud_addr1_freq = non_fraud_addr1_freq.rename_axis('addr1').reset_index(name='freq')
non_fraud_addr1_freq['percents'] = non_fraud_addr1_freq['freq'] * 100.0 / non_fraud_addr1_freq.freq.sum()

fraud_addr1_freq = fraud_addr1_freq[0:10]
non_fraud_addr1_freq = non_fraud_addr1_freq[0:10]

figures, axes = plt.subplots(1, 2)
fraud_addr1_freq[['addr1', 'percents']].plot.bar(x='addr1', y = ['percents'], ax=axes[0])
non_fraud_addr1_freq[['addr1', 'percents']].plot.bar(x='addr1', y = ['percents'], ax=axes[1])

axes[0].set_title('fraud')
axes[1].set_title('non_fraud')

plt.tight_layout()


# ### Conclusion (addr1): ###
# 1.  Some codes like 337 are relatively more frequent in fraudulent transactions than in non-fraudulent transactions

# ### [Part 1] Exploring values of addr2 (field) for fraud vs non-fraud transactions ###

# In[16]:


fraud_addr2_freq = fraudulent_df.addr2.value_counts().sort_values(ascending=False)
fraud_addr2_freq = fraud_addr2_freq.rename_axis('addr2').reset_index(name='addr2_counts')
fraud_addr2_freq['addr2_percents'] = fraud_addr2_freq['addr2_counts'] * 100.0 / fraud_addr2_freq.addr2_counts.sum()

non_fraud_addr2_freq = non_fraudulent_df.addr2.value_counts().sort_values(ascending=False)
non_fraud_addr2_freq = non_fraud_addr2_freq.rename_axis('addr2').reset_index(name='addr2_counts')
non_fraud_addr2_freq['addr2_percents'] = non_fraud_addr2_freq['addr2_counts'] * 100.0 / non_fraud_addr2_freq.addr2_counts.sum()

combined_addr2_freq = pd.merge(fraud_addr2_freq, non_fraud_addr2_freq, on='addr2', how='outer', suffixes=['_in_fraud', '_in_non_fraud'])
combined_addr2_freq = combined_addr2_freq.fillna(0)
combined_addr2_freq = combined_addr2_freq[0:10]

addr2_freq_scaled = combined_addr2_freq[['addr2', 'addr2_percents_in_fraud', 'addr2_percents_in_non_fraud']]
addr2_freq_scaled['addr2_percents_in_fraud'] = np.sqrt(addr2_freq_scaled['addr2_percents_in_fraud'])
addr2_freq_scaled['addr2_percents_in_non_fraud'] = np.sqrt(addr2_freq_scaled['addr2_percents_in_non_fraud'])
addr2_freq_scaled.plot.bar(x='addr2', y=['addr2_percents_in_fraud', 'addr2_percents_in_non_fraud'])
plt.tight_layout()


# In[17]:


combined_addr2_freq['percent_ratio'] = combined_addr2_freq['addr2_percents_in_fraud'] / combined_addr2_freq['addr2_percents_in_non_fraud']


# In[18]:


combined_addr2_freq.head(10)


# ### Conclusion (addr2): ###
# 1.  Countries with country code: 60, 96, 65, 10, indicate significant transactions in fraud data-set relative to non-fraud data-set
# 2.  Distribution percentage of above codes [2.1, 0.6, 0.3, 0.06] % in fraud whereas in non-fraud it is [0.5, 0.1, 0, 0]
# 3.  Though the range of percentage is too small to make major conclusions.
# 4.  Yet, the corresponding ratio is in order of 3 times, 4 times, 45 times, indefinite times
# 5.  Example for country code 10.0, all the transactions were fraud and none of the transactions made from country-code 10.0 are non-fraud

# ### [Part 1] Exploring P_emaildomain (purchaser email) between fraud and non-fraud transactions

# In[19]:


fraud_p_email_freq =  fraudulent_df.P_emaildomain.value_counts().sort_values(ascending=False)
fraud_p_email_freq = fraud_p_email_freq.rename_axis('p_email').reset_index(name='freq')
fraud_p_email_freq['percent'] = fraud_p_email_freq['freq'] * 100.0 / fraud_p_email_freq.freq.sum()
fraud_p_email_freq = fraud_p_email_freq[0:10]

non_fraud_p_email_freq = non_fraudulent_df.P_emaildomain.value_counts().sort_values(ascending=False)
non_fraud_p_email_freq = non_fraud_p_email_freq.rename_axis('p_email').reset_index(name='freq')
non_fraud_p_email_freq['percent'] = non_fraud_p_email_freq['freq'] * 100.0 / non_fraud_p_email_freq.freq.sum()
non_fraud_p_email_freq = non_fraud_p_email_freq[0:10]

combined_p_email_freq = pd.merge(fraud_p_email_freq, non_fraud_p_email_freq, how='outer', on='p_email', suffixes=["_in_fraud", "_in_non_fraud"])
combined_p_email_freq = combined_p_email_freq.fillna(0.0)

combined_p_email_freq[['p_email', 'percent_in_fraud', 'percent_in_non_fraud']].plot.bar(x='p_email')
plt.tight_layout()


# ### Conclusion (P_emaildomain):
# 1.  Almost all the email domains used in non-fraud transactions are popular in fraud transactinos
# 2.  The only difference is yahoo.com, anonymous.com aol.com which are used less frequent in fraud transactions
# 3.  The other key difference is mail.com is explicitly used in fraud transactions

# ### [Part 1] Exploring R_emaildomain (receipient) between fraud and non-fraud transactions

# In[21]:


fraud_r_email_freq =  fraudulent_df.R_emaildomain.value_counts().sort_values(ascending=False)
fraud_r_email_freq = fraud_r_email_freq.rename_axis('r_email').reset_index(name='freq')
fraud_r_email_freq['percent'] = fraud_r_email_freq['freq'] * 100.0 / fraud_r_email_freq.freq.sum()
fraud_r_email_freq = fraud_r_email_freq[0:10]

non_fraud_r_email_freq = non_fraudulent_df.R_emaildomain.value_counts().sort_values(ascending=False)
non_fraud_r_email_freq = non_fraud_r_email_freq.rename_axis('r_email').reset_index(name='freq')
non_fraud_r_email_freq['percent'] = non_fraud_r_email_freq['freq'] * 100.0 / non_fraud_r_email_freq.freq.sum()
non_fraud_r_email_freq = non_fraud_r_email_freq[0:10]

combined_r_email_freq = pd.merge(fraud_r_email_freq, non_fraud_r_email_freq, how='outer', on='r_email', suffixes=["_in_fraud", "_in_non_fraud"])
combined_r_email_freq = combined_r_email_freq.fillna(0.0)

combined_r_email_freq[['r_email', 'percent_in_fraud', 'percent_in_non_fraud']].plot.bar(x='r_email')
plt.tight_layout()


# ### Conclusion (R_emaildomain):
# 1.  gmail.com is proportionally more frequent in fraud transactions than non-fraud transactions
# 2.  outlook.es, mail.com, live.com.mx are explicitly used in fraud transactions

# ### [Part 1] Observing distributions of fields among fraudulent and non-fraudulent transactions ###

# In[24]:


fraud_hists = fraudulent_df.hist()
plt.tight_layout()


# In[25]:


non_fraud_hists = non_fraudulent_df.hist()
plt.tight_layout()


# ## Conclusion ##
# Apart from maximum frequency all histograms depict same kind of spread of values between fraudulent transactions and non-fraudulent transactions. The maximum frequency is different obviously because fraudulent transactions are only small fraction (~ 4%) of the global data-set

# ## Observing max-min-std of the fields among fraud vs non-fraud transactions ##

# In[32]:


fraudulent_df.describe()


# In[33]:


non_fraudulent_df.describe()


# ## Conclusion ##
# 
# 
# 1.   The max value of `TransactionAmt` for fraudulent transactions is 5191, whereas for non-fraudulent transaction it is 31937 (which is roughly 6 times). This suggests that fraudulent transactions are possibly done in smaller denominations may be to avoid flagging or alerting existing Fraud-Detection Systems 
# 2.   The max value of `dist1` for fraudulent transactions is 4942, whereas for non-fraudulent transaction it is 10286 (which is roughly twice). It suggests that either the fraudster did frauds at smaller distances or  fraudster recorded fake details regarding address or distances in their order details of locations they are aware of (possibly closer to their residences)
# 3.   The standard deviation of `addr2` of fraudulent transactions is 4.93, whereas for non-fraudulent transactions is 2.6 (which is roughly half). This indicates that the spread of billing region or billing country details for fraudulent transactions is much higher may be because the fraudster were faking their details
# 
# 

# ## Part 2 - Transaction Frequency

# ### Citation:
# 1. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
# 2. https://stackoverflow.com/questions/21441259/pandas-groupby-range-of-values

# In[34]:


country_code_freq = basic_fields_df['addr2'].value_counts()
country_code_freq = country_code_freq.rename_axis('country_code').reset_index(name='country_transaction_counts')
country_code_freq.head()


# ### [Part-2] Maximum transaction country ###
# We can see in the above step that the country with maximum transactions is the country having country-code, 87 (~ 88% of transactions).
# So in the next step we observe the values of country with code 87. 
# 1. First we will calculate day-level-tags (ith day in the time frame of whole data-set)
# 2. Then we will calculate hour-level-tags (ith hour in the time frame of whole data-set)
# 3. Then we will calculate minute-level-tags(ith minute in the time frame of whole day)
# 4. In the end we will analyse time tag wise distribution for countries with code 87

# In[35]:


max_transaction_dt = 16000000 # max value in data-set
transaction_bin_width = 86400  # no. of seconds in a day
possible_bins = np.arange(0, max_transaction_dt + transaction_bin_width, transaction_bin_width)
row_bin_tags = pd.cut(basic_fields_df["TransactionDT"], bins=possible_bins, labels=np.arange(len(possible_bins) - 1))
basic_fields_df['day_tags'] = row_bin_tags

transaction_bin_width = 3600 # no. of seconds in an hour
possible_bins = np.arange(0, max_transaction_dt + transaction_bin_width, transaction_bin_width)
row_bin_tags = pd.cut(basic_fields_df["TransactionDT"], bins=possible_bins, labels=np.arange(len(possible_bins) - 1))
basic_fields_df['hour_tags'] = row_bin_tags

transaction_bin_width = 60 # no. of seconds in an minute
possible_bins = np.arange(0, max_transaction_dt + transaction_bin_width, transaction_bin_width)
row_bin_tags = pd.cut(basic_fields_df["TransactionDT"], bins=possible_bins, labels=np.arange(len(possible_bins) - 1))
basic_fields_df['minute_tags'] = row_bin_tags

# Converting categorical columns to int
categorical_columns = ['day_tags', 'hour_tags', 'minute_tags']
basic_fields_df[categorical_columns] = basic_fields_df[categorical_columns].apply(lambda x: x.cat.codes)
basic_fields_df['day_tags'] = basic_fields_df['day_tags'].astype(np.int64)
basic_fields_df['hour_tags'] = basic_fields_df['hour_tags'].astype(np.int64)
basic_fields_df['minute_tags'] = basic_fields_df['minute_tags'].astype(np.int64)

# Calculating hour of the day (0 to 23) based on hour_tags
basic_fields_df['hour_of_day'] = basic_fields_df['hour_tags'] % 24
basic_fields_df['minute_of_day'] = basic_fields_df['minute_tags'] % 1440


# In[36]:


most_frequent_country = basic_fields_df[basic_fields_df.addr2 == 87]


# In[37]:


dayWiseTransactionCount = most_frequent_country.groupby("day_tags")["TransactionID"].agg(['count'])
dayWiseTransactionCount.plot.bar()
plt.tight_layout()


# ## Conclusion ##
# As you can see, majority of the transactions were done in the earlier days in the time frame. There are only few outliers (a day or two) in the six month time frame

# In[38]:


hourWiseTransactionCount = most_frequent_country.groupby("hour_of_day")["TransactionID"].agg(['count'])
hourWiseTransactionCount.plot.bar()
plt.tight_layout()


# ## Conclusion ##
# 1. As we can see in the above plot, that a majority of transactions are calculated in hours (hour 0 to hour 2) and in late hours (hour 15 to hour 23).
# 2. This indicates majority of the people who reside in country with code 87, have their waking hours as per base reference, from hour 0 to hour 4 and hour 13 to hour 23. So waking hours of people in country code 87, lies in ranges [13, 23], [0, 4]

# ## Part 3 - Product Code

# In[48]:


# TODO: code to analyze prices for different product codes
values = basic_fields_df.ProductCD.value_counts()
values = values.rename_axis('product').reset_index(name='freq')
values.head()


# In[49]:


sns.countplot(x='ProductCD', data=basic_fields_df)


# In[50]:


product_agg_df = basic_fields_df.groupby(["ProductCD"])["TransactionID", "TransactionAmt"].agg(["min", "max", "count"])
product_agg_df.columns = ["_".join(x) for x in product_agg_df.columns.ravel()] # normalizing hierarchical column names
product_agg_df


# In[51]:


figure, axes = plt.subplots(1,3)

product_agg_df["TransactionID_count"].plot.bar(ax=axes[0])
product_agg_df["TransactionAmt_min"].plot.bar(ax=axes[1])
product_agg_df["TransactionAmt_max"].plot.bar(ax=axes[2])


axes[0].set_title("Transaction counts")
# axes[0].title.set_size(10)
axes[1].set_title("Min Transaction amt")
# axes[1].title.set_size(10)
axes[2].set_title("Max Transaction amt")
# axes[2].title.set_size(10)
# axes[1].set_title("non_fraud")
# plt.subplot(1,0, 0)

plt.tight_layout()


# ## Conclusion ##
# 
# 
# 1.   Product R is the product which has relatively highest minimum transaction amount among its peers. It also has moderate to least amount of purchases. This indicates that is a product which is bought least amount of times as well as whenever it was bought in a minimum amount transaction it would cost more than its peers
# 2.   Product R is most expensive product
# 3.   Product C has more balance stats. Its' transaction counts are second highest (indicating it is a product that people buy most). As well as the minimum transaction amount of Product C is lowest.
# 3.   Product C is cheapest

# Write your answer here

# ## Part 4 - Correlation Coefficient

# Since, the TransactionAmt is a highly skewed signal, we will use natural logarithm of the field to understand it's nature with other signals 

# In[85]:


basic_fields_df["TransactionAmtLog"] = np.log(basic_fields_df["TransactionAmt"])
basic_fields_df["hour_of_day_log"] = np.log(basic_fields_df["hour_of_day"])
basic_fields_df["minute_of_day_log"] = np.log(basic_fields_df["minute_of_day"])


# In[86]:


basic_fields_df["TransactionAmtLog"].hist(bins=100)


# ### Citation ###
# https://stats.stackexchange.com/questions/127121/do-logs-modify-the-correlation-between-two-variables

# In[87]:


# TODO: code to calculate correlation coefficient
amountVsTime = basic_fields_df[["hour_of_day", "hour_of_day_log", "minute_of_day", "minute_of_day_log", "TransactionAmt", "TransactionAmtLog"]].corr()
amountVsTime


# In[88]:


basic_fields_df[["hour_of_day", "hour_of_day_log", "minute_of_day", "minute_of_day_log", "TransactionAmt", "TransactionAmtLog"]].corr(method='spearman')


# In[89]:


sns.heatmap(amountVsTime)


# #### Since, we do not observe any significant correlation between time features and TransactionAmt, we will observe correlation between aggregation and time feature

# In[99]:


hour_of_day_agg = basic_fields_df[["TransactionAmt", "hour_of_day"]].groupby(["hour_of_day"]).agg(["min", "max", "sum"])
hour_of_day_agg = hour_of_day_agg.reset_index()


# In[96]:


hour_of_day_agg.corr()


# In[98]:


sns.heatmap(hour_of_day_agg.corr())


# ### Conclusion
# 1. Moderate correlation between TransactionAmt and hour of day is observed
# 2. Correlation value = 0.642

# ## Part 5 - Interesting Plot

# In[100]:


expenses = basic_fields_df.groupby(['card6'])['TransactionAmt'].agg(['sum'])
expenses = expenses.rename(columns={"sum": "exp"})

expenses['exp_log'] = np.log10(expenses['exp'])

figure, axes = plt.subplots(1,2)
expenses.plot.bar(ax=axes[0], y='exp')
expenses.plot.bar(ax=axes[1], y='exp_log')

axes[0].set_title("normal scale")
axes[1].set_title("logarithmic scale")
plt.tight_layout()


# # Conclusion (Insight) 
# 1.  As you can notice that, most expenses are done through debit card followed by credit card. The least expenses are done through charge card
# #### 2.  This indicates that people prefer to stay under their credit limit, else they will have to penalty interest rates. Possibly fear of penalty drive people to use less or avoid using credit card for major expenses. This reveals risk-averse nature of people
# 3.  Also, the minimal expenses on charge card compared to credit card indicate that people are ready to face interest rates of credit card for late payment but are not ready to face heavy late penalty fees on charge card. A simple search in the domain card types highlights that late penalty feese on charge card are in higher order than on interest on credit card

# Write your answer here

# ## Part 6 - Prediction Model

# In[101]:


# TODO: code for your final model


# ## Checking Duplicates for the significant fields across whole data-set
# Testing whether there are any duplicates in the base data (TransactionID, DeviceType, DeviceInfo, TransactionDT, TransactionAmt, ProductCD, addr1, addr2, dist1, dist2, card4, card6, P_emaildomain, R_emaildomain) of transactions and devices.
# 
# ---
# 
# Found that there are no duplicates since shape on `base_df[base_duplicates]` responds with 0 count against number of rows
# 

# In[102]:


base_duplicates = basic_fields_df.duplicated()
print("Number of duplicates in base fields: " + str(basic_fields_df[base_duplicates].shape[0]))


# ### Loading train and test data in dataframes

# In[103]:


train_identity_df = pd.read_csv("../data/train_identity.csv")
train_transaction_df = pd.read_csv("../data/train_transaction.csv")
test_identify_df = pd.read_csv('../data/test_identity.csv')
test_transaction_df = pd.read_csv('../data/test_transaction.csv')


# ### Missing values in card6/card4 field (credit/debit)
# Since card6 and card4 field null counts are not high, filling it with place-holders of credit/debit with equal probability

# In[104]:


def handle_missing_values_in_card6(input_df, is_test_data):
    print("==================== Solving missing values for card6 ====================")
    modified_df = input_df.copy(deep=True)
    nan_indices = modified_df.index[modified_df.card6.isnull()]
    test_card6_replacements = random.choices(['credit', 'debit'], [1, 1], k=nan_indices.shape[0])
    i = 0
    for index in nan_indices:
        modified_df.iloc[index, modified_df.columns.get_loc('card6')] = test_card6_replacements[i]
        if i % 100 == 0:
            print("card6 filling, done with " + str(i) + " , remaining: " + str(len(nan_indices) - i))
        i += 1
    
    print("==================== Resolved card6 ====================")
    
    return modified_df


# In[105]:


def handle_missing_values_in_card4(input_df, test_data):
    print("==================== Solving missing values for card4 ====================")
    
    modified_df = input_df.copy(deep=True)
    nan_indices = modified_df.index[modified_df.card4.isnull()]
    test_card4_replacements = random.choices(['visa', 'mastercard', 'american express', 'discover'], [1, 1, 1, 1], k=nan_indices.shape[0])
    i = 0
    for index in nan_indices:
        modified_df.iloc[index, modified_df.columns.get_loc('card4')] = test_card4_replacements[i]
        if i % 100 == 0:
            print("Test data card4 filling, done with " + str(i) + " , remaining: " + str(len(nan_indices) - i))
        i += 1
    
    print("==================== Resolved card4 ====================")
    
    return modified_df


# In[106]:


def handle_missing_values_in_addr(input_df):
    print("==================== Solving missing values for addr1/addr2 ====================")
    
    modified_df = input_df.copy(deep=True)
    
    modified_df['addr1'] = modified_df['addr1'].fillna('special_region')
    modified_df['addr2'] = modified_df['addr2'].fillna('special_country')
    
    print("==================== Resolved addr1/addr2 ====================")
    
    return modified_df


# ### Mapping multiple same information to common field and handling missing values in email-related fields
# 1. Multiple same emails belonging to same domain like 'gmail.com' and 'gmail' are incorrectly present in training-dataset
# 2. Imputing missing values of purchaser and receipient email

# In[107]:


def handle_emails(input_df):
    modified_df = input_df.copy(deep=True)
    
    email_domain_map = {
	'gmail.com': ['gmail.com', 'gmail'], 
  	'yahoo.com': ['yahoo.com', 'yahoo.com.mx', 'ymail.com', 'yahoo.de', 'yahoo.fr', 'yahoo.es', 'yahoo.co.uk', 'yahoo.co.jp'] , 
  	'hotmail.com': ['hotmail.com', 'outlook.com', 'live.com.mx', 'hotmail.es', 'msn.com', 'live.com', 'outlook.es', 'hotmail.de', 'hotmail.fr', 'hotmail.co.uk'],
  	'netzero.net': ['netzero.com', 'netzero.net'] ,
  	'icloud.com': ['mac.com', 'icloud.com']
    }

    reverse_email_domain_map = {}
    for root_domain in email_domain_map:
        for domain in email_domain_map[root_domain]:
            reverse_email_domain_map[domain] = root_domain
    # reverse_email_domain_map

    modified_df['P_emaildomain'] = modified_df['P_emaildomain'].fillna('special_purchaser_email')
    modified_df['R_emaildomain'] = modified_df['R_emaildomain'].fillna('special_receipient_email')
  
    modified_df['P_emaildomain'] = modified_df['P_emaildomain'].apply(lambda x: reverse_email_domain_map[x] if x in reverse_email_domain_map else x)
    modified_df['R_emaildomain'] = modified_df['R_emaildomain'].apply(lambda x: reverse_email_domain_map[x] if x in reverse_email_domain_map else x)
    
    return modified_df


# ### Utility to drop columns from data-frame

# In[108]:


def drop_columns(input_df, column_names):
    print("==================== Dropping some columns ====================")
    return input_df.drop(column_names, axis = 1) # non-inplace drop, the input_df will still have its all columns


# ### Utility to encode categorical variables

# In[109]:


def one_hot_encode(df, col_name, col_prefix=''):
    print("==================== Encoding "+ col_name + " ====================")
    
    if len(col_prefix) > 0:
        one_hot_encoding = pd.get_dummies(df[col_name], prefix=col_prefix)
    else:
        one_hot_encoding = pd.get_dummies(df[col_name])
    new_df = df.copy(deep=True)
    new_df = new_df.drop(col_name, axis=1)
    new_df = new_df.join(one_hot_encoding)
    
    return new_df


# In[121]:


def encode_categorical_fields(input_df):
    print("==================== Encoding categorical columns ====================")
    pre_processing_df = input_df.copy(deep=True)
    
    pre_processing_df = one_hot_encode(pre_processing_df, 'card4')
    pre_processing_df = one_hot_encode(pre_processing_df, 'card6')
    pre_processing_df = one_hot_encode(pre_processing_df, 'ProductCD')
    pre_processing_df = one_hot_encode(pre_processing_df, 'DeviceType')
    pre_processing_df = one_hot_encode(pre_processing_df, 'addr1', col_prefix='addr1_')
    pre_processing_df = one_hot_encode(pre_processing_df, 'addr2', col_prefix='addr2_')
    pre_processing_df = one_hot_encode(pre_processing_df, 'P_emaildomain', col_prefix='P_emaildomain_')
    pre_processing_df = one_hot_encode(pre_processing_df, 'R_emaildomain', col_prefix='R_emaildomain_')
    
    print("==================== Done encoding categorical fields ====================")

    return pre_processing_df


# ### Pre-processing training data-set

# ### Currently not addressing missing values in DeviceInfo and distance fields

# In[110]:


def pre_process_df(identity_df, transaction_df):
    print("==================== Pre-Processing begins! ====================")
    
    global_df = pd.merge(left=transaction_df, right=identity_df, left_on=["TransactionID"],
                                       right_on="TransactionID", how="left")
    basic_fields_df = global_df[["TransactionID", "TransactionDT", "TransactionAmt",
                                              "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", 
                                              "addr1", "addr2", "dist1", "dist2", "isFraud", "DeviceType", "DeviceInfo"]]
    modified_df = basic_fields_df.copy(deep=True)
    print("Pre-Processing initial shape: " + str(modified_df.shape))
    
    modified_df = handle_missing_values_in_card6(modified_df, False)
    modified_df = handle_missing_values_in_card4(modified_df, False)
    modified_df = handle_missing_values_in_addr(modified_df)
    modified_df = handle_emails(modified_df)
    
    pre_processing_df = drop_columns(modified_df, ['TransactionID', 'dist1', 'dist2', 'DeviceInfo'])
    pre_processing_df = encode_categorical_fields(pre_processing_df)
    
    print("Pre-Processing final shape: " + str(pre_processing_df.shape))
    print("==================== Done Pre-Processing ====================")
    
    
    return pre_processing_df
    


# In[111]:


def add_missing_dummy_columns(test_df, train_columns ):
    missing_cols = set(train_columns) - set(test_df.columns)
    for c in missing_cols:
        test_df[c] = 0


# ### Citation: To handle categorical values in test data-set
# 1. http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/
# 2. https://stackoverflow.com/questions/41335718/keep-same-dummy-variable-in-training-and-testing-data

# In[114]:


def fix_columns(test_df, train_df):  

    add_missing_dummy_columns(test_df, train_df.columns)

    # make sure we have all the columns we need
    assert(set(train_df.columns) - set( test_df.columns ) == set())

    extra_cols = set(test_df.columns) - set(train_df.columns)
    if extra_cols:
        print("extra columns:", extra_cols)

    test_df = test_df[train_df.columns]
    return test_df


# In[115]:


def pre_process_test_df(identity_df, transaction_df, train_df):
    print("==================== Pre-Processing begins! ====================")
    
    global_df = pd.merge(left=transaction_df, right=identity_df, left_on=["TransactionID"],
                                       right_on="TransactionID", how="left")
    basic_fields_df = global_df[["TransactionID", "TransactionDT", "TransactionAmt",
                                                  "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", 
                                                  "addr1", "addr2", "dist1", "dist2", "DeviceType", "DeviceInfo"]]

    modified_df = basic_fields_df.copy(deep=True)
    print("Pre-Processing initial shape: " + str(modified_df.shape))
    
    modified_df = handle_missing_values_in_card6(modified_df, True)
    modified_df = handle_missing_values_in_card4(modified_df, True)
    modified_df = handle_missing_values_in_addr(modified_df)
    modified_df = handle_emails(modified_df)
    
    pre_processing_df = drop_columns(modified_df, ['TransactionID', 'dist1', 'dist2', 'DeviceInfo'])
    pre_processing_df = encode_categorical_fields(pre_processing_df)
    pre_processing_df = fix_columns(pre_processing_df, train_df)
    
    print("Pre-Processing final shape: " + str(pre_processing_df.shape))
    print("==================== Done Pre-Processing ====================")
    
    
    return pre_processing_df
    


# In[116]:


def split_dataset(df):
    input_dataset_x = df.drop(['isFraud'], axis=1)
    input_dataset_y = df['isFraud']
    return train_test_split(input_dataset_x, input_dataset_y, random_state=31) # returns X_train, X_test, y_train, y_test


# In[117]:


def plot_confusion_matrix(y_test, y_predictions, labels=[1,0]):
    mat = confusion_matrix(y_test, y_predictions, labels)
    print("Confusion Marix: ")
    print(str(mat))
    sns.heatmap(mat, cmap="YlGnBu")
    plt.tight_layout()
    
    return mat


# In[122]:


pre_processed_df = pre_process_df(train_identity_df, train_transaction_df)


# ### Re-sampling
# 1.  Since, the data is highly imbalance, skewed with large number of non-fraud transactions and few fraud transactions
# 2.  We perform over-sampling on fraud-instances

# ### Model
# The best classification model run for above pre-processing steps is Random Forest
# Earlier attempts include: Logistic Regression and Decision Tree

# ### Citation:
# 1. https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets#t5

# In[123]:


# Oversampling with SMOTE
X_train, X_test, y_train, y_test = split_dataset(pre_processed_df)
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

rf_classifier = RandomForestClassifier(random_state=13)
rf_classifier = rf_classifier.fit(X_resampled, y_resampled)


# In[126]:


y_predictions = rf_classifier.predict(X_test)
accuracy_score = rf_classifier.score(X_test, y_test)
confusion_mat = plot_confusion_matrix(y_test, y_predictions, labels=[0,1])
print("Random Forest accuracy: " + str(accuracy_score))
prfs = precision_recall_fscore_support(y_test, y_predictions, average='binary')
print("Precision-Recall-Fscore-Support -> " + str(prfs))


# Write your answer here

# ## Part 7 - Final Result

# Report the rank, score, number of entries, for your highest rank. Include a snapshot of your best score on the leaderboard as confirmation. Be sure to provide a link to your Kaggle profile. Make sure to include a screenshot of your ranking. Make sure your profile includes your face and affiliation with SBU.

# Kaggle Link: https://www.kaggle.com/c/ieee-fraud-detection/submissions

# Highest Rank: 5564

# Score: 0.6891

# Number of entries: 506691

# <img src="RankingImage.png">

# In[ ]:




