import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from src.preperation import load_rfm_data, cat_summary, num_summary

# to display all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = load_rfm_data()

# All invoices starts with C which means return, exclude them.
df = df[~df['Invoice'].str.contains('C', na=False)]
# All customer ID null rows excluded
df = df[df['Customer ID'].notnull()]
df['Customer ID'] = df['Customer ID'].astype(int)

# Recency calculations
last_date = pd.to_datetime(df['InvoiceDate'].max()).date()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
recency_df = df.groupby("Customer ID").agg({"InvoiceDate": lambda x: (last_date - x.max()).days})
recency_df.rename(columns={'InvoiceDate': 'Recency'}, inplace='True')

# Frequency calculations
freq_df = df.groupby(['Customer ID']).agg({'Invoice': 'nunique'})
freq_df.rename(columns={'Invoice': 'Frequency'}, inplace=True)

# Monetary calculations
monetary_df = df.groupby(['Customer ID']).agg({'TotalPrice': 'sum'})
monetary_df.rename(columns={'TotalPrice': 'Monetary'}, inplace=True)

rfm = pd.concat([recency_df, freq_df, monetary_df], axis=1)

rfm['RecencyScore'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['FrequencyScore'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm['rfm_score'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str)

# create a segmentatiton map
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Loose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}

rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)

rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","count"])

# list 'New Customers' segment
new_customer_df = pd.DataFrame()
new_customer_df['CustomerIds'] = rfm[rfm["Segment"] == "New Customers"].index

new_customer_df.to_csv('/Users/basrisahin/Documents/GitHub/data-science/RMF/data/processed/new_customers.csv', index=False)