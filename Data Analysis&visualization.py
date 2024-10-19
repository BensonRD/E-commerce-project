import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
#%%
#設定字體、大小與負號
rcParams['font.family'] = 'Microsoft JhengHei'
rcParams['axes.unicode_minus'] = False

#%%
df = pd.read_csv('C:/Users/user/Desktop/data/cleaned_data.csv')
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])

'''''''''''''''性別分布圖'''''''''''''''
gender_distribution_counts = df['Gender'].value_counts()
labels = ['男','女']
colors = ['#0080ff','#ff85cb']
plt.pie(gender_distribution_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.setp(plt.gca().texts, fontsize=12)
plt.title('性別分布圖', fontsize=14)
plt.savefig('性別分布圖', bbox_inches='tight')
plt.show()


#%%
'''''''''''''''區分年齡組(cut)並畫圖'''''''''''''''
age_bins = [0,9,19,29,39,49,59,69,np.inf]
age_labels = ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70+']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
age_gender_counts = df.groupby(['Age_Group', 'Gender']).size().unstack(fill_value=0)
men_counts = age_gender_counts['Male']
women_counts = age_gender_counts['Female']
index = np.arange(len(age_gender_counts.index))
plt.bar(index, men_counts, 0.35, label='男', color='#0080ff')
plt.bar(index + 0.35, women_counts, 0.35, label='女', color='#ff85cb')
plt.xticks(index + 0.35 / 2, age_gender_counts.index)
plt.xlabel('年齡區間')
plt.ylabel('人\n數',rotation=0,labelpad=15)
plt.title('各年齡區間與性別分布圖', fontsize=14)
plt.savefig('各年齡區間與性別分布圖', bbox_inches='tight')
plt.legend()


#%%
'''''''''''''''RFM分析'''''''''''''''
# df["Purchase_Date"].max() --> Timestamp('2023-09-13 18:42:49') --> 所有遊客最後購買日
analysis_date = pd.to_datetime('2023-09-15')  
recency = df.groupby('Customer_ID')['Purchase_Date'].max()
recency = (analysis_date - recency).dt.days
frequency = df.groupby('Customer_ID').size()
monetary = df.groupby('Customer_ID')['Total_Price'].sum()
rfm = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary})

# Kmeans
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

'''''''''''''''輪廓係數找出最佳分群數'''''''''''''''
'''
silhouette_scores = []
for k in range(2,11):  # 從2開始
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))
plt.figure(figsize=(10, 6))
plt.plot(range(2,11), silhouette_scores, marker='o')
plt.title('輪廓係數圖')
plt.xlabel('最佳群數(K)')
plt.ylabel('輪廓係數')
plt.savefig('kmeans輪廓係數圖',bbox_inches = 'tight')
plt.show() # -->由圖可知，分為四群為較適方式
'''

kmeans = KMeans(n_clusters=5, random_state=42,n_init=10)
labels = kmeans.fit_predict(rfm_scaled)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
rfm['Customer_ID'] = rfm.index
rfm.to_csv('rfm_data.csv', index=False)
# 使用 seaborn 的 pairplot
sns.pairplot(rfm, hue='Cluster', vars=['Recency', 'Monetary', 'Frequency'])
plt.savefig('RFM分析圖', bbox_inches='tight')
plt.show()


# 群組 0：新顧客，最近有購買z，但購買頻率和消費金額低。
# 群組 1：中等價值顧客，購買頻率和消費金額適中。
# 群組 2：高價值顧客，購買頻率和消費金額均高。
# 群組 3：低價值顧客(流失客)，購買頻率和消費金額均低，且最近購買間隔長。
#%%
'''''''''''''''產品類別分析'''''''''''''''
product_category = df['Product_Category'].value_counts().reset_index()
product_category.columns = ['Product_Category', 'Count']
product_category_sales = df.groupby('Product_Category')['Total_Price'].sum().reset_index()
product_category_sales['Total_Price'] = product_category_sales['Total_Price'] / 10000000
product_category_sales.columns = ['Product_Category', 'Product_Category_sales']
product_category = product_category.merge(product_category_sales, on='Product_Category', how='left')

fig, ax1 = plt.subplots()
ax1.set_ylim(4.5, 4.9)
ax1.set_xlabel('產品分類',labelpad=5)
ax1.set_ylabel('銷\n售\n總\n額',rotation=0,labelpad=15, color='tab:blue')
ax1.bar(product_category['Product_Category'], product_category['Product_Category_sales'], color='tab:blue', alpha=0.6, label='銷售總額')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('銷\n售\n量',rotation=0,labelpad=15 ,color='tab:orange')
ax2.plot(product_category['Product_Category'], product_category['Count'], color='tab:orange', marker='o', label='銷售量')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('各總類的銷售輛及銷售總額')
plt.savefig('各總類的銷售輛及銷售總額',bbox_inches = 'tight')
fig.tight_layout()  # 確保圖表不會被遮擋
plt.show()


age_gender_category_quantity = df.groupby(['Age_Group','Gender','Product_Category'])['Quantity'].sum().unstack(fill_value=0)
age_gender_category_price = df.groupby(['Age_Group','Gender','Product_Category'])['Total_Price'].sum().unstack(fill_value=0)


fig, ax = plt.subplots(figsize=(14,10))
age_gender_category_quantity.plot(kind='bar', stacked=True, ax=ax)

ax.set_ylim(0, age_gender_category_quantity.values.max() * 4.5)  

plt.title('各性別及各年齡層之銷售量', fontsize=16)
plt.xlabel('年齡層與性別', fontsize=12,labelpad=5)
plt.ylabel('銷\n售\n量',rotation=0, fontsize=12,labelpad=5)
plt.xticks(rotation=45)
plt.tight_layout()  
plt.legend(title='產品類別', loc='upper left')  
plt.savefig('各性別及各年齡層之銷售量',bbox_inches = 'tight')
plt.show()

fig, ax = plt.subplots(figsize=(14,10))
age_gender_category_price.plot(kind='bar', stacked=True, ax=ax)

ax.set_ylim(0, age_gender_category_price.values.max() * 4.5)  

plt.title('各性別及各年齡層之銷售額', fontsize=16)
plt.xlabel('年齡層與性別', fontsize=12,labelpad=5)
plt.ylabel('銷\n售\n額',rotation=0, fontsize=12,labelpad=5)
plt.xticks(rotation=45)
plt.tight_layout()  
plt.legend(title='產品類別', loc='upper left')  
plt.savefig('各性別及各年齡層之銷售額',bbox_inches = 'tight')
plt.show()







