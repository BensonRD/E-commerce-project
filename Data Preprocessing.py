import pandas as pd


df = pd.read_csv('C:/Users/user/Desktop/data/ecommerce_customer_data_large.csv')
# df.info()
# df.describe(include='O')
# df.isnull().sum()

df.drop_duplicates(inplace=True) # 清除重複值
df['Returns'] = df['Returns'].fillna(0).astype(int) # 空值填入
df.columns = df.columns.str.replace(' ', '_') # 更改欄位名稱

'''''''''''''''更改日期格式並提取年度及月份'''''''''''''''
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
df['Purchase_Year'] = df['Purchase_Date'].dt.year
df['Purchase_Month'] = df['Purchase_Date'].dt.month
df.insert(df.columns.get_loc('Purchase_Date') + 1, 'Purchase_Year', df.pop('Purchase_Year'))
df.insert(df.columns.get_loc('Purchase_Date') + 2, 'Purchase_Month', df.pop('Purchase_Month'))

'''''''''''''''修正購買總額'''''''''''''''
df['Total_Price'] = df['Product_Price'] * df['Quantity']
df.insert(df.columns.get_loc('Total_Purchase_Amount') + 1, 'Total_Price', df.pop('Total_Price'))
df = df.drop(['Total_Purchase_Amount'], axis=1)

'''''''''''''''刪除不要欄位'''''''''''''''
df.drop(['Customer_Name'], axis=1, inplace=True)
df.drop(['Customer_Age'], axis=1, inplace=True)

df.to_csv('cleaned_data.csv', index=False)