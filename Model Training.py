from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb
#%%
'''''''''''''''流失率分析(xgboost)'''''''''''''''
df = pd.read_csv('C:/Users/user/Desktop/data/cleaned_data.csv')
rfm = pd.read_csv('C:/Users/user/Desktop/data/rfm_data.csv')
age_bins = [0,9,19,29,39,49,59,69,np.inf]
age_labels = ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70+']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
le1 = LabelEncoder()
df['Category_Numeric'] = le1.fit_transform(df['Product_Category'])
le2 = LabelEncoder()
df['Age_Numeric'] = le2.fit_transform(df['Age_Group'])
le3 = LabelEncoder()
df['Gender_Numeric'] = le3.fit_transform(df['Gender'])

rfm = pd.merge(rfm, df[['Customer_ID','Age_Numeric','Gender_Numeric','Category_Numeric','Churn']], on='Customer_ID', how='left')
rfm1 = rfm.drop_duplicates(keep='first')
# X = rfm[['Recency', 'Frequency', 'Monetary','Age_Numeric','Gender_Numeric','Category_Numeric']]
X = rfm1[['Recency', 'Frequency', 'Monetary','Age_Numeric']]
y = rfm1['Churn']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
weight_0 = (142623 + 28500) / (2 * 142623)  # 0類的權重
weight_1 = (142623 + 28500) / (2 * 28500)   # 1類的權重
rfm_model = RandomForestClassifier(random_state=42, class_weight={0: weight_0, 1: weight_1})

rfm_model.fit(X_train, y_train)
feature_importances = rfm_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()

y_pred = rfm_model.predict(X_test)
print("混淆矩陣:")
print(confusion_matrix(y_test, y_pred))
print("\n分類報告:")
print(classification_report(y_test, y_pred))

y_prob = rfm_model.predict_proba(X_test)[:, 1]  # 獲取類別為 1 的預測機率
auc_roc = roc_auc_score(y_test, y_prob)
print(f"\nRandomForest AUC-ROC 分數: {auc_roc:.4f}")

