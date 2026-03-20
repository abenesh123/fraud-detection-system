import numpy as np
import pandas as pd
import mysql.connector

conn=mysql.connector.connect(
    host="localhost",
    user="root",
    password="abineshmysql@123",
    database="fraud_detection_db"
)

query="SELECT * FROM model_dataset"
df=pd.read_sql(query,conn)

print("Data succesfully imported from mysql!")
# print(df)


df.shape

df.columns
#total no. of columns are 31

df.info()

df.describe(include="all")
#amount column have very large value

df.isnull().sum()
#no null values present in this dataset

df.duplicated().sum()

df["class"].unique()

df["class"].value_counts()

df["class"].value_counts(normalize=True)*100

print("Basic checks completed")

#libraries for plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
sns.countplot(x="class",data=df,width=0.2)
plt.title("fraud vs normal transaction")
plt.show()

#count of normal(99.93%) and fraud transaction(0.17%)

plt.pie(df["class"].value_counts(),labels=["no","yes"],autopct="%0.2f%%")
plt.title("fraud vs normal")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df["amount"],bins=50)
plt.title("transaction amount distribution")
plt.show()
#distribution of transaction amount is high in class values(0,1)

q1=df["amount"].quantile(0.25)
q3=df["amount"].quantile(0.75)

iqr=q3-q1

lower=q1-1.5*iqr
upper=q3+1.5*iqr

print("lower :",lower,"upper :",upper)

outliers=df[(df["amount"]<lower) | (df["amount"]>upper)]
print("number of outliers :",len(outliers))

plt.figure(figsize=(10,5))
sns.boxplot(x="class",y="amount",data=df)
plt.title("amount distribution between class")
plt.show()

df["hour"]=(df["time"]%86400)/3600
df["hour"]
#here time is given in second we have to convert into hour for that we are dividing with 3600 because 1 hour have 3600seconds

df
#we got hour column in the dataset

plt.figure(figsize=(10,5))
sns.histplot(df["hour"],bins=24)
plt.title("transaction hour distribution")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x="class",y="hour",data=df)
plt.show()
#there is no outliers in class,hour columns

plt.figure(figsize=(30,10))
corr=df.corr()
sns.heatmap(corr,cmap="coolwarm",annot=True)
plt.show()
#correlation of each columns
#time is highly correlated with hour,columns v17,v14,v12,v10 are strongly correlated with class this is fraud indicaters

col=df[['V10', 'V12', 'V14','V17',"class"]]
corr2=col.corr()
plt.figure(figsize=(10,5))
sns.heatmap(corr2,cmap="coolwarm",annot=True)
plt.show()
#this four columns are strongly correlated with class columns

plt.figure(figsize=(10,5))
sns.kdeplot(df[df["class"]==0]["V14"],label="normal",shade=True)
sns.kdeplot(df[df["class"]==1]["V14"],label="fraud",shade=True)
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
sns.kdeplot(df[df["class"]==0]["V17"],label="normal",shade=True)
sns.kdeplot(df[df["class"]==1]["V17"],label="fraud",shade=True)
plt.legend()
plt.show()

fraud=df[df["class"]==1]
normal=df[df["class"]==0].sample(2000)
balanced_df=pd.concat([fraud,normal])
sns.pairplot(balanced_df,hue="class")
plt.show()

sns.histplot(x="amount",hue="class",data=df,bins=50,log_scale=True)
plt.title("transaction amount distribution fraud vs normal")
plt.show()

fraud_hour_rate=df.groupby("hour")["class"].mean()
fraud_hour_rate

plt.figure(figsize=(20,5))
fraud_hour_rate.plot()
plt.xlabel("hour")
plt.ylabel("fraud rate")
plt.title("fraud rate by hour")
plt.show()

features=["amount","V10","V12","V14","V17"]
for col in features:
  plt.figure(figsize=(10,5))
  sns.histplot(df[col],bins=50)
  plt.title(f"distribution of {col}")
  plt.show()

features=["amount","V10","V12","V14","V17"]
corr_features=df[features].corr()
plt.figure(figsize=(10,5))
sns.heatmap(corr_features,cmap="coolwarm",annot=True)
plt.show()

print("Plottings are completed")

"""observation and result of EDA:
1.dataset is highly imbalanced which is fraud transaction is only (0.17%) in the whole dataset
2.no null values present in the dataset
3.most transaction amounts are low and few transaction amounts were very high comparatively
4.new column hour is added to the dataset because,time given in second so that time is modules by 86400(seconds in one day) and then divided with 3600(seconds in 1 hour)
5.certain columns like(v10,v12,v14,v17) are strongly correlated with class column and time is also strongly correlated with hour

"""

x=df.drop("class",axis=1)
y=df["class"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,stratify=y,random_state=42)

y_train.value_counts()

print("Spliting completed")


from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
LR_pipeline=Pipeline([
    ("smote", (SMOTE(random_state=42))),
    ("scaler",StandardScaler()),
    ("model",LogisticRegression(max_iter=10000))
])
LR_pipeline.fit(x_train,y_train)

y_pred_LR=LR_pipeline.predict(x_test)
y_pred_LR

y_prob_LR=LR_pipeline.predict_proba(x_test)[:,1]
y_prob_LR

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_LR))

from sklearn.metrics import precision_score,recall_score,f1_score
LR_precision=precision_score(y_test,y_pred_LR)
LR_recall=recall_score(y_test,y_pred_LR)
LR_f1=f1_score(y_test,y_pred_LR)
print("precision :",LR_precision)
print("recall :",LR_recall)
print("f1 :",LR_f1)

print("Logistic Regression model completed")

from sklearn.ensemble import RandomForestClassifier
RF_pipeline=Pipeline([
    ("smote",(SMOTE(random_state=42))),
    ("model",(RandomForestClassifier(n_estimators=300,max_depth=15,random_state=42)))
])
RF_pipeline.fit(x_train,y_train)

y_pred_RF=RF_pipeline.predict(x_test)
y_pred_RF

y_prob_RF=RF_pipeline.predict_proba(x_test)[:,1]
y_prob_RF

print(classification_report(y_test,y_pred_RF))

RF_precision=precision_score(y_test,y_pred_RF)
RF_recall=recall_score(y_test,y_pred_RF)
RF_f1=f1_score(y_test,y_pred_RF)
print("precision :",RF_precision)
print("recall :",RF_recall)
print("f1 :",RF_f1)



import shap 
RF_explainer=shap.TreeExplainer(RF_pipeline.named_steps["model"])
RF_shap_values=RF_explainer.shap_values(x_test)
shap.summary_plot(RF_shap_values,x_test)



print("Random forest model completed")

from sklearn.ensemble import GradientBoostingClassifier
GB_pipeline=Pipeline([
    ("smote",(SMOTE(random_state=42))),
    ("model",(GradientBoostingClassifier(random_state=42)))
])
GB_pipeline.fit(x_train,y_train)

y_pred_GB=GB_pipeline.predict(x_test)
y_pred_GB

y_prob_GB=GB_pipeline.predict_proba(x_test)[0:,1]
y_prob_GB

GB_precision=precision_score(y_test,y_pred_GB)
GB_recall=recall_score(y_test,y_pred_GB)
GB_f1=f1_score(y_test,y_pred_GB)
print("precision :",GB_precision)
print("recall :",GB_recall)
print("f1 :",GB_f1)

feature_importance_GB=pd.DataFrame({
    "feature":x_train.columns,
    "importance":GB_pipeline.named_steps["model"].feature_importances_
})
feature_importance_GB=feature_importance_GB.sort_values(by="importance",ascending=False)
feature_importance_GB

plt.figure(figsize=(20,5))
sns.barplot(data=feature_importance_GB,x="feature",y="importance")
plt.title("feature importance GB")
plt.show()


print("Gradient boost model completed")


from xgboost import XGBClassifier

XGB_pipeline=Pipeline([
    ("smote",(SMOTE(random_state=42))),
    ("model",XGBClassifier(use_label_encoder=False,eval_metric="logloss"))])

XGB_pipeline.fit(x_train,y_train)

y_pred_XGB=XGB_pipeline.predict(x_test)
y_pred_XGB

y_prob_XGB=XGB_pipeline.predict_proba(x_test)[:,1]
y_prob_XGB

XGB_precision=precision_score(y_test,y_pred_XGB)
XGB_recall=recall_score(y_test,y_pred_XGB)
XGB_f1=f1_score(y_test,y_pred_XGB)
print("precision :",XGB_precision)
print("recall :",XGB_recall)
print("f1 :",XGB_f1)

import pandas as pd
feature_importance_XGB=pd.DataFrame({
    "feature":x_train.columns,
    "importance":XGB_pipeline.named_steps["model"].feature_importances_
})
feature_importance=feature_importance_XGB.sort_values(by="importance",ascending=False)
feature_importance_XGB

plt.figure(figsize=(20,5))
sns.barplot(data=feature_importance,x="feature",y="importance")
plt.title("feature importance XGB")
plt.show()



XGB_explainer=shap.TreeExplainer(XGB_pipeline.named_steps["model"])
XGB_shap_values=XGB_explainer.shap_values(x_test)
shap.summary_plot(XGB_shap_values,x_test)


print("XGB model completed")

from sklearn.model_selection import RandomizedSearchCV
param_grid={
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.2],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]

}
random_search=RandomizedSearchCV(
    XGB_pipeline,
    param_distributions=param_grid,
    scoring="recall",
    n_iter=5,
    cv=3,
    verbose=1
)
random_search.fit(x_train,y_train)
best_model=random_search.best_estimator_
best_model

print("random search cv model completed")


import lightgbm as lgb
from lightgbm import LGBMClassifier
LGBM_pipeline=Pipeline([
    ("smote",(SMOTE(random_state=42))),
    ("model",LGBMClassifier(random_state=42))
])
LGBM_pipeline.fit(x_train,y_train)

y_pred_LGBM=LGBM_pipeline.predict(x_test)
y_pred_LGBM

y_prob_LGBM=LGBM_pipeline.predict_proba(x_test)[:,1]
y_prob_LGBM

LGBM_precision=precision_score(y_test,y_pred_LGBM)
LGBM_recall=recall_score(y_test,y_pred_LGBM)
LGBM_f1=f1_score(y_test,y_pred_LGBM)
print("precision :",LGBM_precision)
print("recall :",LGBM_recall)
print("f1 :",LGBM_f1)

feature_importance_LGBM=pd.DataFrame({
    "feature":x_train.columns,
    "importance":LGBM_pipeline.named_steps["model"].feature_importances_
})
feature_importance_LGBM=feature_importance_LGBM.sort_values(by="importance",ascending=False)
feature_importance_LGBM

plt.figure(figsize=(20,5))
sns.barplot(data=feature_importance_LGBM,x="feature",y="importance")
plt.title("feature importance LGBM")
plt.show()


LGBM_explainer=shap.TreeExplainer(LGBM_pipeline.named_steps["model"])
LGBM_shap_values=LGBM_explainer.shap_values(x_test)
shap.summary_plot(LGBM_shap_values,x_test)


print("LGB model completed")

from sklearn.ensemble import IsolationForest
IF_pipeline=Pipeline([
    ("smote",(SMOTE(random_state=42))),
    ("model",(IsolationForest(n_estimators=200,contamination=0.017,random_state=42)))
])
IF_pipeline.fit(x_train,y_train)

y_pred_IF=IF_pipeline.predict(x_test)
y_pred_IF=np.where(y_pred_IF==-1,1,0)
y_pred_IF

y_prob_IF = -IF_pipeline.named_steps["model"].decision_function(x_test)
y_prob_IF

print(classification_report(y_test,y_pred_IF))

IF_precision=precision_score(y_test,y_pred_IF)
IF_recall=recall_score(y_test,y_pred_IF)
IF_f1=f1_score(y_test,y_pred_IF)
print("precision :",IF_precision)
print("recall :",IF_recall)
print("f1 :",IF_f1)


print("IF model completed")

from sklearn.metrics import roc_auc_score
LR_roc=roc_auc_score(y_test,y_prob_LR)
RF_roc=roc_auc_score(y_test,y_prob_RF)
GB_roc=roc_auc_score(y_test,y_prob_GB)
XGB_roc=roc_auc_score(y_test,y_prob_XGB)
LGBM_roc=roc_auc_score(y_test,y_prob_LGBM)
IF_roc=roc_auc_score(y_test,y_prob_IF)
print("LR_roc :",LR_roc)
print("RF_roc :",RF_roc)
print("GB_roc :",GB_roc)
print("XGB_roc :",XGB_roc)
print("LGBM_roc :",LGBM_roc)
print("IF_roc :",IF_roc)

from sklearn.metrics import roc_curve,auc
fpr,tpr,thershold=roc_curve(y_test,y_prob_LR)
roc_auc_LR=auc(fpr,tpr)

plt.plot(fpr,tpr,label=f"LR AUC : {roc_auc_LR}")
plt.plot([0,1],[0,1],color="red",linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve LR")
plt.legend()
plt.show()

fpr,tpr,thershold=roc_curve(y_test,y_prob_RF)
roc_auc_RF=auc(fpr,tpr)

plt.plot(fpr,tpr,label=f"RF AUC : {roc_auc_RF}")
plt.plot([0,1],[0,1],color="red",linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve RF")
plt.legend()
plt.show()

fpr,tpr,thershold=roc_curve(y_test,y_prob_GB)
roc_auc_GB=auc(fpr,tpr)

plt.plot(fpr,tpr,label=f"GB AUC : {roc_auc_GB}")
plt.plot([0,1],[0,1],color="red",linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve GB")
plt.legend()
plt.show()

fpr,tpr,thershold=roc_curve(y_test,y_prob_XGB)
roc_auc_XGB=auc(fpr,tpr)

plt.plot(fpr,tpr,label=f"XGB AUC : {roc_auc_XGB}")
plt.plot([0,1],[0,1],color="red",linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve XGB")
plt.legend()
plt.show()

fpr,tpr,thershold=roc_curve(y_test,y_prob_LGBM)
roc_auc_LGBM=auc(fpr,tpr)

plt.plot(fpr,tpr,label=f"LGBM AUC : {roc_auc_LGBM}")
plt.plot([0,1],[0,1],color="red",linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve LGBM")
plt.legend()
plt.show()

fpr,tpr,thershold=roc_curve(y_test,y_prob_IF)
roc_auc_IF=auc(fpr,tpr)

plt.plot(fpr,tpr,label=f"IF AUC : {roc_auc_IF}")
plt.plot([0,1],[0,1],color="red",linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve IF")
plt.legend()
plt.show()

comparision_table_metrics=pd.DataFrame({
    "model":["LR","RF","GB","XGB","LGBM","IF"],
    "precision":[LR_precision,RF_precision,GB_precision,XGB_precision,LGBM_precision,IF_precision],
    "recall":[LR_recall,RF_recall,GB_recall,XGB_recall,LGBM_recall,IF_recall],
    "f1":[LR_f1,RF_f1,GB_f1,XGB_f1,LGBM_f1,IF_f1],
    "roc":[LR_roc,RF_roc,GB_roc,XGB_roc,LGBM_roc,IF_roc],
    "auc":[roc_auc_LR,roc_auc_RF,roc_auc_GB,roc_auc_XGB,roc_auc_LGBM,roc_auc_IF]
})
comparision_table_metrics

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
precision,recall,thershold=precision_recall_curve(y_test,y_prob_LR)
plt.plot(recall,precision,label="LR")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

precision,recall,thershold=precision_recall_curve(y_test,y_prob_RF)
plt.plot(recall,precision,label="RF")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

precision,recall,thershold=precision_recall_curve(y_test,y_prob_GB)
plt.plot(recall,precision,label="GB")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

precision,recall,thershold=precision_recall_curve(y_test,y_prob_XGB)
plt.plot(recall,precision,label="XGB")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

precision,recall,thershold=precision_recall_curve(y_test,y_prob_LGBM)
plt.plot(recall,precision,label="LGBM")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

precision,recall,thershold=precision_recall_curve(y_test,y_prob_IF)
plt.plot(recall,precision,label="IF")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
y_pred_best=best_model.predict(x_test)
cm=confusion_matrix(y_test,y_pred_best)
display=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["normal","fraud"])
plt.figure(figsize=(10,5))
display.plot(cmap="Blues",values_format="d")
plt.title('Final Model: Fraud Detection Performance')
plt.show()



import joblib
joblib.dump(best_model, "fraud_model.pkl")
print("Model saved successfully")
