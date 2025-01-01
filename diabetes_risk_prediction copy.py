#data source
#kaggle pima indians diabetes database
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

"""
ML FLOW CHART
1.import libraries
2.import data and EDA(exploratory data analysis)
3.outlier detection
4.train test split
5.standardization
6.model training and evaluation
7.hyperparameter tuning
8.model testing with real examples
"""

#1. IMPORT LIBRARIES   (sklearn is the main library for ML)

import pandas as pd #data science library
import numpy as np #numeric python library
import matplotlib.pyplot as plt  #for data visualization
import seaborn as sns  #for data visualization

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier

import warnings    #a modification made to avoid confusion regarding warnings
warnings.filterwarnings("ignore")

#2. IMPORT DATA AND EDA

#loading data
df = pd.read_csv("diabetes.csv")
df_name = df.columns

df.info()

describe=df.describe()

plt.figure()
sns.pairplot(df, hue = "Outcome")
plt.show()           #data visualization

#correlation evaluation
def plot_correlation_heatmap(df):
    corr_matrix = df.corr()
    
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap = "coolwarm", fmt = ".3f", linewidths=0.5)
    plt.title("correlation of Features")
    plt.show()

plot_correlation_heatmap(df)
#age,pregnancies and GLUCOSE!


#3. OUTLIER DETECTION

def detect_outliers_iqr(df):
    outlier_indices=[]
    outliers_df=pd.DataFrame()
    
    for col in df.select_dtypes(include=np.number).columns:
        
        Q1 = df[col].quantile(0.25) #first quartile
        Q3 = df[col].quantile(0.75) #third queartile
        
        IQR = Q3 - Q1  #interquartile range
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outlier_indices.extend(outliers_in_col.index)
        
        outliers_df = pd.concat([outliers_df, outliers_in_col], axis = 0)
        
    #remove duplicate indices
    outlier_indices = list(set(outlier_indices))
    
    #remove duplicate reows in the outliers dataframe
    outliers_df = outliers_df.drop_duplicates()
    
    return outliers_df, outlier_indices

outliers_df, outlier_indices = detect_outliers_iqr(df)

#remove outliers from the dataframe

df_cleaned = df.drop(outlier_indices).reset_index(drop=True )
         # 129 outliers!


#4. TRAIN TEST SPLIT

X = df_cleaned.drop(["Outcome"], axis = 1)
y = df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#5. Standardization

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#6. MODEL TRAINING AND EVALUATION

"""
LogisticRegression
DecisionTreeClassifier
KNeighborsClassifier
GaussianNB
SVC
AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
"""

def getBasedModel():
    basedModels = []
    basedModels.append(("LR", LogisticRegression()))
    basedModels.append(("DT", DecisionTreeClassifier()))
    basedModels.append(("KNN", KNeighborsClassifier()))
    basedModels.append(("NB", GaussianNB()))
    basedModels.append(("SVM", SVC()))
    basedModels.append(("AdaB", AdaBoostClassifier()))
    basedModels.append(("GBM", GradientBoostingClassifier()))
    basedModels.append(("RF", RandomForestClassifier()))                 

    return basedModels

def baseModelTraining(X_train, y_train, models):
    
    results= []
    names= []
    
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name}: accuracy: {cv_results.mean()}, std: {cv_results.std()}")
        
    return names, results

def plot_box(names, results):
    df = pd.DataFrame({names[i]: results[i] for i in range(len(names))})
    plt.figure(figsize = (12,8))
    sns.boxplot(data=df)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
    
models = getBasedModel()
names, results = baseModelTraining(X_train, y_train, models)
plot_box(names, results)

""" 
LR: accuracy: 0.782845744680851, std: 0.06205751329302423
most successful model
"""


#7. HYPERPARAMETER TUNING

# DT hyperparameter set

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10,20,30,40,50],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4]
    }

dt = DecisionTreeClassifier()

#grid search cv
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring="accuracy")

#training
grid_search.fit(X_train, y_train)

print("BEST PARAMS: ", grid_search.best_params_)

best_dt_model = grid_search.best_estimator_

y_pred = best_dt_model.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
"""
[[89 20]
 [28 23]]
"""
print("classification_report")
print(classification_report(y_test, y_pred))


#8. MODEL TESTING WITH REAL DATA

new_data=np.array([[6,149,72,35,0,34.6,0.627,51]])
new_prediction=best_dt_model.predict(new_data)

print("New prediction: ",new_prediction)
# New prediction:  [0]


