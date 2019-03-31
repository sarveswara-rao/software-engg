import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold

print("this is a part of se project")
# = pd.read_csv('D:\\bda_project\\Final\\Project1\\ibm_churn_prediction.csv')
np.random.seed(0)
missing = []
for i in range(7043):
    try:
        np.float(df['TotalCharges'][i])
    except:
        missing.append(i)
df = df.drop(missing)
df = df.drop(columns = 'customerID')
    
df['TotalCharges'] = df['TotalCharges'].astype(np.float64)

X = df.iloc[:, 0:19]    
y = df.iloc[:, 19]
y = pd.DataFrame(data = y, columns={'Churn'})
    
    # Encoding of categorical values to numerical values
y['Churn'] = LabelEncoder().fit_transform(y['Churn'])
X['gender'] = LabelEncoder().fit_transform(X['gender'])
X['Partner'] = LabelEncoder().fit_transform(X['Partner'])
X['Dependents'] = LabelEncoder().fit_transform(X['Dependents'])
X['PhoneService'] = LabelEncoder().fit_transform(X['PhoneService'])
X['PaperlessBilling'] = LabelEncoder().fit_transform(X['PaperlessBilling'])

features = ['MultipleLines','InternetService','OnlineSecurity', 
            'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies','Contract', 'PaymentMethod']
X = pd.get_dummies(X, columns = features)

col_names = ['tenure', 'MonthlyCharges', 'TotalCharges']
features = X[col_names]
features = StandardScaler().fit_transform(features)
X[col_names] = features
y = np.ravel(y)

from sklearn.feature_selection import SelectKBest, f_classif
test = SelectKBest(score_func = f_classif, k = 8)
X_test = test.fit_transform(X, y)
X = pd.DataFrame(X_test)

# #############################################################################
# Classification and ROC analysis

cv = StratifiedKFold(n_splits=10)

#  CART----------------------------------------------
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth = 6, criterion = 'entropy', min_samples_split = 50)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
accuracy = []
from sklearn.metrics import accuracy_score, classification_report
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    y_pred = classifier.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    accuracy.append(accuracy_score(y[test], y_pred))
    print(classification_report(y[test], y_pred))
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i += 1
a = 0
for i in accuracy:
    a += i/10
print("Avg. accuracy:",a)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='red',
         label=r'CART (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

# KNN-------------------------------------
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 30)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
accuracy = []
from sklearn.metrics import accuracy_score, classification_report
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    y_pred = classifier.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    accuracy.append(accuracy_score(y[test], y_pred))
    print(classification_report(y[test], y_pred))
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i += 1
a = 0
for i in accuracy:
    a += i/10
print("Avg. accuracy:",a)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='black',
         label=r'KNeighbors (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)


# NB-------------------------------------
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
accuracy = []
from sklearn.metrics import accuracy_score, classification_report
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    y_pred = classifier.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    accuracy.append(accuracy_score(y[test], y_pred))
    print(classification_report(y[test], y_pred))
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i += 1
a = 0
for i in accuracy:
    a += i/10
print("Avg. accuracy:",a)


mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='yellow',
         label=r'Naive_bayes(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

# LR-------------------------------------
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
accuracy = []
from sklearn.metrics import accuracy_score, classification_report
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    y_pred = classifier.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    accuracy.append(accuracy_score(y[test], y_pred))
    print(classification_report(y[test], y_pred))
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i += 1
a = 0
for i in accuracy:
    a += i/10
print("Avg. accuracy:",a)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='green',
         label=r'Logistic Regression (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

# SVM-------------------------------------
from sklearn.svm import SVC
classifier = SVC(C=0.1, kernel = 'linear', probability = True)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
accuracy = []
from sklearn.metrics import accuracy_score, classification_report
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    y_pred = classifier.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    accuracy.append(accuracy_score(y[test], y_pred))
    print(classification_report(y[test], y_pred))
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i += 1
a = 0
for i in accuracy:
    a += i/10
print("Avg. accuracy:",a)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'SVM (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

# Random Forest-------------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators  = 20, max_depth = 6)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
accuracy = []
from sklearn.metrics import accuracy_score, classification_report
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    y_pred = classifier.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    accuracy.append(accuracy_score(y[test], y_pred))
    print(classification_report(y[test], y_pred))
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i += 1
a = 0
for i in accuracy:
    a += i/10
print("Avg. accuracy:",a)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='cyan',
         label=r'Random Forest (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

# MLP-------------------------------------
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(9,), alpha=1e-4,momentum=0.3, 
                    random_state=0,)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
accuracy = []
from sklearn.metrics import accuracy_score, classification_report
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    y_pred = classifier.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    accuracy.append(accuracy_score(y[test], y_pred))
    print(classification_report(y[test], y_pred))
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i += 1
a = 0
for i in accuracy:
    a += i/10
print("Avg. accuracy:",a)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='magenta',
         label=r'MLP (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of models (Stratified 10 Fold CV)')
plt.legend(loc="lower right")
plt.show()
