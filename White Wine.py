import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score,KFold
#matplotlib inline
df = pd.read_csv('F:\wineQualityWhites.csv')


bins=(2,6.5,9)
group_names=['bad','good']
df['quality']=pd.cut(df['quality'],bins=bins,labels=group_names)
label_quality=LabelEncoder()
df['quality']=label_quality.fit_transform(df['quality'])
print(df['quality'].value_counts())
X=df.drop('quality',axis=1)
y=df['quality']
col_=X.columns
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


models = []
models.append(('SupportVectorClassifier', SVC()))
models.append(('StochasticGradientDecentC', SGDClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('LogisticRegression', LogisticRegression()))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
   kfold = KFold(n_splits=10, random_state=7,shuffle=True)
   cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(msg)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

from sklearn.metrics import mean_squared_error
print("Random Forest Classifier")
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test,pred_rfc))
score=rfc.score(X_test,y_test)
print('Original Accuracy: ',score)
MSE=mean_squared_error(y_test,pred_rfc)
RMSE=np.sqrt(MSE)
print('Mean Square Error: ',MSE)
print('Root Mean Square Error: ',RMSE)
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
print('Accuracy after Cross Validation: ',rfc_eval.mean())
x_predict = list(rfc.predict(X_test))
df1 = {'predicted':x_predict,'orignal':y_test}
print('Model Predictions:')
print(pd.DataFrame(df1).head(10))
feat_importances = pd.Series(rfc.feature_importances_, index=col_)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10),title='Important Features that determine Quality ')
plt.show()


print("Stochastic Gradient Classifier")
sgd=SGDClassifier(penalty=None)
sgd.fit(X_train,y_train)
pred_sgd=sgd.predict(X_test)
print(classification_report(y_test,pred_sgd))
print(confusion_matrix(y_test,pred_sgd))
score=sgd.score(X_test,y_test)
print('Original Accuracy: ',score)
MSE=mean_squared_error(y_test,pred_sgd)
RMSE=np.sqrt(MSE)
print('Mean Square error: ',MSE)
print('Root Mean Square Error: ',RMSE)
sgd_eval = cross_val_score(estimator = sgd, X = X_train, y = y_train, cv = 10)
print('Accuracy after Cross Validation: ',sgd_eval.mean())
x_predict = list(sgd.predict(X_test))
df1 = {'predicted':x_predict,'orignal':y_test}
print('Model Predictions:')
print(pd.DataFrame(df1).head(10))





#GRIDSEARCH CV
print("Support Vector Machine Classifier ")
'''param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
print(grid_svc.best_params_)'''
svc2 = SVC(C = 1.4, gamma =  1.1, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))
print(confusion_matrix(y_test,pred_svc2))
score=svc2.score(X_test,y_test)
print('Accuracy After Boosting: ',score)
MSE=mean_squared_error(y_test,pred_svc2)
RMSE=np.sqrt(MSE)
print('Mean Square Error: ',MSE)
print('Root Mean Square Error: ',RMSE)
x_predict = list(svc2.predict(X_test))
df1 = {'predicted':x_predict,'orignal':y_test}
print('Model Predictions:')
print(pd.DataFrame(df1).head(10))



print("Logistic Regression")
lgr=LogisticRegression(random_state=0)
lgr.fit(X_train,y_train)
pred_lgr=lgr.predict(X_test)
print(classification_report(y_test,pred_lgr))
print(confusion_matrix(y_test,pred_lgr))
score=lgr.score(X_test,y_test)
print('Original Accuracy: ',score)
MSE=mean_squared_error(y_test,pred_lgr)
RMSE=np.sqrt(MSE)
print('Mean Square Error: ',MSE)
print('Root Mean Square Error: ',RMSE)
x_predict = list(lgr.predict(X_test))
df1 = {'predicted':x_predict,'orignal':y_test}
print('Model Predictions:')
print(pd.DataFrame(df1).head(10))
