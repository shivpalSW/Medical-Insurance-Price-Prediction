print
("ModelTraining File")


# %%time
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,plot_confusion_matrix,classification_report

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,roc_auc_score,roc_curve

import matplotlib.pyplot as plt 
import seaborn as sns 
 

#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

################################################################
mse_Lin_Train= mean_squared_error(y_train,y_pred_train)
print("Mean Squared Error is: ",mse_Lin_Train)

rmse_Lin_Train= np.sqrt(mse_Lin_Train)
print("Root Mean Squared Error is: ",rmse_Lin_Train)

mae_Lin_Train= mean_absolute_error(y_train,y_pred_train)
print("Mean Absolute Error is: ",mae_Lin_Train)

r2Score_Lin_Train= r2_score(y_train,y_pred_train)
print("R Squared is: ",r2Score_Lin_Train)


################################################################
mse_Lin_Test= mean_squared_error(y_test,y_pred)
print("Mean Squared Error is: ",mse_Lin_Test)

rmse_Lin_Test= np.sqrt(mse_Lin_Test)
print("Root Mean Squared Error is: ",rmse_Lin_Test)

mae_Lin_Test= mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error is: ",mae_Lin_Test)

r2score_Lin_Test= r2_score(y_test,y_pred)
print("R Squared is: ",r2score_Lin_Test)