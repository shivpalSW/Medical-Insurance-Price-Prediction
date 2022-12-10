# %%time
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,plot_confusion_matrix,classification_report

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,roc_auc_score,roc_curve

import matplotlib.pyplot as plt 
import seaborn as sns



################################################################
mse_Lin_Train= mean_squared_error(y_train,y_pred_train)
print("Mean Squared Error is: ",mse_Lin_Train)
