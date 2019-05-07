import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Loading data 
def loading_train_homedata():
    return pd.read_csv(r'\Users\Rafeeq\Desktop\MLproject\premData-home.csv')

def loading_train_awaydata():
        return pd.read_csv(r'\Users\Rafeeq\Desktop\MLproject\premData-away.csv')

homeData = loading_train_homedata()
homeData = homeData.fillna(0)

x = homeData.drop("FTHG", axis=1)
y = homeData["FTHG"].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)

####################### HOME TEAM SOFTMAX REGRESSION #######################
softmax_reg = LogisticRegression(multi_class = "multinomial", solver = "saga", C = 10, max_iter = 1000)
softmax_reg = softmax_reg.fit(x_train, y_train)

y_pred = softmax_reg.predict(x_test)

softmax_mse = metrics.mean_squared_error(y_test,y_pred)
print("MSE: ", softmax_mse) 

softmax_rmse = np.sqrt(softmax_mse)
print("RMSE: ", softmax_rmse)

accuracy = softmax_reg.score(x_test,y_test)
print('home Accuracy: ' + str(np.round((accuracy)*100, 2)) + '%')

################# AWAY TEAM SOFTMAX REGRESSION ################3
awayData = loading_train_awaydata()
awayData = awayData.fillna(0)

x_away = awayData.drop("FTAG", axis=1)
y_away = awayData["FTAG"].copy()

x_train_away, x_test_away, y_train_away, y_test_away = train_test_split(x_away, y_away, test_size=0.2, random_state=0)

softmax_reg_away = LogisticRegression(multi_class = "multinomial", solver = "saga", C = 10, max_iter = 1000)
softmax_reg_away = softmax_reg.fit(x_train_away, y_train_away)

y_pred_away = softmax_reg_away.predict(x_test_away)

softmax_mse_away = metrics.mean_squared_error(y_test_away, y_pred_away)
print("Away MSE: ", softmax_mse_away) 

softmax_rmse_away = np.sqrt(softmax_mse_away)
print("Away RMSE: ", softmax_rmse_away)

accuracy_away = softmax_reg_away.score(x_test_away,y_test_away)
print('away Accuracy: ' + str(np.round((accuracy_away)*100, 2)) + '%')
