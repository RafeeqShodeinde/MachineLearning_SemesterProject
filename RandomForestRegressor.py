import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
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

###################### HOME RANDOM FOREST REGRESSOR #############
forest_reg = RandomForestRegressor(n_estimators = 20, bootstrap = False, random_state = 0)
forest_reg = forest_reg.fit(x_train, y_train)

y_pred = forest_reg.predict(x_test)

forest_mse = metrics.mean_squared_error(y_test,y_pred)
print("MSE: ", forest_mse) 

forest_rmse = np.sqrt(forest_mse)
print("RMSE: ", forest_rmse)

accuracy = forest_reg.score(x_test,y_test)
print('home Accuracy: ' + str(np.round(accuracy*100, 2)) + '%')



##########AWAY TEAM RANDOM FOREST REGRESSOR##################
awayData = loading_train_awaydata()
awayData = awayData.fillna(0)

x_away = awayData.drop("FTAG", axis=1)
y_away = awayData["FTAG"].copy()

x_train_away, x_test_away, y_train_away, y_test_away = train_test_split(x_away, y_away, test_size=0.2, random_state=0)

forest_reg_away = RandomForestRegressor(n_estimators = 100, bootstrap = False, random_state = 0)
forest_reg_away = forest_reg_away.fit(x_train_away, y_train_away)

y_pred_away = forest_reg_away.predict(x_test_away)

forest_mse_away = metrics.mean_squared_error(y_test_away, y_pred_away)
print("Away MSE: ", forest_mse_away) 

forest_rmse_away = np.sqrt(forest_mse_away)
print("Away RMSE: ", forest_rmse_away)

accuracy_away = forest_reg_away.score(x_test_away,y_test_away)
print('away Accuracy: ' + str(np.round((accuracy_away)*100, 2)) + '%')

