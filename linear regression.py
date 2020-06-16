#importing important libraries.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

#alloting data from table to variables

Data_Table = pd.read_csv('50_startups.csv')
X_data = Data_Table.iloc[:,:3]
y_data = Data_Table.iloc[:,4]

#splitting data for testing and training

X_Train , X_Test , Y_Train , Y_Test = train_test_split(X_data , y_data , test_size=1/10)

#fitting linear regression - model

L_regg = LinearRegression()
L_regg.fit(X_Train,Y_Train)
Y_predict = L_regg.predict(X_Test)

#predicting profit from external data

Ex_data = np.array([76253.86,113867.3,298664.47])
Y_ex_data_pred = L_regg.predict(Ex_data.reshape(1,-3))
print(Y_ex_data_pred)

#predicting profit from taking user input

RnD_cost=float(input('Enter the R&D spend of your startup-'))
Admin_cost=float(input('Enter the administration cost of your startup-'))
Market_cost=float(input('Enter the marketing cost of your startup-'))
User_data=np.array([RnD_cost,Admin_cost,Market_cost])
Y_user_data_pred= L_regg.predict(User_data.reshape(1,-3))
print(Y_user_data_pred)
