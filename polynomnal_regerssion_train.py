#Polynormial Regression
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

""" Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""
#fitting the linear regresson
from sklearn.linear_model import LinearRegression
linreg1=LinearRegression()
linreg1.fit(x, y)
 
#fitting the poly regression
from sklearn.preprocessing import PolynomialFeatures
polyreg=PolynomialFeatures(degree=10)
x_poly=polyreg.fit_transform(x)
linreg2=LinearRegression()
linreg2.fit(x_poly, y)


#visualising the Linear regerssion
plt.scatter(x, y,color='red')
plt.plot(x, linreg1.predict(x),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('postion')
plt.ylabel('salary')

plt.show()



#visualtising the poly regerssion
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x, y,color='red')
plt.plot(x_grid, linreg2.predict(polyreg.fit_transform(x_grid)),color='blue')
plt.title('Bluff')
plt.xlabel('postion')
plt.ylabel('salary')

plt.show()



#predict new value
linreg1.predict([[6.5]])
#predict new value of poly
linreg2.predict(polyreg.fit_transform([[6.5]]))