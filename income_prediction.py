import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_excel("canadaincome.xlsx")
print(df)

plt.scatter(df.year, df.income)
plt.xlabel('Year')
plt.ylabel('Income')
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[["year"]],df.income)
predicted_income = reg.predict([[2017]])
print(predicted_income)

plt.scatter(df.year, df.income, color='red', marker='+')
plt.xlabel('Year', fontsize=20, color='purple')
plt.ylabel('Income', fontsize=20, color='purple')
plt.plot(df.year, reg.predict(df[['year']]),color='blue' )
plt.show()

d = pd.read_excel("years.xlsx")
p = reg.predict(d)
d['income'] = p
print(d)
d.to_excel("incomeprediction.xlsx")

plt.scatter(d.year, d.income, color='brown', marker='o')
plt.xlabel('Year', fontsize=20, color='purple')
plt.ylabel('Income', fontsize=20, color='purple')
plt.show()
