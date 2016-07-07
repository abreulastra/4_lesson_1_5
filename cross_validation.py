# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:04:32 2016

@author: AbreuLastra_Work
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

loansData = pd.read_csv('/Users/AbreuLastra_Work/Dropbox (Personal)/thinkful/4_lesson_1_5/loansData_clean.csv')


# To run regressions
X=loansData[['FICO_Score', 'Amount.Requested']]
y=loansData['clean_Interest_rate']

lm2 = LinearRegression()
print cross_val_score(lm2, X, y, cv=10, scoring='mean_squared_error')


#Save database as a CSV files
loansData.to_csv('loansData_clean.csv', header=True, index=False)
