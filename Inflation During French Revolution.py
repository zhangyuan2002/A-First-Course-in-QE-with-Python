# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:16:07 2024

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

base_url = 'E:/econ 自学资料/dataset/'

fig_3_url = f'{base_url}fig_3.xlsx'
dette_url = f'{base_url}dette.xlsx'
assignat_url = f'{base_url}assignat.xlsx'

# Read the data from Excel file
data2 = pd.read_excel(dette_url, 
        sheet_name='Militspe', usecols='M:X', 
        skiprows=7, nrows=102, header=None)

# French military spending, 1685-1789, in 1726 livres
data4 = pd.read_excel(dette_url, 
        sheet_name='Militspe', usecols='D', 
        skiprows=3, nrows=105, header=None).squeeze()
        
years = range(1685, 1790)

plt.figure()
plt.plot(years, data4, '*-', linewidth=0.8)

plt.plot(range(1689, 1791), data2.iloc[:, 4], linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(labelsize=12)
plt.xlim([1689, 1790])
plt.xlabel('*: France')
plt.ylabel('Millions of livres')
plt.ylim([0, 475])

plt.tight_layout()
plt.show()

# Read the data from Excel file
data2 = pd.read_excel(dette_url, sheet_name='Militspe', usecols='M:X', 
                      skiprows=7, nrows=102, header=None)

# Plot the data
plt.figure()
plt.plot(range(1689, 1791), data2.iloc[:, 5], linewidth=0.8)
plt.plot(range(1689, 1791), data2.iloc[:, 11], linewidth=0.8, color='red')
plt.plot(range(1689, 1791), data2.iloc[:, 9], linewidth=0.8, color='orange')
plt.plot(range(1689, 1791), data2.iloc[:, 8], 'o-', 
         markerfacecolor='none', linewidth=0.8, color='purple')

# Customize the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(labelsize=12)
plt.xlim([1689, 1790])
plt.ylabel('millions of pounds', fontsize=12)

# Add text annotations
plt.text(1765, 1.5, 'civil', fontsize=10)
plt.text(1760, 4.2, 'civil plus debt service', fontsize=10)
plt.text(1708, 15.5, 'total govt spending', fontsize=10)
plt.text(1759, 7.3, 'revenues', fontsize=10)

plt.tight_layout()
plt.show()



# Read the data from the Excel file
data1 = pd.read_excel(dette_url, sheet_name='Debt', 
            usecols='R:S', skiprows=5, nrows=99, header=None)
data1a = pd.read_excel(dette_url, sheet_name='Debt', 
            usecols='P', skiprows=89, nrows=15, header=None)

# Plot the data
plt.figure()
plt.plot(range(1690, 1789), 100 * data1.iloc[:, 1], linewidth=0.8)

date = np.arange(1690, 1789)
index = (date < 1774) & (data1.iloc[:, 0] > 0)
plt.plot(date[index], 100 * data1[index].iloc[:, 0], 
         '*:', color='r', linewidth=0.8)

# Plot the additional data
plt.plot(range(1774, 1789), 100 * data1a, '*:', color='orange')

# Note about the data
# The French data before 1720 don't match up with the published version
# Set the plot properties
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().set_xlim([1688, 1788])
plt.ylabel('% of Taxes')

plt.tight_layout()
plt.show()


# Read data from Excel file
data5 = pd.read_excel(dette_url, sheet_name='Debt', usecols='K', 
                    skiprows=41, nrows=120, header=None)

# Plot the data
plt.figure()
plt.plot(range(1726, 1846), data5.iloc[:, 0], linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(labelsize=12)
plt.xlim([1726, 1845])
plt.ylabel('1726 = 1', fontsize=12)

plt.tight_layout()
plt.show()


# Read data from Excel file
data11 = pd.read_excel(assignat_url, sheet_name='Budgets',
        usecols='J:K', skiprows=22, nrows=52, header=None)

# Prepare the x-axis data
x_data = np.concatenate([
    np.arange(1791, 1794 + 8/12, 1/12),
    np.arange(1794 + 9/12, 1795 + 3/12, 1/12)
])

# Remove NaN values from the data
data11_clean = data11.dropna()

# Plot the data
plt.figure()
h = plt.plot(x_data, data11_clean.values[:, 0], linewidth=0.8)
h = plt.plot(x_data, data11_clean.values[:, 1], '--', linewidth=0.8)

# Set plot properties
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.xlim([1791, 1795 + 3/12])
plt.xticks(np.arange(1791, 1796))
plt.yticks(np.arange(0, 201, 20))

# Set the y-axis label
plt.ylabel('millions of livres', fontsize=12)

plt.tight_layout()
plt.show()



# Read data from Excel file
data12 = pd.read_excel(assignat_url, sheet_name='seignor', 
         usecols='F', skiprows=6, nrows=75, header=None).squeeze()

# Create a figure and plot the data
plt.figure()
plt.plot(pd.date_range(start='1790', periods=len(data12), freq='M'),
         data12, linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axhline(y=472.42/12, color='r', linestyle=':')
plt.xticks(ticks=pd.date_range(start='1790', 
           end='1796', freq='AS'), labels=range(1790, 1797))
plt.xlim(pd.Timestamp('1791'),
         pd.Timestamp('1796-02') + pd.DateOffset(months=2))
plt.ylabel('millions of livres', fontsize=12)
plt.text(pd.Timestamp('1793-11'), 39.5, 'revenues in 1788', 
         verticalalignment='top', fontsize=12)

plt.tight_layout()
plt.show()



# Read the data from Excel file
data7 = pd.read_excel(assignat_url, sheet_name='Data', 
          usecols='P:Q', skiprows=4, nrows=80, header=None)
data7a = pd.read_excel(assignat_url, sheet_name='Data', 
          usecols='L', skiprows=4, nrows=80, header=None)
# Create the figure and plot
plt.figure()
x = np.arange(1789 + 10/12, 1796 + 5/12, 1/12)
h, = plt.plot(x, 1. / data7.iloc[:, 0], linestyle='--')
h, = plt.plot(x, 1. / data7.iloc[:, 1], color='r')

# Set properties of the plot
plt.gca().tick_params(labelsize=12)
plt.yscale('log')
plt.xlim([1789 + 10/12, 1796 + 5/12])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add vertical lines
plt.axvline(x=1793 + 6.5/12, linestyle='-', linewidth=0.8, color='orange')
plt.axvline(x=1794 + 6.5/12, linestyle='-', linewidth=0.8, color='purple')

# Add text
plt.text(1793.75, 120, 'Terror', fontsize=12)
plt.text(1795, 2.8, 'price level', fontsize=12)
plt.text(1794.9, 40, 'gold', fontsize=12)


plt.tight_layout()
plt.show()


# Read the data from Excel file
data7 = pd.read_excel(assignat_url, sheet_name='Data', 
        usecols='P:Q', skiprows=4, nrows=80, header=None)
data7a = pd.read_excel(assignat_url, sheet_name='Data', 
        usecols='L', skiprows=4, nrows=80, header=None)

# Create the figure and plot
plt.figure()
h = plt.plot(pd.date_range(start='1789-11-01', periods=len(data7), freq='M'), 
            (data7a.values * [1, 1]) * data7.values, linewidth=1.)
plt.setp(h[1], linestyle='--', color='red')

plt.vlines([pd.Timestamp('1793-07-15'), pd.Timestamp('1793-07-15')], 
           0, 3000, linewidth=0.8, color='orange')
plt.vlines([pd.Timestamp('1794-07-15'), pd.Timestamp('1794-07-15')], 
           0, 3000, linewidth=0.8, color='purple')

plt.ylim([0, 3000])

# Set properties of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(labelsize=12)
plt.xlim(pd.Timestamp('1789-11-01'), pd.Timestamp('1796-06-01'))
plt.ylabel('millions of livres', fontsize=12)

# Add text annotations
plt.text(pd.Timestamp('1793-09-01'), 200, 'Terror', fontsize=12)
plt.text(pd.Timestamp('1791-05-01'), 750, 'gold value', fontsize=12)
plt.text(pd.Timestamp('1794-10-01'), 2500, 'real value', fontsize=12)


plt.tight_layout()
plt.show()


def fit(x, y):

    b = np.cov(x, y)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean()

    return a, b

# Load data
caron = np.load('E:/econ 自学资料/dataset/caron.npy')
nom_balances = np.load('E:/econ 自学资料/dataset/nom_balances.npy')

infl = np.concatenate(([np.nan], 
      -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000


# Regress y on x for three periods
a1, b1 = fit(bal[1:31], infl[1:31])
a2, b2 = fit(bal[31:44], infl[31:44])
a3, b3 = fit(bal[44:63], infl[44:63])

# Regress x on y for three periods
a1_rev, b1_rev = fit(infl[1:31], bal[1:31])
a2_rev, b2_rev = fit(infl[31:44], bal[31:44])
a3_rev, b3_rev = fit(infl[44:63], bal[44:63])

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# First subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', 
         color='blue', label='real bills period')

# Second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# Third subsample
plt.plot(bal[44:63], infl[44:63], '*', 
        color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()


# Regress y on x for three periods
a1, b1 = fit(bal[1:31], infl[1:31])
a2, b2 = fit(bal[31:44], infl[31:44])
a3, b3 = fit(bal[44:63], infl[44:63])

# Regress x on y for three periods
a1_rev, b1_rev = fit(infl[1:31], bal[1:31])
a2_rev, b2_rev = fit(infl[31:44], bal[31:44])
a3_rev, b3_rev = fit(infl[44:63], bal[44:63])


plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# First subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')

# Second subsample
plt.plot(bal[34:44], infl[34:44], '+', color='red', label='terror')

# Third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# First subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', 
        color='blue', label='real bills period')
plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue')

# Second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# Third subsample
plt.plot(bal[44:63], infl[44:63], '*', 
        color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()



plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# First subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', 
        color='blue', label='real bills period')

# Second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
plt.plot(a2_rev + b2_rev * infl[31:44], infl[31:44], color='red')

# Third subsample
plt.plot(bal[44:63], infl[44:63], '*', 
        color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()



plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# First subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', 
        color='blue', label='real bills period')

# Second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# Third subsample
plt.plot(bal[44:63], infl[44:63], '*', 
    color='orange', label='classic Cagan hyperinflation')
plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()



plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# First subsample
plt.plot(bal[1:31], infl[1:31], 'o', 
    markerfacecolor='none', color='blue', label='real bills period')

# Second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# Third subsample
plt.plot(bal[44:63], infl[44:63], '*', 
        color='orange', label='classic Cagan hyperinflation')
plt.plot(a3_rev + b3_rev * infl[44:63], infl[44:63], color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()

























