# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:58:21 2024

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import wbgapi as wb
import plotly.express as px


def lorenz_curve(y):
    """
    Calculates the Lorenz Curve, a graphical representation of
    the distribution of income or wealth.

    It returns the cumulative share of people (x-axis) and
    the cumulative share of income earned.

    Parameters
    ----------
    y : array_like(float or int, ndim=1)
        Array of income/wealth for each individual.
        Unordered or ordered is fine.

    Returns
    -------
    cum_people : array_like(float, ndim=1)
        Cumulative share of people for each person index (i/n)
    cum_income : array_like(float, ndim=1)
        Cumulative share of income for each person index


    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Lorenz_curve

    Examples
    --------
    >>> a_val, n = 3, 10_000
    >>> y = np.random.pareto(a_val, size=n)
    >>> f_vals, l_vals = lorenz(y)

    """

    n = len(y)
    y = np.sort(y)
    s = np.zeros(n + 1)
    s[1:] = np.cumsum(y)
    cum_people = np.zeros(n + 1)
    cum_income = np.zeros(n + 1)
    for i in range(1, n + 1):
        cum_people[i] = i / n
        cum_income[i] = s[i] / s[n]
    return cum_people, cum_income


n = 2000
sample = np.exp(np.random.randn(n))

fig, ax = plt.subplots()

f_vals, l_vals = lorenz_curve(sample)
ax.plot(f_vals, l_vals, label=f'lognormal sample', lw=2)
ax.plot(f_vals, f_vals, label='equality', lw=2)

ax.vlines([0.8], [0.0], [0.43], alpha=0.5, colors='k', ls='--')
ax.hlines([0.43], [0], [0.8], alpha=0.5, colors='k', ls='--')
ax.set_xlim((0, 1))
ax.set_xlabel("share of households")
ax.set_ylim((0, 1))
ax.set_ylabel("share of wealth")
ax.legend()
plt.show()


url = 'E:/econ 自学资料/dataset/SCF_plus_mini.txt'
df = pd.read_csv(url,delimiter=',')
df_income_wealth = df.dropna()

df_income_wealth.head(n=5)

df = df_income_wealth 

varlist = ['n_wealth',    # net wealth 
           't_income',    # total income
           'l_income']    # labor income

years = df.year.unique()

# Create lists to store Lorenz data

F_vals, L_vals = [], []

for var in varlist:
    # create lists to store Lorenz curve data
    f_vals = []
    l_vals = []
    for year in years:

        # Repeat the observations according to their weights
        counts = list(round(df[df['year'] == year]['weights'] )) 
        y = df[df['year'] == year][var].repeat(counts)
        y = np.asarray(y)
        
        # Shuffle the sequence to improve the plot
        rd.shuffle(y)    
               
        # calculate and store Lorenz curve data
        f_val, l_val = lorenz_curve(y)
        f_vals.append(f_val)
        l_vals.append(l_val)
        
    F_vals.append(f_vals)
    L_vals.append(l_vals)

f_vals_nw, f_vals_ti, f_vals_li = F_vals
l_vals_nw, l_vals_ti, l_vals_li = L_vals

fig, ax = plt.subplots()
ax.plot(f_vals_nw[-1], l_vals_nw[-1], label=f'net wealth')
ax.plot(f_vals_ti[-1], l_vals_ti[-1], label=f'total income')
ax.plot(f_vals_li[-1], l_vals_li[-1], label=f'labor income')
ax.plot(f_vals_nw[-1], f_vals_nw[-1], label=f'equality')
ax.set_xlabel("share of households")
ax.set_ylabel("share of income/wealth")
ax.legend()
plt.show()


fig, ax = plt.subplots()
f_vals, l_vals = lorenz_curve(sample)
ax.plot(f_vals, l_vals, label=f'lognormal sample', lw=2)
ax.plot(f_vals, f_vals, label='equality', lw=2)
ax.fill_between(f_vals, l_vals, f_vals, alpha=0.06)
ax.set_ylim((0, 1))
ax.set_xlim((0, 1))
ax.text(0.04, 0.5, r'$G = 2 \times$ shaded area')
ax.set_xlabel("share of households (%)")
ax.set_ylabel("share of wealth (%)")
ax.legend()
plt.show()


fig, ax = plt.subplots()
f_vals, l_vals = lorenz_curve(sample)
ax.plot(f_vals, l_vals, label='lognormal sample', lw=2)
ax.plot(f_vals, f_vals, label='equality', lw=2)
ax.fill_between(f_vals, l_vals, f_vals, alpha=0.06)
ax.fill_between(f_vals, l_vals, np.zeros_like(f_vals), alpha=0.06)
ax.set_ylim((0, 1))
ax.set_xlim((0, 1))
ax.text(0.55, 0.4, 'A')
ax.text(0.75, 0.15, 'B')
ax.set_xlabel("share of households")
ax.set_ylabel("share of wealth")
ax.legend()
plt.show()


def gini_coefficient(y):
    r"""
    Implements the Gini inequality index

    Parameters
    ----------
    y : array_like(float)
        Array of income/wealth for each individual.
        Ordered or unordered is fine

    Returns
    -------
    Gini index: float
        The gini index describing the inequality of the array of income/wealth

    References
    ----------

    https://en.wikipedia.org/wiki/Gini_coefficient
    """
    n = len(y)
    i_sum = np.zeros(n)
    for i in range(n):
        for j in range(n):
            i_sum[i] += abs(y[i] - y[j])
    return np.sum(i_sum) / (2 * n * np.sum(y))


%%time
k = 5
σ_vals = np.linspace(0.2, 4, k)
n = 2_000

ginis = []

for σ in σ_vals:
    μ = -σ**2 / 2
    y = np.exp(μ + σ * np.random.randn(n))
    ginis.append(gini_coefficient(y))
    
    
def plot_inequality_measures(x, y, legend, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', label=legend)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax

fix, ax = plot_inequality_measures(σ_vals, 
                                  ginis, 
                                  'simulated', 
                                  '$\sigma$', 
                                  'Gini coefficients')
plt.show()


wb.search("gini")


# Fetch gini data for all countries
gini_all = wb.data.DataFrame("SI.POV.GINI")
# remove 'YR' in index and convert to integer
gini_all.columns = gini_all.columns.map(lambda x: int(x.replace('YR',''))) 

# Create a long series with a multi-index of the data to get global min and max values
gini_all = gini_all.unstack(level='economy').dropna()

# Build a histogram
ax = gini_all.plot(kind="hist", bins=20)
ax.set_xlabel("Gini coefficient")
ax.set_ylabel("frequency")
plt.show()

data = wb.data.DataFrame("SI.POV.GINI", "USA")
data.head(n=5)
# remove 'YR' in index and convert to integer
data.columns = data.columns.map(lambda x: int(x.replace('YR','')))

data = data.T           # Obtain years as rows
data_usa = data['USA']  # pd.Series of US data

fig, ax = plt.subplots()
ax = data_usa.plot(ax=ax)
ax.set_ylim(data_usa.min()-1, data_usa.max()+1)
ax.set_ylabel("Gini coefficient (income)")
ax.set_xlabel("year")
plt.show()

df_income_wealth.year.describe()

data_url = 'E:/econ 自学资料/dataset/usa-gini-nwealth-tincome-lincome.csv'
ginis = pd.read_csv(data_url, index_col='year')
ginis.head(n=5)


fig, ax = plt.subplots()
ax.plot(years, ginis["n_wealth"], marker='o')
ax.set_xlabel("year")
ax.set_ylabel("Gini coefficient")
plt.show()

data = gini_all.unstack()
data.columns

ax = data[['USA','GBR', 'NOR']].plot()
ax.set_xlabel('year')
ax.set_ylabel('Gini coefficient')
ax.legend(title="")
plt.show()

data[['NOR']].dropna().head(n=5)


data['NOR'] = data['NOR'].ffill()
ax = data[['USA','GBR', 'NOR']].plot()
ax.set_xlabel('year')
ax.set_ylabel('Gini coefficient')
ax.legend(title="")
plt.show()

countries = ['USA', 'NOR', 'GBR']
gdppc = wb.data.DataFrame("NY.GDP.PCAP.KD", countries)
# remove 'YR' in index and convert to integer
gdppc.columns = gdppc.columns.map(lambda x: int(x.replace('YR',''))) 
gdppc = gdppc.T

plot_data = pd.DataFrame(data[countries].unstack())
plot_data.index.names = ['country', 'year']
plot_data.columns = ['gini']

pgdppc = pd.DataFrame(gdppc.unstack())
pgdppc.index.names = ['country', 'year']
pgdppc.columns = ['gdppc']
plot_data = plot_data.merge(pgdppc, left_index=True, right_index=True)
plot_data.reset_index(inplace=True)

min_year = plot_data.year.min()
max_year = plot_data.year.max()

labels = [1979, 1986, 1991, 1995, 2000, 2020, 2021, 2022] + \
         list(range(min_year,max_year,5))
plot_data.year = plot_data.year.map(lambda x: x if x in labels else None)

fig = px.line(plot_data, 
              x = "gini", 
              y = "gdppc", 
              color = "country", 
              text = "year", 
              height = 800,
              labels = {"gini" : "Gini coefficient", "gdppc" : "GDP per capita"}
             )
fig.update_traces(textposition="bottom right")
fig.show()



# transfer the survey weights from absolute into relative values
df1 = df_income_wealth
df2 = df1.groupby('year').sum(numeric_only=True).reset_index()
df3 = df2[['year', 'weights']]
df3.columns = 'year', 'r_weights'
df4 = pd.merge(df3, df1, how="left", on=["year"])
df4['r_weights'] = df4['weights'] / df4['r_weights']

# create weighted nw, ti, li
df4['weighted_n_wealth'] = df4['n_wealth'] * df4['r_weights']
df4['weighted_t_income'] = df4['t_income'] * df4['r_weights']
df4['weighted_l_income'] = df4['l_income'] * df4['r_weights']

# extract two top 10% groups by net wealth and total income.
df6 = df4[df4['nw_groups'] == 'Top 10%']
df7 = df4[df4['ti_groups'] == 'Top 10%']

# calculate the sum of weighted top 10% by net wealth,
#   total income and labor income.
df5 = df4.groupby('year').sum(numeric_only=True).reset_index()
df8 = df6.groupby('year').sum(numeric_only=True).reset_index()
df9 = df7.groupby('year').sum(numeric_only=True).reset_index()

df5['weighted_n_wealth_top10'] = df8['weighted_n_wealth']
df5['weighted_t_income_top10'] = df9['weighted_t_income']
df5['weighted_l_income_top10'] = df9['weighted_l_income']

# calculate the top 10% shares of the three variables.
df5['topshare_n_wealth'] = df5['weighted_n_wealth_top10'] / \
    df5['weighted_n_wealth']
df5['topshare_t_income'] = df5['weighted_t_income_top10'] / \
    df5['weighted_t_income']
df5['topshare_l_income'] = df5['weighted_l_income_top10'] / \
    df5['weighted_l_income']

# we only need these vars for top 10 percent shares
df_topshares = df5[['year', 'topshare_n_wealth',
                    'topshare_t_income', 'topshare_l_income']]


fig, ax = plt.subplots()
ax.plot(years, df_topshares["topshare_l_income"],
        marker='o', label="labor income")
ax.plot(years, df_topshares["topshare_n_wealth"],
        marker='o', label="net wealth")
ax.plot(years, df_topshares["topshare_t_income"],
        marker='o', label="total income")
ax.set_xlabel("year")
ax.set_ylabel("top $10\%$ share")
ax.legend()
plt.show()











