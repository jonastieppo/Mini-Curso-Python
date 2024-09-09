# %%
'''
Gerando 10 corpos de prova fakes a partir de um corpo base
'''
import pandas as pd
from experimentalTreatingIsiPol.main import ReadExperimentalData, plot_helper
import matplotlib.pyplot as plt
import numpy as np
classInit = ReadExperimentalData(archive_name=r'D:\Jonas\Mini Curso Python\experimental_data\exemplo_polimero.csv', 
                                 skiprows=10)

df = classInit.raw_data
columns = ['Tempo', 'Travessa', 'Carga', 'Extensometro']
df.columns = columns
fig, ax = plt.subplots(figsize=(4,4))

plot_helper(ax=ax,x=df['Extensometro'], 
            y=df['Carga'], 
            label='Polimero', 
            xlabel='Deslocamento [mm]', 
            ylabel='Carga [N]')


# %%
'''
Pegando declinidade inicial
'''
from scipy.optimize import curve_fit

def linear_func(x,a,b):
    return a*x+b

# x_linear = df['Extensometro'][df['Extensometro']<2.3][df['Extensometro']>2]
# y_linear = df['Carga'][df['Extensometro']<2.3][df['Extensometro']>2]

strain_max = 0.8
strain_min = 0.4

x_linear = df['Extensometro'][df['Extensometro']<strain_max][df['Extensometro']>strain_min]
y_linear = df['Carga'][df['Extensometro']<strain_max][df['Extensometro']>strain_min]


coef, _ = curve_fit(linear_func, xdata=x_linear, ydata=y_linear)

K = coef[0]
offset = coef[1]


fig2, ax2 = plt.subplots(figsize=(4,4))
plot_helper(ax=ax2,
            x=x_linear, 
            y=y_linear, 
            label='Polimero', 
            xlabel='Deslocamento [mm]', 
            ylabel='Carga [N]')

plot_helper(ax=ax2,
            x=x_linear, 
            y=K*x_linear+offset, 
            label='Rigidez', 
            xlabel='Deslocamento [mm]', 
            ylabel='Carga [N]')
# %%
'''
Limpando dados
'''
df = df[df['Extensometro']>strain_min]

fig, ax = plt.subplots(figsize=(4,4))

plot_helper(ax=ax,x=df['Extensometro'], 
            y=df['Carga'], 
            label='Polimero', 
            xlabel='Deslocamento [mm]', 
            ylabel='Carga [N]')

# %%
from scipy.optimize import fsolve
import math

max_force = 140
def equations(p):
    a, b = p
    # eq1 = a - K/(b*(3**(b-1)))
    eq1 = a*b*(3**(b-1))-K
    eq2 = a*3**b-max_force

    return (eq1, eq2)


a, b =  fsolve(equations, (0.5,10 ))



# %%

x_fake = np.linspace(0,3,1000)
y_fake = a*x_fake**(b)

fig2, ax2 = plt.subplots(figsize=(4,4))
plot_helper(ax=ax2,
            x=x_fake, 
            y=y_fake, 
            label='Polimero', 
            xlabel='Deslocamento [mm]', 
            ylabel='Carga [N]')

data = {'Carga':[], 'Extensometro':[]}

data['Carga'] =  list(y_fake) + list(df['Carga']) 


data['Extensometro'] = list(x_fake) + list(df['Extensometro']+3-(min(df['Extensometro']))) 

fig3, ax3 = plt.subplots(figsize=(8,8))
plot_helper(ax=ax3,
            x=data['Extensometro'], 
            y=data['Carga'], 
            label='Polimero', 
            xlabel='Deslocamento [mm]', 
            ylabel='Carga [N]')


# %%
'''
Criando dados para tratamento posterior
'''
df_inital_disp = pd.DataFrame(data=data)

df_inital_disp.to_csv("exemplo_curva_inicial.csv",sep=';', decimal=',')
