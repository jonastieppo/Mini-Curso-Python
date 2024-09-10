# %%
def fibonacci_recursive(n,a,b):
  if n == 1:
    return a
  if n == 2:
    return b
  else:
    return fibonacci_recursive(n - 1,a,b) + fibonacci_recursive(n - 2,a,b)
  

def fibonnacci(n,a,b):
  
  if n==1:
    return a
  
  if n==2:
    return b

  counter = 3
  while counter<=n:
    n_fibo = a + b
    a = b
    b = n_fibo
    counter = counter + 1


  return n_fibo
# %%
'''
Número de Lucas
'''
import numpy as np
def lucasNumber(n):
  '''
  Retorna a sequência de Fibonacci com F(0)=1 e F(1)=3.
  '''

  return ((1+np.sqrt(5))/2)**n+((1-np.sqrt(5))/2)**n
