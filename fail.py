import numpy as np
import lmfit
from scipy.stats import norm
from scipy.stats import poisson
import matplotlib.pyplot as plt

def model_poisson_forced(x, gain, shift, sigma, mu):
	p = poisson.pmf(range(5), mu)
	cdf = np.zeros(len(x))
	for k in range(len(p)):
		cdf += p[k]*norm.cdf(x, loc = gain*k + shift, scale = sigma)
	return cdf

def model_lin(x, gain):
	return x*gain

active_area = np.genfromtxt('temp_active_area').transpose()
x_vals = np.genfromtxt('temp_x').transpose()
ecdf = np.genfromtxt('temp_ecdf').transpose()

model = lmfit.Model(model_poisson_forced)
params = model.make_params()
params['gain'].set(value = 1, max = 1.5, min = .5)
params['shift'].set(value = 0, max = 1, min = -1)
params['sigma'].set(value = 300**-.5*4)
params['mu'].set(value = active_area.ravel().std()*.9, min = 0)

fit_result = model.fit(ecdf, params, x = x_vals) # Comment this line and see the data

fig, ax = plt.subplots()
x_axis = np.linspace(min(x_vals),max(x_vals),300)
ax.plot(x_axis, model.eval(params, x = x_axis), label = 'fit')
ax.plot(x_vals, ecdf, label = 'data')
ax.legend()
plt.show()
