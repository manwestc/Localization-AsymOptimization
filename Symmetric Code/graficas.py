# Load libraries
from matplotlib import pyplot
from math import sqrt
from math import ceil
from math import fabs
from numpy import arange
from numpy import array
from numpy import isnan
from numpy import where
from numpy import take
from numpy import vstack
from numpy import logspace
from scipy.stats import mode
from scipy.stats import nanmean
from scipy.stats import describe
from pandas import read_csv
from pandas import set_option
from pandas import DataFrame
from pandas.tools.plotting import scatter_matrix

dataset = read_csv("raspberry.csv")
x = ['Tx01', 'Tx02', 'Tx03', 'Tx04', 'Tx05', 'Tx06', 'Tx07']
#pyplot.plot(y1)
#pyplot.show()

fig = pyplot.figure()
fig.suptitle('Precision')
ax = fig.add_subplot(111)
y = []
for i in range(1, 8):
	y.append(tuple(dataset.values[:, i]))

ax.boxplot(y)
"""
ax.boxplot(tuple(dataset.values[:, 1]))
ax.boxplot(tuple(dataset.values[:, 2]))
ax.boxplot(tuple(dataset.values[:, 3]))
ax.boxplot(tuple(dataset.values[:, 4]))
ax.boxplot(tuple(dataset.values[:, 5]))
ax.boxplot(tuple(dataset.values[:, 6]))
ax.boxplot(tuple(dataset.values[:, 7]))
"""
ax.set_xticklabels(x)
#pyplot.show()

fig = pyplot.figure()
fig.suptitle('Error en metros')
ax = fig.add_subplot(111)
y = []
for i in range(8, 15):
	y.append(tuple(dataset.values[:, i]))

ax.boxplot(y)
"""
ax.boxplot(tuple(dataset.values[:, 1]))
ax.boxplot(tuple(dataset.values[:, 2]))
ax.boxplot(tuple(dataset.values[:, 3]))
ax.boxplot(tuple(dataset.values[:, 4]))
ax.boxplot(tuple(dataset.values[:, 5]))
ax.boxplot(tuple(dataset.values[:, 6]))
ax.boxplot(tuple(dataset.values[:, 7]))
"""
ax.set_xticklabels(x)

fig = pyplot.figure()
fig.suptitle('Algoritmo por Intensidad')
y = []
for i in range(0, 21):
	y.append(tuple(dataset.values[i, 1:8]))


x = list(dataset.values[:, 0])
for i in range(7):
	pyplot.xticks(range(21), x)
	pyplot.plot([y[j][i] for j in range(21)], '-+', label="xd")

pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pyplot.show()


dataset.plot(kind='density', subplots=True, xlim=[-120, -49])



fig = pyplot.figure()
for i in range(1, 8):
	dataset = read_csv("raspberry/Tx_0x0" + str(i) + ".csv")
	ax = fig.add_subplot(33*10 + i)
	ax.set_xlabel('Tx=0x0'+str(i))
	cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
	if(i%3 == 0):
		fig.colorbar(cax)

for i in range(1, 8):
	dataset = read_csv("raspberry/Tx_0x0" + str(i) + ".csv")
	plt = dataset.plot(kind='density', subplots=True, layout=(2, 3), sharex=False, sharey=True, title='Raspberry Tx=0x0' + str(i), legend=True)
	plt[0][0].set_xlabel('Frecuency dBm')
	plt[0][1].set_xlabel('Frecuency dBm')
	plt[0][2].set_xlabel('Frecuency dBm')
	plt[1][0].set_xlabel('Frecuency dBm')
	plt[1][1].set_xlabel('Frecuency dBm')
	plt[1][2].set_xlabel('Sector       ')

fig = pyplot.figure()
ax = fig.add_subplot(331)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
ax.set_xlabel('jeje')
ax = fig.add_subplot(332)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
ax.set_xlabel('jojo')
ax = fig.add_subplot(333)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
ax = fig.add_subplot(334)#c3 f1
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
ax = fig.add_subplot(335)# f2
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
ax = fig.add_subplot(336)#c3 f1
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
ax = fig.add_subplot(337)#c3 f1
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
pyplot.show()


from pandas.tools.plotting import radviz
columns=["Be07","Be08","Be09","Be10","Be11", "Sector"]
i = 1
indices = [x in [1,2,3,4,5,6] for x in dataset.values[:,5]]
arr = array(list(compress(dataset.values[:,:], indices)))
radviz(DataFrame(arr, columns=columns), 'Sector')
ax.grid(True)
pyplot.show()
