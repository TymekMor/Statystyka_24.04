import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

setosa = iris[iris['species']=='setosa']['petal_length']
versicolor = iris[iris['species']=='versicolor']['petal_length']

def kde(x, data, h, kernel):
    kde_vals = np.zeros_like(x)
    n = len(data)
    for i in range(len(x)):
        kde_vals[i] = (1/(n*h)) * np.sum(kernel((x[i]-data)/h))
    return kde_vals

def triangle_kernel(x):
    return np.maximum(1 - np.abs(x), 0)

def gaussian_kernel(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

def epanechnikov_kernel(x):
    return 0.75 * np.maximum(1 - x ** 2, 0)

x_vals = np.linspace(0,8,100)
h = 0.4

# Gauss Setosa & Versi
kde_gauss_setosa = kde(x_vals,setosa,h,gaussian_kernel)
kde_gauss_versicolor = kde(x_vals,versicolor,h,gaussian_kernel)
# Triangle Setosa & Versi
kde_triangle_setosa = kde(x_vals,setosa,h,triangle_kernel)
kde_triangle_versicolor = kde(x_vals,versicolor,h,triangle_kernel)
# Epanechnikov Setosa & Versi
kde_epa_setosa = kde(x_vals,setosa,h,epanechnikov_kernel)
kde_epa_versicolor = kde(x_vals,versicolor,h,epanechnikov_kernel)

plt.ylim(top=3)

#Histogramy Setosy i Versi
plt.hist(setosa,bins=20,density=True,alpha=0.6,label="Setosa")
plt.hist(versicolor,bins=20,density=True,alpha=0.6,label="Versicolor")

#KDE Plot Gauss
plt.plot(x_vals, kde_gauss_setosa, label="Kde_Gauss_Setosa")
plt.plot(x_vals, kde_gauss_versicolor, label="Kde_Gauss_Versicolor")
#KDE Plot Triangle
plt.plot(x_vals, kde_triangle_setosa, label="Kde_Triangle_Setosa")
plt.plot(x_vals, kde_triangle_versicolor, label="Kde_Triangle_Versicolor")
#KDE Plot Epanechnikov
plt.plot(x_vals, kde_epa_setosa, label="Kde_Triangle_Setosa")
plt.plot(x_vals, kde_epa_versicolor, label="Kde_Triangle_Versicolor")
plt.legend()
plt.xlabel('Wartości')
plt.ylabel('Gęstość prawdopodobieństwa')
plt.show()

#wnioski
# plt.ylim(top=...) ustawia górny limit wykresu co ułatwia jego czytanie

# Jądro Epanechnikova i Triangle ( przy odpowiednie dobranej szerokości jądra )
# lepiej obrazują spefcyfikę przyjętych danych
# rozkłady ich pradwopodbieństw są mniej wygładzone dla Setosy i Versicolor

# Szczególnie można zauważyć jak istotne jest odpowiednie przyjęcie jądra przy danych VersiColor
# dla h < 0.35 można dostrzec specyfikę i ekstrema lokalne
# dla h > 0.35 wszystkie rozkłady są znacznie bardziej wygładzone

# Dane Setosa przypominają rozkład gaussa już na histogramie, więc w zależoności od jądra,
# rozkład jest mniej lub bardziej wygładzony, przy czym Gauss zawsze jest najbardziej