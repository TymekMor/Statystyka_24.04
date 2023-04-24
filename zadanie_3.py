import numpy as np
import matplotlib.pyplot as plt

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

# przykładowe dane
data = np.random.randn(100)

# punkty na osi x, dla których będzie obliczana gęstość
x_vals = np.linspace(-5, 5, 200)

# szerokość jądra
h = 1

# obliczenie gęstości dla trzech różnych jąder
kde_tri = kde(x_vals, data, h, triangle_kernel)
kde_gauss = kde(x_vals, data, h, gaussian_kernel)
kde_epan = kde(x_vals, data, h, epanechnikov_kernel)

# wykres
plt.plot(x_vals, kde_tri, label='trójkątne jądro')
plt.plot(x_vals, kde_gauss, label='Gaussowskie jądro')
plt.plot(x_vals, kde_epan, label='Epanechnikova jądro')
plt.xlabel("Wartości")
plt.ylabel("Gęstość")
plt.legend()
plt.show()

#Wnioski
# Przy liczbie zmiennych >1000 wykres z jądrem trójkątnym i
# epanechnikova praktycznie całkowicie się pokrywają
# Wykres z Jądrem Gaussa jest bardziej wygładzony i symetryczny,
# niezależnie od ilości zmiennych
# Ten przykład obrazuje znaczenie doboru szerokości jądra do naszych danych statystycznych
# Jak bardzo wpływa na jakość rozkładu np: jak przy zbyt małej wartości pojawia się wiele ekstremów lokalnych
# i przy zbyt dużej wartości następuje maskowanie specyficnzych cech rozkładu

