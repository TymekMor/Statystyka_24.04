import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def gaussian_kde(x, data, h):
    kde_values = []
    for xi in x:
        kernel_sum = 0
        for di in data:
            kernel_sum += norm.pdf((xi - di) / h)
        kde_values.append(kernel_sum / len(data))
    return kde_values

#W każdym kroku funkcja gaussian_kde iteruje przez wszystkie wartości
# xi z siatki wartości x i dla każdej wartości wylicza sumę wartości funkcji
# gęstości dla wszystkich wartości di w danych data.
# Parametr h określa szerokość jądra gaussa
# i wpływa na wygładzenie funkcji gęstości.

#Przykładowe dane
data = np.random.normal(loc=0,scale=1,size=100)
data2 = np.random.normal(loc=1,scale=2,size=100)
#Siatka wartości
x = np.linspace(-5, 5, 100)
# Wyliczenie KDE dla jądra gausssa
kde_values = gaussian_kde(x, data, h=0.5)
kde_values2 = gaussian_kde(x,data2,h=0.5)
#Wykresy
plt.plot(x, kde_values, color='red', lw=2)
plt.plot(x,kde_values2,color='blue',lw=2) # z poza instrukcji

# Dla standardowego rozkładu normalnego plot Kde wygląda jak rozkład Gaussa
# W innych przypadkach rozkład wygląda zupełnie inaczej dla tego samego h
# dla wystarczająco dużego h wszystko przypomina rozkład gaussa

plt.xlabel('Wartości')

plt.ylabel('Gęstości')

plt.title('Plot Kde z jadrem gaussa')

plt.show()

#Wnioski
#Wykres ten pozwala nam wyestymować z
# jakim prawdopodobieństwiem zmienna losowa przyjmie wartość z danego przedziału.