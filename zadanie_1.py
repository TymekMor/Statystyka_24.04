import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# zadanie 1
# Generowanie dwóch zmiennych losowych
x = np.random.randn(100000)
y = np.random.randn(10)
# Narysowanie KDE plot
sns.set_style('darkgrid')
sns.kdeplot(x, color='red', fill=True, alpha=0.1, label='x')
sns.kdeplot(y, color='blue', fill=True, alpha=0.5, label='y')

# Dodanie etykiet osi i tytułu wykresu

plt.xlabel('Wartości')
plt.ylabel('Gęstość prawdopodobieństwa')
plt.title('Porównanie rozkładów dwóch zmiennych losowych')
plt.show()
#Wnioski
#Wspólny obszar nachodzenia krzywych wskazuje na podobieństwo rozkładów zmiennych losowych.
#Im większa ilość zmiennych tym bardziej rozkład wygląda jak rozkład gaussa