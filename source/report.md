\clearpage

# Introduction

BE.[@BE]


Ce Bureau d'étude est réalisé sous le logiciel `Matlab`.

-----

# KNN

## Knn_compute_distances_two_loops.m

Dans cette partie on commence par implémenter le code qui mesure la matrice distance entre tout les training et les tests exemples.Par exemple si on a Ntr training exemples et Nte exemples test on obtient une matrice de taille Nte*Ntr ou chaque élément (i,j) est la distance entre le i ème test et le j ème train et ceci via un double boucle for.

Ceci est le code :
\lstinputlisting{../assets/classifier/knn/knn_compute_distances_two_loops.m}

Aprés l'exécution on obtient une matrice de taille 500 * 5000:

Imprime écran



### Image

![Fonctionnement de l'agorithme de récupération des descripteurs](images/screenshot.png)

### Code

\lstinputlisting{../assets/classifier/knn/dataset/get_datasets.m}

### Tableau

+--------+--------+--------+
|        |  non_0 |  non_1 |
+--------+--------+--------+
|  non_0 |  0.000 |  0.109 |
|  non_1 |  0.109 |  0.000 |
+--------+--------+--------+



-----

# SVM

The gradient could not be strictly differentiable, as in our hinge loss case. In a 1D case, a point right before the hinge should have an analytical gradient of 0, while the numerical gradient would be greater than 0.

# Conclusion

Ce bureau d'étude nous a permis de réaliser...


-----