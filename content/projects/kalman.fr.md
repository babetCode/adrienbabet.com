---
draft: true
date: '2025-01-28'
draft: false
title: 'Filtre de Kalman Étendu pour IMU 6 Axes'
tags:
  - Biomécanique
  - Math
math: true
---

## Vue d'ensemble

**L'étude de la marche humaine est une tâche importante dans les soins de santé, le sport et la réhabilitation.** Cela nous aide à identifier les problèmes de marche, suivre les progrès de récupération, et même concevoir de meilleurs traitements pour des conditions comme l'AVC ou la maladie de Parkinson. **Un excellent outil pour cela est l'Unité de Mesure Inertielle (IMU) à 6 axes.** Ces appareils combinent un accéléromètre 3 axes et un gyroscope 3 axes, et sont très pratiques pour collecter des données sur le mouvement. Vous pouvez les coller sur une chaussure, les attacher à une jambe, ou les intégrer dans un appareil portable pour suivre les mouvements d'une personne.

Mais voici le problème : les accéléromètres et gyroscopes, bien que puissants, ont leurs inconvénients. Les accéléromètres mesurent l'accélération linéaire et peuvent estimer la distance parcourue ou la vitesse, **mais ils sont bruyants et peuvent être perturbés par des secousses rapides ou des vibrations.** Les gyroscopes, qui mesurent la vitesse de rotation, sont excellents pour capturer des mouvements de rotation fluides et précis, comme l'angle d'un pied pendant un pas. Cependant, ils ont tendance à dériver avec le temps, conduisant à de petites erreurs qui s'accumulent en grosses erreurs.

C'est là que la fusion de capteurs a beaucoup à offrir. En combinant les données des deux capteurs, les algorithmes de fusion peuvent équilibrer les forces de chacun pour aider à compenser les faiblesses de l'autre. Les données du gyroscope peuvent stabiliser les lectures bruyantes de l'accéléromètre, et les données de l'accéléromètre peuvent réduire la dérive du gyroscope. **Les algorithmes de fusion de capteurs discutés dans cet article aident cette combinaison à fonctionner de manière fluide**, produisant des données propres et fiables sur la façon dont une personne marche.

Les avantages d'une analyse précise de la marche incluent l'aide à prédire les chutes chez les adultes âgés, le suivi de la récupération d'un patient après une chirurgie, ou même l'ajustement fin des performances athlétiques. Sans fusion de capteurs, les données IMU seraient trop désordonnées ou peu fiables pour remplir ces fonctions. **Continuez pour une plongée profonde dans le fonctionnement de la fusion de capteurs dans le contexte de l'IMU 6 axes.**

## Filtre de Kalman

Le filtre de Kalman est un algorithme récursif utilisé pour estimer l'état de systèmes dynamiques à partir de mesures bruyantes. Une de ses applications est le suivi de l'orientation d'une Unité de Mesure Inertielle (IMU). Les IMU fournissent des données sur l'accélération et la vitesse angulaire mais sont sujettes au bruit et à la dérive dans le temps. Le filtre de Kalman aide à fusionner ces données de capteurs bruyantes avec un modèle prédictif pour estimer plus précisément l'orientation de l'IMU compte tenu des erreurs de mesure. En général, la capacité du filtre de Kalman à combiner les prédictions du système avec les observations en temps réel pour produire des estimations d'état optimales le rend utile dans les systèmes dynamiques où l'état change dans le temps. Il est aussi computationnellement efficace grâce à sa nature récursive, le rendant adapté aux applications en temps réel. Ici, je décris un EKF (filtre de Kalman étendu --- un des nombreux types de filtres de Kalman) pour un IMU 6 axes.

## Équations de Kalman

Le filtre de Kalman estime un vecteur colonne d'état nx1 ($\mathbf x$), basé sur un vecteur colonne de mesure mx1 ($\mathbf z$), en utilisant un modèle de système :
$$
\begin{align*}
\text{matrice de transition d'état} &: \mathbf A  &(& \text{matrice nxn}), \\\\
\text{covariance du bruit de processus} &: \mathbf Q  &(& \text{matrice diagonale nxn}), \\\\
\text{covariance de mesure} &: \mathbf C  &(& \text{matrice mxm}), \\\\
\text{matrice du modèle de mesure} &: \mathbf H  &(& \text{matrice mxn}).
\end{align*}
$$

Après que le modèle de système ait été défini, il y a cinq étapes du filtre de Kalman simple :

0. <u>Définir les valeurs initiales</u>
$$
\begin{align*}
\mathbf x_0 &= \text{état initial} &(& \text{vecteur colonne nx1}), \\\\
\mathbf P_0 &= \text{covariance d'erreur initiale} &(& \text{matrice nxn}).
\end{align*}
$$

1. <u>Prédire l'état et la covariance d'erreur :</u>
$$
\begin{align*}
\mathbf{\bar x_k} &= \mathbf A \mathbf x_{k-1}, \\\\
\mathbf{\bar P_k} &= \mathbf A \mathbf P_{k-1} \mathbf A^T + \mathbf Q.
\end{align*}
$$

2. <u>Calculer le gain de Kalman :</u>
$$
\mathbf K_k = \mathbf{\bar P}_k \mathbf H^T \left(\mathbf H \mathbf{\bar P}_k \mathbf H^T + \mathbf R\right)^{-1}.
$$

3. <u>Calculer l'estimation (équation de mise à jour d'état) :</u>
$$
\mathbf x_k = \mathbf{\bar x}_k + \mathbf K_k \left(\mathbf z_k - \mathbf H \mathbf{\bar x}_k\right).
$$

4. <u>Calculer la covariance d'erreur :</u>
$$
\mathbf P_k = \mathbf{\bar P}_k - \mathbf K_k \mathbf H \mathbf{\bar P}_k.
$$

Les étapes 1-4 sont ensuite répétées pour mettre à jour récursivement avec chaque nouveau $\mathbf z_k$.

*Notes :*
1. *Ici, les notations avec barre dénotent les valeurs prédites avant la mesure.*

2. *Le terme $(\mathbf z_k - \mathbf H \mathbf{\bar x}_k)$ dans l'équation de mise à jour d'état est important, car il représente l'écart entre notre prédiction et notre mesure. En raison de cette importance, il reçoit le nom de "résidu de mesure" ou "innovation".*

Ceci peut être appliqué aux systèmes uni-variés et multi-variés, et la notation n'est malheureusement pas toujours cohérente. Ci-dessous est une explication de Roger Labbe dans son livre [Filtres de Kalman et Bayésiens en Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) (notez que Labbe fait référence à la transition d'état comme $\mathbf F$ plutôt que $\mathbf A$) :

>"[les équations univariées et multivariées]... sont assez similaires.
>
><u>**Prédire**</u>
>
>
>$
\begin{array}{|l|l|l|}
\hline
\text{Univarié} & \text{Univarié} & \text{Multivarié}\\\\
& \text{(forme Kalman)} & \\\\
\hline
\bar \mu = \mu + \mu_{f_x} & \bar x = x + dx & \bar{\mathbf x} = \mathbf{Fx} + \mathbf{Bu}\\\\
\bar\sigma^2 = \sigma_x^2 + \sigma_{f_x}^2 & \bar P = P + Q & \bar{\mathbf P} = \mathbf{FPF}^\mathsf T + \mathbf Q \\\\
\hline
\end{array}
$
>
>Sans s'inquiéter des spécificités de l'algèbre linéaire, nous pouvons voir que :
$\mathbf x,\, \mathbf P$ sont la moyenne et la covariance d'état. Ils correspondent à $x$ et $\sigma^2$.
$\mathbf F$ est la *fonction de transition d'état*. Quand multiplié par $\bf x$ il calcule l'a priori.
$\mathbf Q$ est la covariance de processus. Il correspond à $\sigma^2_{f_x}$.
$\mathbf B$ et $\mathbf u$ sont nouveaux pour nous. Ils nous permettent de modéliser les entrées de contrôle du système.
>
><u>**Mise à jour**</u>
>
>
>$
>\begin{array}{|l|l|l|}
>\hline
>\text{Univarié} & \text{Univarié} & \text{Multivarié}\\\\
>& \text{(forme Kalman)} & \\\\
>\hline
>& y = z - \bar x & \mathbf y = \mathbf z - \mathbf{H\bar x} \\\\
>& K = \frac{\bar P}{\bar P+R}&
>\mathbf K = \mathbf{\bar{P}H}^\mathsf T (\mathbf{H\bar{P}H}^\mathsf T + \mathbf R)^{-1} \\\\
>\mu=\frac{\bar\sigma^2\, \mu_z + \sigma_z^2 \, \bar\mu} {\bar\sigma^2 + \sigma_z^2} & x = \bar x + Ky & \mathbf x = \bar{\mathbf x} + \mathbf{Ky} \\\\
>\sigma^2 = \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2} & P = (1-K)\bar P & \mathbf P = (\mathbf I -\mathbf{KH})\mathbf{\bar{P}} \\\\
>\hline
>\end{array}
>$
>
>$\mathbf H$ est la fonction de mesure. Nous n'avons pas encore vu cela dans ce livre et je l'expliquerai plus tard. Si vous enlevez mentalement $\mathbf H$ des équations, vous devriez pouvoir voir que ces équations sont similaires aussi.
>
>$\mathbf z, \mathbf R$ sont la moyenne de mesure et la covariance de bruit. Ils correspondent à $z$ et $\sigma_z^2$ dans le filtre univarié (j'ai substitué $\mu$ par $x$ pour les équations univariées pour rendre la notation aussi similaire que possible).
>
>$\mathbf y$ et $\mathbf K$ sont le résidu et le gain de Kalman.
>
>Les détails seront différents du filtre univarié car ce sont des vecteurs et des matrices, mais les concepts sont exactement les mêmes :
>-  Utiliser une gaussienne pour représenter notre estimation de l'état et de l'erreur
>-  Utiliser une gaussienne pour représenter la mesure et son erreur
>-  Utiliser une gaussienne pour représenter le modèle de processus
>-  Utiliser le modèle de processus pour prédire l'état suivant (l'a priori)
>-  Former une estimation à mi-chemin entre la mesure et l'a priori
>Votre travail en tant que concepteur sera de concevoir l'état $\left(\mathbf x, \mathbf P\right)$, le processus $\left(\mathbf F, \mathbf Q\right)$, la mesure $\left(\mathbf z, \mathbf R\right)$, et la fonction de mesure $\mathbf H$. Si le système a des entrées de contrôle, comme un robot, vous concevrez aussi $\mathbf B$ et $\mathbf u$."

Essayons d'appliquer ceci à notre problème. Dans les sections suivantes, je décris les mathématiques et le code python que j'ai utilisé pour implémenter le filtre.

### Informations Python
{{< details title="Débuter avec Numpy" closed="true">}}
Python est un langage idéal pour construire un filtre de Kalman grâce à sa simplicité, sa lisibilité, et ses bibliothèques robustes pour l'algèbre linéaire et l'analyse de données.

**Tableaux Numpy**<hr>
NumPy (abréviation de "python numérique") est une bibliothèque python largement utilisée pour le calcul scientifique, spécialement lors de l'exécution de calculs d'algèbre linéaire. Les tableaux NumPy sont des structures de données n-dimensionnelles qui sont bien adaptées pour les manipulations matricielles dans le filtrage de Kalman.

**Multiplication Matricielle avec Numpy**<hr>
Considérez l'expression
$$\begin{bmatrix}6&2&4 \\\ -1&4&3 \\\ -2&9&3\end{bmatrix}
\begin{bmatrix}4 \\\ -2 \\\ 1\end{bmatrix}.$$

Nous pouvons l'évaluer avec NumPy en utilisant :
```py
import numpy as np

# Définir la matrice
matrix = np.array([[6, 2, 4],
                   [-1, 4, 3],
                   [-2, 9, 3]])

# Définir le vecteur
vector = np.array([[4], [-2], [1]])

# Effectuer la multiplication matrice-vecteur
result = matrix @ vector  # Alternativement, utiliser np.dot(matrix, vector)

print(result)
```
Ceci affichera :  
<span style="font-family:monospace">[[ 24]  
&nbsp;[ -9]  
&nbsp;[-23]]</span>

**Équations de Kalman avec NumPy**<hr>
```py
# Prédire l'État et la Covariance d'Erreur
xp = A @ x
Pp = A @ P @ A.T + Q

# Calculer le Gain de Kalman
K = Pp @ H.T @ numpy.linalg.inv(H @ Pp @ H.T + R)

# Mise à Jour d'État
x = xp + K @ (z - H @ xp)

# Calculer la Covariance d'Erreur
P = Pp - (K @ H @ Pp)
```

{{< /details >}}

## Définir la Variable de Mesure

Lors de l'utilisation d'un IMU pour l'analyse de la marche, nous aimerions utiliser les mesures de l'IMU pour calculer l'attaque du talon, le décollage des orteils, et la longueur de pas (et peut-être nous ajouterons la pose des orteils et le décollage du talon si nous nous sentons ambitieux). À tout moment donné $k$, l'IMU nous donnera des données d'accéléromètre le long de ses trois axes locaux. Nous pouvons penser à ces données d'accélération comme un vecteur $\mathbf a^\text{local}$, où au temps $k$, nous avons
$$
\mathbf a^{\text{local}}_k = \begin{bmatrix} a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\ a^{\text{yaw}}_k \end{bmatrix}.
$$

Il nous donnera aussi la vitesse de rotation le long de ces axes locaux que nous pouvons écrire comme
$$
\boldsymbol\omega^{local}_k = \begin{bmatrix} \omega^{\text{pitch}}_k \\\ \omega^{\text{roll}}_k \\\ \omega^{\text{yaw}}_k \end{bmatrix}.
$$

En mettant ceux-ci ensemble, nous pouvons penser à nos mesures comme étant représentées par une variable $\mathbf z$, où au temps $k$ l'IMU nous donne la lecture
$$
\mathbf z_k = \begin{bmatrix} a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\ a^{\text{yaw}}_k \\\ \omega^{\text{pitch}}_k \\\ \omega^{\text{roll}}_k \\\ \omega^{\text{yaw}}_k \end{bmatrix}.
$$

Il est important de garder à l'esprit que ces mesures sont par rapport au repère local de l'IMU, et non au repère mondial.

## Définir la Variable d'État

Pour déterminer quand et comment les événements de marche se produisent, nous aurions besoin de connaître la position et l'orientation de l'IMU dans les axes du repère mondial, comme les axes nord($N$)-est($E$)-bas($D$). De plus, il serait agréable d'avoir la vitesse et l'accélération de l'IMU dans le repère mondial. Pour visualiser cela, nous pourrions assigner des variables à la position, la vitesse linéaire, l'accélération linéaire, l'orientation, et la vitesse angulaire, comme ceci :
$$
\begin{align*}
\mathbf p^{\text{world}}_k &= \begin{bmatrix} p^{\text{N}}_k \\\ p^{\text{E}}_k \\\ p^{\text{D}}_k \end{bmatrix}, \\\\
\mathbf v^{\text{world}}_k &= \begin{bmatrix} v^{\text{N}}_k \\\ v^{\text{E}}_k \\\ v^{\text{D}}_k \end{bmatrix}, \\\\
\mathbf a^{\text{world}}_k &= \begin{bmatrix} a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k \end{bmatrix}, \\\\
\mathbf q^{\text{world}}_k &= \begin{bmatrix} q^0_k \\\ q^1_k \\\ q^2_k \\\ q^3_k \end{bmatrix}, \\\\
\boldsymbol\omega^{\text{world}}_k &= \begin{bmatrix} \omega^{\text{N}}_k \\\ \omega^{\text{E}}_k \\\ \omega^{\text{D}}_k \end{bmatrix}.
\end{align*}
$$

Ici, $\mathbf q_k^\text{world}$ est une représentation vectorielle du quaternion $\left[q^0_k + i\left(q^1_k\right) + j\left(q^2_k\right) + k\left(q^3_k\right)\right]$. J'utilise des quaternions plutôt que des matrices pour représenter l'orientation car ils nous permettent de mettre à jour notre orientation en utilisant la fonction de mise à jour des quaternions
$$
\mathbf q_{k+1} = \mathbf q_k+\frac12dt\cdot\mathbf q_k\otimes\left[0 + i\left(\omega^{\text{N}}_k\right) + j\left(\omega^{\text{E}}_k\right) + k\left(\omega^{\text{D}}_k\right)\right],
$$
fournissant une méthode cohérente pour interpoler les angles entre les pas de temps *($\otimes$ représente le "[produit de Hamilton](https://fr.wikipedia.org/wiki/Quaternion#Produit_de_quaternions)", alias multiplication de quaternions).*

En mettant ceux-ci ensemble, nous pouvons penser à l'état de notre système (au moins les parties qui nous intéressent) comme étant représenté par une variable $\mathbf x$, où au temps $k$ nous estimons que ses propriétés sont
$$
\mathbf x_k = \begin{bmatrix} p^{\text{N}}_k \\\ p^{\text{E}}_k \\\ p^{\text{D}}_k \\\ v^{\text{N}}_k \\\ v^{\text{E}}_k \\\ v^{\text{D}}_k \\\ a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k \\\ q^0_k \\\ q^1_k \\\ q^2_k \\\ q^3_k \\\ \omega^{\text{N}}_k \\\ \omega^{\text{E}}_k \\\ \omega^{\text{D}}_k \end{bmatrix}.
$$

### Informations sur les Quaternions
{{< details-html title="Que sont les Quaternions ?" closed="true" >}}
{{< md >}}

Les quaternions sont des nombres hypercomplexes à quatre dimensions qui offrent une approche puissante pour le suivi d'orientation grâce à leur capacité à représenter des rotations tridimensionnelles. Spécifiquement, un seul quaternion unitaire peut représenter n'importe quelle rotation 3D sans les pièges communs d'autres représentations de rotation, comme le verrouillage de cardan. De plus, les quaternions permettent des calculs efficaces et numériquement stables, particulièrement bénéfiques dans les contextes de suivi en temps réel comme l'analyse de la marche. L'approche basée sur les quaternions simplifie la composition des rotations en encapsulant des transformations rotationnelles complexes dans la multiplication de quaternions, réduisant le risque de dérive et améliorant la précision dans les algorithmes de fusion de capteurs. Cette capacité rend les quaternions particulièrement bien adaptés pour la nature récursive des modèles d'estimation d'état, où les données d'orientation de mesures IMU consécutives doivent être intégrées de manière transparente dans le temps.

Les quaternions supportent aussi des méthodes d'interpolation telles que l'interpolation linéaire sphérique (SLERP), qui préserve le chemin le plus court de rotation et minimise l'erreur, critique dans les applications comme l'analyse de la marche où un suivi précis de l'orientation est nécessaire. Cette combinaison de stabilité, d'efficacité, et de capacité à gérer des rotations continues fait des quaternions un choix optimal pour un suivi d'orientation robuste et fiable dans l'analyse de la marche basée sur IMU.

Les quaternions représentent l'orientation à travers une structure à quatre dimensions qui encode la rotation tridimensionnelle dans un seul quaternion unitaire, communément noté $q = w + xi + yj + zk$, où $w$, $x$, $y$, et $z$ sont des nombres réels et $i$, $j$, et $k$ sont des unités imaginaires. Le produit $ijk$ est défini comme égal à $-1$, et la multiplication entre deux imaginaires suit les règles du tableau ci-dessous, où la colonne de gauche montre le facteur de gauche et la ligne du haut montre le facteur de droite (notez que les produits ne sont *pas* commutatifs) :

| $\boldsymbol\otimes$ | $\textbf{\textit i}$ | $\textbf{\textit j}$ | $\textbf{\textit k}$ |
|----------------------|----------------------|----------------------|----------------------|
| $\textbf{\textit i}$ |  $-1$   |   $k$   |  $-j$   |
| $\textbf{\textit j}$ |  $-k$   |  $-1$   |   $i$   |
| $\textbf{\textit k}$ |   $j$   |  $-i$   |  $-1$   |

La multiplication entre deux quaternions $q_1 = w_1 + x_1i + y_1j + z_1k$ et $q_2 = w_2 + x_2i + y_2j + z_2k$, est définie selon la formule du produit de Hamilton :
$$\begin{align*}
q=q_1 \otimes q_2=&\ \ \left(w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2\right) \\\
&+ \left(w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2\right)i \\\
&+ \left(w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2\right)j \\\
&+ \left(w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2\right)k,
\end{align*}$$
qui n'est aussi *pas* commutative.

La multiplication de quaternions est utile pour représenter les rotations 3D car elle peut facilement calculer une rotation par un angle donné autour d'un axe passant par l'origine. Considérez la rotation d'un point $P$ par un angle de $\theta$ autour d'un axe $A$ qui passe par l'origine. D'abord, nous normalisons le vecteur $\overrightarrow A$ tel que s'il a des composantes $x$, $y$, $z$, alors $x^2+y^2+z^2=1$. Ensuite, nous construisons un quaternion $q$ tel que
$$q = (\cos(\theta/2)+\sin(\theta/2)(xi+yj+zk)),$$
et son inverse $q^{-1}$ tel que
$$q^{-1} = (\cos(\theta/2)-\sin(\theta/2)(xi+yj+zk)).$$
Maintenant, nous faisons un quaternion pour notre point $P$ de sorte que si notre point $P$ avait des coordonnées $a$, $b$, $c$, alors notre quaternion $p$ est défini par
$$p = ai+bj+ck.$$
Maintenant, pour trouver la projection $P^\prime$ après la rotation, nous avons
$$P^\prime = q\otimes p\otimes q^{-1}.$$
Par convention, les quaternions sont utilisés pour représenter l'orientation en représentant la rotation nécessaire pour qu'un objet passe d'une position "neutre" prédéfinie à son orientation actuelle. En d'autres mots, un quaternion de la forme $\left[\cos(\theta/2) + \sin(\theta/2)\left(xi + yj + zk\right)\right]$ représente une rotation depuis le neutre de $\theta$ autour du vecteur $\langle x,y,z \rangle$.

Ce cadre de fonctionnement des quaternions n'est certes pas intuitif. Les vidéos ci-dessous de 3Blue1Brown fournissent une explication plus approfondie de la mécanique, avec des visuels utiles.

Partie 1 :
{{< /md >}}
{{< youtube d4EgbgTm0Bg >}}
{{< md >}}Partie 2 :{{< /md >}}
{{< youtube zjMuIxRvygQ >}}
{{< /details-html >}}

## Traduction Entre les Repères Locaux et Mondiaux

Notre objectif est d'utiliser nos mesures du système de coordonnées local pitch-roll-yaw pour estimer l'état du système en termes de système de coordonnées global. Une difficulté avec le calcul de l'accélération de cette manière est que la direction de la gravité changera à mesure que nos axes locaux tournent, et nos accéléromètres ne seront pas capables de distinguer ce changement de gravité d'un changement d'accélération linéaire. Dans cette section, nous utilisons notre orientation quaternion pour traiter le problème.

À un temps donné $k$, nous aurons l'orientation de l'IMU stockée comme un quaternion $\mathbf q_k$ qui représente la rotation d'une orientation "neutre" à l'orientation actuelle de l'IMU. Pour calculer la direction "bas" à partir de cela, il est plus efficace de convertir ce quaternion en matrice. La matrice de rotation $\mathbf C_k$, définie comme
$$
\mathbf C_k = \begin{bmatrix}
1 - 2\big((q^2_k)^2 + (q^3_k)^2\big) & 2\big(q^1_k q^2_k - q^0_k q^3_k\big) & 2\big(q^1_k q^3_k + q^0_k q^2_k\big) \\\\
2\big(q^1_k q^2_k + q^0_k q^3_k\big) & 1 - 2\big((q^1_k)^2 + (q^3_k)^2\big) & 2\big(q^2_k q^3_k - q^0_k q^1_k\big) \\\\
2\big(q^1_k q^3_k - q^0_k q^2_k\big) & 2\big(q^2_k q^3_k + q^0_k q^1_k\big) & 1 - 2\big((q^1_k)^2 + (q^2_k)^2\big)
\end{bmatrix},
$$
fait tourner un vecteur du repère local au repère mondial. En d'autres mots, nous avons
$$
\begin{align*}
\mathbf a^{\text{world}}_k = \mathbf C_k \cdot \mathbf a^{\text{local}}_k, \\\\
\boldsymbol\omega^{\text{world}}_k = \mathbf C_k \cdot \boldsymbol\omega^{\text{local}}_k.
\end{align*}
$$
De plus, puisque $\mathbf C_k$ est une matrice orthogonale, son inverse doit être égal à sa transposée $\mathbf C^T_k$, et ainsi
$$
\begin{align*}
\mathbf a^{\text{local}}_k = \mathbf C^T_k \cdot \mathbf a^{\text{world}}_k, \\\\
\boldsymbol\omega^{\text{local}}_k = \mathbf C^T_k \cdot \boldsymbol\omega^{\text{world}}_k.
\end{align*}
$$
À cause de cela, nous pouvons calculer l'accélération du repère mondial à partir de nos mesures locales d'une manière qui tient compte de la gravité. Si nous mesurons l'accélération en unités m/s² alors nous aurons toujours
$$\mathbf a^{\text{world}}_k = \begin{bmatrix} a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k \end{bmatrix} = \begin{bmatrix} 0 \\\ 0 \\\ 9,81 \end{bmatrix}.$$

Par conséquent, un capteur stationnaire avec n'importe quels axes pitch-roll-yaw donnés devrait lire
$$
\mathbf a^{\text{local}}_k = \mathbf C^T_k \begin{bmatrix} 0 \\\ 0 \\\ 9,81 \end{bmatrix}.
$$
Par extension, toute déviation de cette valeur signifie que le capteur accélère dans le repère mondial, donc à tout moment $k$ notre capteur devrait lire
$$
\mathbf a^{\text{local}}_k = \mathbf C^T_k \left(\mathbf a^{\text{world}}_k + \begin{bmatrix}0 \\ 0 \\ 9,8\end{bmatrix}\right).
$$
Cela ressemble exactement à ce dont nous avons besoin ! Pour rendre les choses plus concises, nous ajouterons $\mathbf a^{\text{world}}$ et la gravité ensemble en un vecteur, et écrirons
$$
\mathbf a^{\text{local}}_k = \mathbf C^T_k \begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9,8\end{bmatrix}.
$$

Maintenant que nous avons défini notre problème et vu un peu comment nos variables de mesure et d'état se rapportent l'une à l'autre, il est temps de construire un algorithme de fusion de capteurs pour estimer l'état à partir des mesures.

{{< details title="Matrice de Rotation des Quaternions avec Numpy" closed="true" >}}
```py
def c_matrix(quaternion: np.ndarray) -> np.ndarray:
    if len(quaternion) != 4:
        raise ValueError(f"Quaternion de longueur 4 attendu, obtenu {len(quaternion)} à la place")

    quaternion = quaternion.reshape(-1, 1)
    q0 = quaternion[0, 0]
    q1 = quaternion[1, 0]
    q2 = quaternion[2, 0]
    q3 = quaternion[3, 0]

    c = np.array([[1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                  [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
                  [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]])

    return c
```

Comme vérification de base, nous pouvons vérifier que le quaternion identité <span style="font-family:monospace">x[9:13]</span> produit la matrice identité :
```py
quat = x[9:13]
print(c_matrix(quat))
```
Ceci devrait retourner :  
<span style="font-family:monospace">[[1 0 0]  
&nbsp;[0 1 0]  
&nbsp;[0 0 1]]</span>

Plus généralement, notre <a href="/overview/#:~:text=orientation%20en%20utilisant%20la-,fonction%20de%20mise%20à%20jour%20quaternion,-%F0%9D%91%9E">équation de quaternion d'avant</a> nous dit qu'un quaternion de la forme $\left[\cos(\theta/2) + \sin(\theta/2)\left(xi + yj + zk\right)\right]$ représente une rotation de $\theta$ autour du vecteur $\langle x,y,z \rangle$.
{{< /details >}}

## État

L'état du filtre de Kalman est décrit par la variable d'état $\mathbf x$ et la covariance $\mathbf P$. Dans cette section, nous discuterons de comment définir leurs valeurs initiales. Après avoir défini leurs valeurs initiales, notre filtre de Kalman les mettra à jour en interne à chaque pas de temps.

### x

Comme décrit dans la section *"Variable d'État"*, nous voulons que $\mathbf x$ soit un vecteur 16x1. Si nous pouvions placer notre origine à la position initiale de l'IMU, et que nous pouvions être assez certains qu'il serait stationnaire et aligné avec les axes $N$-$E$-$D$ quand nous commençons à enregistrer des données, alors un état initial raisonnable $\mathbf x_0$ pourrait ressembler à :
$$
\mathbf p^{\text{world}}_k = \begin{bmatrix}0 \\\ 0 \\\ 0\end{bmatrix},\ \ 
\mathbf v^{\text{world}}_k = \begin{bmatrix}0 \\\ 0 \\\ 0\end{bmatrix},\ \ 
\mathbf a^{\text{world}}_k = \begin{bmatrix}0 \\\ 0 \\\ 0\end{bmatrix},\ \ 
\mathbf q^{\text{world}}_k = \begin{bmatrix}1 \\\ 0 \\\ 0 \\\ 0\end{bmatrix},\ \ 
\boldsymbol\omega^{\text{world}}_k = \begin{bmatrix}0 \\\ 0 \\\ 0\end{bmatrix},
$$
$$
\implies\mathbf x_0 = \begin{bmatrix}0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 1 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0\end{bmatrix}.
$$

{{< details title="x Initial avec NumPy" closed="true" >}}
```py
x = np.array([[0], [0], [0], # position
              [0], [0], [0], # vitesse
              [0], [0], [0], # accélération
              [1], [0], [0], [0], # quaternion d'orientation
              [0], [0], [0]]) # vitesse de rotation
```
Ceci représente une variable d'état initiale où l'IMU est stationnaire et les axes locaux pitch-roll-yaw s'alignent avec les axes mondiaux nord-est-bas.
{{< /details >}}

### P

La covariance d'état $\mathbf P$ sera une matrice 16x16 (ou 13x13) qui représente la covariance de l'état. Un $\mathbf P_0$ raisonnable serait :
$$
\begin{bmatrix}
\sigma_{p_0^{\text{N}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & \sigma_{p_0^{\text{E}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & \sigma_{p_0^{\text{D}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & \sigma_{v_0^{\text{N}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & \sigma_{v_0^{\text{E}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & \sigma_{v_0^{\text{D}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & \sigma_{a_0^{\text{N}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{a_0^{\text{E}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{a_0^{\text{D}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{q_0^0}^2 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{q_0^1}^2 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{q_0^2}^2 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{q_0^3}^2 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{\omega_0^{\text{N}}}^2 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{\omega_0^{\text{E}}}^2 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{\omega_0^{\text{D}}}^2 \\\\
\end{bmatrix},
$$
où $\sigma^2_{p^N_0}$ est la variance de la position initiale dans la direction nord, et ainsi de suite. Comme règle générale, il vaut mieux surestimer que sous-estimer --- le filtre convergera si $\mathbf P_0$ est trop grand, mais pourrait ne pas converger s'il est trop petit.

## Processus

Le processus du filtre de Kalman est décrit par $\mathbf F$ (la fonction de transition d'état) et $\mathbf Q$ (la covariance du processus).

### F

Puisque nous n'avons pas d'entrées de contrôle prédéterminées ici, nous voulons une matrice 16x16 $\mathbf F$ que nous pouvons multiplier par l'état actuel pour obtenir notre état prédit au pas de temps suivant. Pour visualiser cela, nous voulons une matrice qui satisfasse
$$
\begin{bmatrix}?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\ \end{bmatrix}
\begin{bmatrix}
p^\text N_{k} \\\
p_k^\text E \\\
p_k^\text D \\\
v_k^\text N \\\
v_k^\text E \\\
v_k^\text D \\\
a_k^\text N \\\
a_k^\text E \\\
a_k^\text D \\\
q_k^0 \\\
q_k^1 \\\
q_k^2 \\\
q_k^3 \\\
\omega_k^\text{N} \\\
\omega_k^\text{E} \\\
\omega_k^\text{D} \\\
\end{bmatrix}
= \begin{bmatrix}
p^\text N_{k+1} \\\
p^\text E_{k+1} \\\
p^\text D_{k+1} \\\
v^\text N_{k+1} \\\
v_{k+1}^\text E \\\
v_{k+1}^\text D \\\
a_{k+1}^\text N \\\
a_{k+1}^\text E \\\
a_{k+1}^\text D \\\
q_{k+1}^0 \\\
q_{k+1}^1 \\\
q_{k+1}^2 \\\
q_{k+1}^3 \\\
\omega_{k+1}^\text{N} \\\
\omega_{k+1}^\text{E} \\\
\omega_{k+1}^\text{D} \\\
\end{bmatrix}.
$$

Il y a quelques choses que nous devons faire pour que cela arrive.

#### 1. Mise à Jour de Position

Puisque le temps entre les mesures est $dt$, alors nous voulons qu'il mette à jour la position d'une manière qui satisfasse
$$
\mathbf p_{k+1} = \mathbf p_k + (\mathbf v_k)dt,
$$ 
où $dt$ est le pas de temps entre les mesures. Cela se développe en
$$
\begin{bmatrix}p^\text N_{k+1} \\\ p^\text E_{k+1} \\\ p^\text D_{k+1}\end{bmatrix} = \begin{bmatrix}p^\text N_{k} \\\ p^\text E_{k} \\\ p^\text D_{k}\end{bmatrix} + \begin{bmatrix}v^\text N_{k} \\\ v^\text E_{k} \\\ v^\text D_{k}\end{bmatrix}dt.
$$
Par conséquent, les trois premières lignes de notre matrice seront
$$
\begin{bmatrix}
1&0&0&dt&0&0&0&0&0&0&0&0&0&0&0&0\\\\
0&1&0&0&dt&0&0&0&0&0&0&0&0&0&0&0\\\\
0&0&1&0&0&dt&0&0&0&0&0&0&0&0&0&0
\end{bmatrix}.
$$

#### 2. Mise à Jour de Vitesse

Nous voulons aussi qu'il mette à jour la vitesse d'une manière qui satisfasse
$$
\mathbf v_{k+1} = \mathbf v_k + (\mathbf a_k)dt,
$$
qui se développe de manière similaire en
$$
\begin{bmatrix}v_{k+1}^\text N \\\ v_{k+1}^\text E \\\ v_{k+1}^\text D\end{bmatrix} = \begin{bmatrix}v_{k}^\text N \\\ v_{k}^\text E \\\ v_{k}^\text D\end{bmatrix} + \begin{bmatrix}a_{k}^\text N \\\ a_{k}^\text E \\\ a_{k}^\text D\end{bmatrix} dt,
$$
et nous donne de manière similaire les trois lignes suivantes de notre matrice
$$
\begin{bmatrix}
0&0&0&1&0&0&dt&0&0&0&0&0&0&0&0&0\\\\
0&0&0&0&1&0&0&dt&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&1&0&0&dt&0&0&0&0&0&0&0
\end{bmatrix}.
$$

#### 3. Mise à Jour d'Accélération

Pour garder les choses simples, nous ne prédirons pas de changement ici, donc nous utiliserons
$$
\mathbf a_{k+1} = \mathbf a_k.
$$
Par conséquent, les trois lignes suivantes de notre matrice seront
$$
\begin{bmatrix}
0&0&0&0&0&0&1&0&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&1&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&0&1&0&0&0&0&0&0&0
\end{bmatrix}.
$$

#### 4. Mise à Jour d'Orientation

Nous voulons qu'il mette à jour l'orientation d'une manière qui satisfasse la fonction de mise à jour des quaternions
$$
\mathbf q_{k+1} = \mathbf q_k+\frac12dt\cdot\mathbf q_k\otimes\begin{bmatrix}0 \\ \omega x_k \\ \omega y_k \\ \omega z_k\end{bmatrix}.
$$
Satisfaire notre équation de mise à jour de rotation. Commençons par développer notre terme de multiplication de quaternions.

Nous savons que le produit de deux quaternions
$$
\mathbf q_1 = (w_1 + x_1i + y_1j + z_1k)
$$
et
$$
\mathbf q_2 = (w_2 + x_2i + y_2j + z_2k)
$$
est calculé en utilisant la formule :
$$
\begin{align*}
\mathbf q=\mathbf q_1 \otimes \mathbf q_2=&\ \ \ \left(w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2\right) \\\\
&+ \left(w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2\right)i \\\\
&+ \left(w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2\right)j \\\\
&+ \left(w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2\right)k.
\end{align*}
$$
En substituant
$$
\begin{align*}
\begin{bmatrix}w_1\\ x_1\\ y_1\\ z_1\end{bmatrix} &= \begin{bmatrix}q^0_k\\ q^1_k\\ q^2_k\\ q^3_k\end{bmatrix},\\\\
\begin{bmatrix}w_2\\ x_2\\ y_2\\ z_2\end{bmatrix} &= \begin{bmatrix}0 \\ \omega^N_k \\ \omega^E_k \\ \omega^D_k\end{bmatrix},
\end{align*}
$$
donne
$$
\begin{align*}
\mathbf q_k\otimes\begin{bmatrix}0 \\ \omega^N_k \\ \omega^E_k \\ \omega^D_k\end{bmatrix} &= \begin{bmatrix}q^0_k\\ q^1_k\\ q^2_k\\ q^3_k\end{bmatrix}\otimes\begin{bmatrix}0 \\ \omega^N_k \\ \omega^E_k \\ \omega^D_k\end{bmatrix} \\\\
&=\ \ \ \left(q^0_k0 - q^1_k\omega^N_k - q^2_k\omega^E_k - q^3_k\omega^D_k\right)  \\\\
&\ \ \ \ + \left(q^0_k\omega^N_k + q^1_k0 + q^2_k\omega^D_k - q^3_k\omega^E_k\right)i  \\\\
&\ \ \ \ + \left(q^0_k\omega^E_k - q^1_k\omega^D_k + q^2_k0 + q^3_k\omega^N_k\right)j  \\\\
&\ \ \ \ + \left(q^0_k\omega^D_k + q^1_k\omega^E_k - q^2_k\omega^N_k + q^3_k0\right)k.
\end{align*}
$$
En écrivant ce résultat sous forme vectorielle, nous avons
$$
\begin{bmatrix}
(q^0_k)(0) &- (q^1_k)(\omega^N_k) &- (q^2_k)(\omega^E_k) &- (q^3_k)(\omega^D_k)\\\\
(q^0_k)(\omega^N_k) &+ (q^1_k)(0) &+ (q^2_k)(\omega^D_k) &- (q^3_k)(\omega^E_k)\\\\
(q^0_k)(\omega^E_k) &- (q^1_k)(\omega^D_k) &+ (q^2_k)(0) &+ (q^3_k)(\omega^N_k)\\\\
(q^0_k)(\omega^D_k) &+ (q^1_k)(\omega^E_k) &- (q^2_k)(\omega^N_k) &+ (q^3_k)(0)
\end{bmatrix}.
$$
Nous voyons que chaque composante est de la forme $[a(q0_k)+b(q1_k)+c(q2_k)+d(q3_k)]$, pour certaines constantes $a$, $b$, $c$, et $d$. Cela ressemble beaucoup à la forme que nous aimerions pour notre matrice de transition d'état ! Nous pouvons substituer
$$
\mathbf q_k\otimes\begin{bmatrix}0 \\ \omega_x \\ \omega_y \\ \omega_z\end{bmatrix} = \begin{bmatrix}
(q^0_k)(0) &- (q^1_k)(\omega^N_k) &- (q^2_k)(\omega^E_k) &- (q^3_k)(\omega^D_k)\\\\
(q^0_k)(\omega^N_k) &+ (q^1_k)(0) &+ (q^2_k)(\omega^D_k) &- (q^3_k)(\omega^E_k)\\\\
(q^0_k)(\omega^E_k) &- (q^1_k)(\omega^D_k) &+ (q^2_k)(0) &+ (q^3_k)(\omega^N_k)\\\\
(q^0_k)(\omega^D_k) &+ (q^1_k)(\omega^E_k) &- (q^2_k)(\omega^N_k) &+ (q^3_k)(0)
\end{bmatrix}
$$
dans notre équation de mise à jour de rotation pour obtenir
$$
\mathbf q_{k+1} = \mathbf q_k+\frac12dt\cdot
\begin{bmatrix}
(q^0_k)(0) &- (q^1_k)(\omega^N_k) &- (q^2_k)(\omega^E_k) &- (q^3_k)(\omega^D_k)\\\\
(q^0_k)(\omega^N_k) &+ (q^1_k)(0) &+ (q^2_k)(\omega^D_k) &- (q^3_k)(\omega^E_k)\\\\
(q^0_k)(\omega^E_k) &- (q^1_k)(\omega^D_k) &+ (q^2_k)(0) &+ (q^3_k)(\omega^N_k)\\\\
(q^0_k)(\omega^D_k) &+ (q^1_k)(\omega^E_k) &- (q^2_k)(\omega^N_k) &+ (q^3_k)(0)
\end{bmatrix}.
$$

En écrivant tout le côté droit comme un vecteur donne
$$
\mathbf q_{k+1} = \begin{bmatrix}
q^0_k + (dt/2)((q^0_k)(0) - (q^1_k)(\omega^N_k) - (q^2_k)(\omega^E_k) - (q^3_k)(\omega^D_k)) \\\\
q^1_k + (dt/2)((q^0_k)(\omega^N_k) + (q^1_k)(0) + (q^2_k)(\omega^D_k) - (q^3_k)(\omega^E_k)) \\\\
q^2_k + (dt/2)((q^0_k)(\omega^E_k) - (q^1_k)(\omega^D_k) + (q^2_k)(0) + (q^3_k)(\omega^N_k)) \\\\
q^3_k + (dt/2)((q^0_k)(\omega^D_k) + (q^1_k)(\omega^E_k) - (q^2_k)(\omega^N_k) + (q^3_k)(0))
\end{bmatrix}.
$$
Par conséquent, les quatre lignes suivantes de notre matrice seront
$$
\begin{bmatrix}
0&0&0&0&0&0&0&0&0&1&-(dt\cdot\omega^N_k)/2&-(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^D_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^N_k)/2&1&(dt\cdot\omega^D_k)/2&-(dt\cdot\omega^E_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^D_k)/2&1&(dt\cdot\omega^N_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^D_k)/2&(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^N_k)/2&1&0&0&0
\end{bmatrix}.
$$

#### 5. Mise à Jour de Vitesse Angulaire

Nous garderons les choses simples ici aussi en laissant
$$
\boldsymbol\omega_{k+1} = \boldsymbol\omega_k.
$$
Par conséquent, les trois dernières lignes de notre matrice seront
$$
\begin{bmatrix}
0&0&0&0&0&0&0&0&0&0&0&0&0&1&0&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&1&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&0&1
\end{bmatrix}.
$$

Nous avons maintenant la matrice $\mathbf F$ :
$$
\begin{bmatrix}
1&0&0&dt&0&0&0&0&0&0&0&0&0&0&0&0\\\\
0&1&0&0&dt&0&0&0&0&0&0&0&0&0&0&0\\\\
0&0&1&0&0&dt&0&0&0&0&0&0&0&0&0&0\\\\
0&0&0&1&0&0&dt&0&0&0&0&0&0&0&0&0\\\\
0&0&0&0&1&0&0&dt&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&1&0&0&dt&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&1&0&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&1&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&0&1&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&0&0&1&-(dt\cdot\omega^N_k)/2&-(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^D_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^N_k)/2&1&(dt\cdot\omega^D_k)/2&-(dt\cdot\omega^E_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^D_k)/2&1&(dt\cdot\omega^N_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^D_k)/2&(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^N_k)/2&1&0&0&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&1&0&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&1&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&0&1
\end{bmatrix}.
$$

*Notes :*  
$\bullet$ *Nous avons pu écrire les lignes 10-13 de cette matrice en utilisant notre observation que chaque composante de $\mathbf q_{k+1}$ était de la forme $[a(q^0_k)+b(q^1_k)+c(q^2_k)+d(q^3_k)]$, mais nous aurions aussi pu trouver la forme de la $n^\text{ème}$ composante comme étant $[q^n + a(\omega^N_k)+b(\omega^E_k)+c(\omega^D_k)]$. Cette seconde forme nous permettra d'écrire une seconde matrice équivalente. Comme exercice intellectuel, vous pouvez vérifier cette mathématique en construisant ce filtre avec la seconde matrice et en vérifiant qu'il produit les mêmes résultats.*  
$\bullet$ *Cette matrice de transition d'état suppose que l'accélération et la vitesse angulaire au temps $t_{k+1}$ seront approximativement égales à celles au temps $t_k$, ce qui est une limitation de ce filtre pendant les événements saccadés comme l'attaque du talon.*

<br>
{{< details title="Matrice F avec NumPy" closed="true" >}}
```py
dt = .01 # Ajuster au taux de données selon les besoins

wN = x[13]
wE = x[14]
wD = x[15]

F_upper_left = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, dt, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, dt, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, dt, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, dt, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, dt]])

F_lower_right = np.array([[1, -dt*wN/2, -dt*wE/2, -dt*wD/2],
                          [dt*wN/2, 1, dt*wD/2, -dt*wE/2],
                          [dt*wE/2, -dt*wD/2, 1, dt*wN/2],
                          [dt*wD/2, dt*wE/2, -dt*wN/2, 1]])

F = np.eye(16)
F[:6,:9] = F_upper_left
F[9:13,9:13] = F_lower_right
```
{{< /details >}}

### Q

La matrice de covariance de bruit de processus $\mathbf Q$ représente les incertitudes dans la dynamique du système. Pour notre vecteur d'état, cela ressemblerait à
$$
\mathbf Q =
\begin{bmatrix}
\sigma^2_{p^{\text{N}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & \sigma^2_{p^{\text{E}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & \sigma^2_{p^{\text{D}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & \sigma^2_{v^{\text{N}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & \sigma^2_{v^{\text{E}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & \sigma^2_{v^{\text{D}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{a^{\text{N}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{a^{\text{E}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{a^{\text{D}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{q^0} & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{q^1} & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{q^2} & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{q^3} & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{N}}} & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{E}}} & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{D}}} 
\end{bmatrix},
$$
Où : $\sigma^2_{p^\text N}$ est la variance dans la position de l'axe nord, et ainsi de suite. Les valeurs spécifiques pour ces variances dépendraient des caractéristiques du système et du bruit de processus attendu.

## Mesure

La mesure du filtre de Kalman est décrite par la moyenne de mesure $\mathbf z$, et la covariance de bruit $\mathbf R$.

### z

Comme décrit dans la section *"Variable de Mesure"*, nous aurons
$$
\mathbf z_k = \begin{bmatrix} a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\ a^{\text{yaw}}_k \\\ \omega^{\text{pitch}}_k \\\ \omega^{\text{roll}}_k \\\ \omega^{\text{yaw}}_k \end{bmatrix}.
$$

{{< details title="z Initial avec NumPy" closed="true" >}}
```py
z = np.array([[0.], [0.], [0.], # accélération
              [0.], [0.], [0.]]) # vitesse de rotation
```
{{< /details >}}

### R

Puisque $\mathbf z$ est un vecteur 6x1, $\mathbf R$ sera une matrice 6x6 représentant la covariance de bruit de nos mesures. Un $\mathbf R$ raisonnable serait :
$$
\mathbf R =
\begin{bmatrix}
\sigma^2_{a^{\text{pitch}}} & 0 & 0 & 0 & 0 & 0 \\\\
0 & \sigma^2_{a^{\text{roll}}} & 0 & 0 & 0 & 0 \\\\
0 & 0 & \sigma^2_{a^{\text{yaw}}} & 0 & 0 & 0 \\\\
0 & 0 & 0 & \sigma^2_{\omega^{\text{pitch}}} & 0 & 0 \\\\
0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{roll}}} & 0\\\\
0 & 0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{yaw}}}
\end{bmatrix},
$$
où $\sigma^2_{a^{\text{pitch}}}$ est la variance dans les mesures d'accélération de pitch, et ainsi de suite. Si nous nous attendons à ce que les accéléromètres et gyroscopes aient la même variance dans toutes les directions, nous pouvons choisir d'utiliser une seule valeur pour $\sigma^2_a$ et une seule valeur pour $\sigma^2_{\omega}$.

## Fonction de Mesure : H

Nos formes données de $\mathbf x_k$ et $\mathbf z_k$ signifient que nous aurons une fonction de mesure 6x16 $\mathbf H$ telle que
$$
\mathbf y_k = \mathbf z_k - \left(\mathbf H \mathbf\cdot\mathbf x_k\right).
$$

Pour visualiser cela sous forme développée, nous voulons une matrice $\mathbf H$ qui satisfasse
$$
\mathbf y_k = \begin{bmatrix} a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\ a^{\text{yaw}}_k \\\ \omega^{\text{pitch}}_k \\\ \omega^{\text{roll}}_k \\\ \omega^{\text{yaw}}_k \end{bmatrix} -
\left(\begin{bmatrix}?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\end{bmatrix}
\begin{bmatrix} p^\text{N}_k\\\\ p^\text{E}_k\\\\ p^\text{D}_k\\\\ v^\text{N}_k\\\\ v^\text{E}_k\\\\ v^\text{D}_k\\\\ a^\text{N}_k\\\\ a^\text{E}_k\\\\ a^\text{D}_k\\\\ q^0_k\\\\ q^1_k\\\\ q^2_k\\\\ q^3_k\\\\ \omega^\text{N}_k\\\\ \omega^\text{E}_k\\\\ \omega^\text{D}_k \end{bmatrix}\right).
$$
Nous voyons que les trois premières lignes seront chacune multipliées scalaire avec $\mathbf x_k$ pour obtenir l'accélération locale, et les trois lignes suivantes seront multipliées scalaire avec $\mathbf x_k$ pour obtenir la vitesse de rotation locale. En utilisant notre matrice $\mathbf C$ de la section *"Traduction Entre les Repères Locaux et Mondiaux"*, nous avons
$$
\mathbf a^{\text{local}}_k = \mathbf C^T_k \begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9,8\end{bmatrix}.
$$
Cela se développe en
$$
\begin{bmatrix}a^{\text{pitch}}_k \\\\ a^{\text{roll}}_k \\\\ a^{\text{yaw}}_k\end{bmatrix} =
\begin{bmatrix}
1 - 2\big((q^2_k)^2 + (q^3_k)^2\big) & 2\big(q^1_k q^2_k - q^0_k q^3_k\big) & 2\big(q^1_k q^3_k + q^0_k q^2_k\big) \\\\
2\big(q^1_k q^2_k + q^0_k q^3_k\big) & 1 - 2\big((q^1_k)^2 + (q^3_k)^2\big) & 2\big(q^2_k q^3_k - q^0_k q^1_k\big) \\\\
2\big(q^1_k q^3_k - q^0_k q^2_k\big) & 2\big(q^2_k q^3_k + q^0_k q^1_k\big) & 1 - 2\big((q^1_k)^2 + (q^2_k)^2\big)
\end{bmatrix}^T
\begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9,8\end{bmatrix}.
$$
Pour simplifier les choses, définissons
$$
\begin{align*}
c^0_k &= 1 - 2\big((q^2_k)^2 + (q^3_k)^2\big) \\\\
c^1_k &= 2\big(q^1_k q^2_k - q^0_k q^3_k\big) \\\\
c^2_k &= 2\big(q^1_k q^3_k + q^0_k q^2_k\big) \\\\
c^3_k &= 2\big(q^1_k q^2_k + q^0_k q^3_k\big) \\\\
c^4_k &= 1 - 2\big((q^1_k)^2 + (q^3_k)^2\big) \\\\
c^5_k &= 2\big(q^2_k q^3_k - q^0_k q^1_k\big) \\\\
c^6_k &= 2\big(q^1_k q^3_k - q^0_k q^2_k\big) \\\\
c^7_k &= 2\big(q^2_k q^3_k + q^0_k q^1_k\big) \\\\
c^8_k &= 1 - 2\big((q^1_k)^2 + (q^2_k)^2\big),
\end{align*}
$$
pour que nous puissions écrire
$$
\begin{bmatrix}a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\\ a^{\text{yaw}}_k\end{bmatrix} =
\begin{bmatrix}
c^0_k & c^1_k & c^2_k \\\
c^3_k & c^4_k & c^5_k \\\
c^6_k & c^7_k & c^8_k
\end{bmatrix}^T
\begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9,8\end{bmatrix}
= \begin{bmatrix}
c^0_k & c^3_k & c^6_k \\\
c^1_k & c^4_k & c^7_k \\\
c^2_k & c^5_k & c^8_k
\end{bmatrix}
\begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9,8\end{bmatrix}.
$$
À partir de là, nous pouvons commencer à remplir les trois premières lignes de $\mathbf H$ :
$$
\mathbf H = \begin{bmatrix}
0&0&0&0&0&0&c^0_k&c^3_k&\left(c^6_k + 9,8c^6_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&c^1_k&c^4_k&\left(c^7_k + 9,8c^7_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&c^2_k&c^5_k&\left(c^8_k + 9,8c^8_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?
\end{bmatrix}.
$$

Les trois lignes du bas seront assez similaires. Pour les trouver, nous utiliserons
$$
\boldsymbol\omega^{\text{local}}_k = \mathbf C^T_k \cdot \boldsymbol\omega^{\text{world}}_k,
$$
Qui se développe en 
$$
\begin{bmatrix}\omega^{\text{pitch}}_k \\\\ \omega^{\text{roll}}_k \\\\ \omega^{\text{yaw}}_k\end{bmatrix} =
\begin{bmatrix}
c^0_k & c^3_k & c^6_k \\\\
c^1_k & c^4_k & c^7_k \\\\
c^2_k & c^5_k & c^8_k
\end{bmatrix}
\begin{bmatrix}\omega^{\text N}_k \\\\ \omega^{\text E}_k \\\\ \omega^{\text D}_k\end{bmatrix},
$$
Signifiant que notre matrice devrait ressembler à
$$
\mathbf H =
\begin{bmatrix}
0&0&0&0&0&0&c^0_k&c^3_k&\left(c^6_k + 9,8c^6_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&c^1_k&c^4_k&\left(c^7_k + 9,8c^7_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&c^2_k&c^5_k&\left(c^8_k + 9,8c^8_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&c^0_k&c^3_k&c^6_k\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&c^1_k&c^4_k&c^7_k\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&c^2_k&c^5_k&c^8_k
\end{bmatrix}.
$$

## Tout Assembler

En utilisant la bibliothèque FilterPy, nous pouvons tout assembler avec

```py
from filterpy.kalman import ExtendedKalmanFilter as EKF

# Paramètres
dt = .001
Q = np.eye(16)
R = np.diag([.5] * 3 + [.9] * 3)
x_0 = np.zeros((16, 1))
x_0[9, 0] = 1.
P_0 = np.eye(16)

def get_F(x, dt):
    wN, wE, wD = [float(x[i, 0]) for i in range(13, 16)]
    F = np.eye(16)
    for i in range(6):
        A[i, i+3] = dt
    bottom = np.array([[1, -dt*wN/2, -dt*wE/2, -dt*wD/2],
                       [dt*wN/2, 1, dt*wD/2, -dt*wE/2],
                       [dt*wE/2, -dt*wD/2, 1, dt*wN/2],
                       [dt*wD/2, dt*wE/2, -dt*wN/2, 1]],
                       dtype=float)
    F[9:13, 9:13] = bottom
    return F

def quat2matrix(q):
    q0, q1, q2, q3 = [float(q[i, 0]) for i in range(4)]
    C = np.array([[1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                  [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
                  [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]])
    return C

def get_H(x):
    C = quat2matrix(x[9:13])
    H = np.zeros((6, 16))
    H[0:3, 6:9] = C.T
    H[3:6, 13:16] = C.T
    return H

def quat_norm(q):
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError('Impossible de normaliser un vecteur zéro')
    return q / norm

# Initialiser l'EKF
ekf = EKF(dim_x=16, dim_z=6)
ekf.Q = Q
ekf.R = R
ekf.x = x_0
ekf.P = P_0
ekf.B = B

columns = ['pN', 'pE', 'pD', 'vN', 'vE', 'vD', 'aN', 'aE', 'aD', 'q0', 'q1', 'q2', 'q3', 'wN', 'wE', 'wD']
predictions = pd.DataFrame(columns=columns)
estimates = pd.DataFrame(columns=columns)
res_columns = ['a_pitch', 'a_roll', 'a_yaw', 'w_pitch', 'w_roll', 'w_yaw']
residuals = pd.DataFrame(columns=res_columns)

# Obtenir les mesures dans un dataframe pandas

for k, measurement in measurements.iterrows():

    # formater la mesure
    measurement = measurement.to_numpy().reshape(-1,1)

    # Étape de prédiction
    ekf.F = get_A(ekf.x, dt)
    ekf.predict()
    ekf.x[9:13] = quat_norm(ekf.x[9:13])
    predictions.loc[len(predictions)] = ekf.x.flatten()

    ekf.F = get_A(ekf.x, dt)
    ekf.update(measurement, HJacobian=get_H, Hx=H_function)
    ekf.x[9:13] = quat_norm(ekf.x[9:13])
    estimates.loc[len(estimates)] = ekf.x.flatten()
    residuals.loc[len(residuals)] = ekf.y.flatten()
```
