---
date: '2025-03-04T09:57:42-07:00'
draft: false
title: 'Mar4'
math: true
weight: 11
---

## Kinematic equations

### optimizing take-off angle

Start with take-off velocity $v$ with components
$$v = \begin{pmatrix}v_x \\\ v_y\end{pmatrix}.$$
Assuming no air resistance, we know that air-time is
$$\frac{2v_y}g.$$
Therefore, distance will be
$$v_x\left(\frac{2v_y}g\right) = \frac{2v_xv_y}g.$$
Assuming some total speed $s$ and angle from horizontal $\theta$, we have
$$\begin{align*}
v_x = s\cos\theta, \\\
v_y = s\sin\theta, \\\
\end{align*}$$
so distance is
$$\frac{2\cdot s\cos\theta\cdot s\sin\theta}g.$$

Since the double angle identity tells us that $\sin\theta\cos\theta = \sin(2\theta)/2$, this is equivalent to
$$\frac{s^2\cdot\sin(2\theta)}{g}.$$

Since $\sin(2\theta)$ is maximized at $2\theta = \pi/2$ or $\theta = \pi/4$, we see that this distance is maximized for a given speed at the angle $\boxed{\theta = \pi/4.}$
