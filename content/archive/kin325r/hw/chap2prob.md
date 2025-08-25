---
date: '2025-01-31T17:22:37-07:00'
draft: false
title: 'Chapter 2 Problems'
math: true
weight: 4
---

## 1.
{{< details title="Problem Statement 1" closed="true" >}}
![problem 1 PNG](/archive/images/kin325r/chap2p1.png)
{{< /details >}}

**Solution:**

First, let's break down the forces into $x$ and $y$ components. If we let positive $x$ be to the right, then we see that "you" are pushing at an angle of $200^\circ$ counter-clockwise from positive $x$, while "friend" is pusing at an angle of $135^\circ$ counter-clockwise from positive $x$. Therefore, we have
$$\begin{align*}
\text{you}_x &= 350\text N (\cos 200^\circ) \approx -329\text N, \\\\
\text{you}_y &= 350\text N (\sin 200^\circ) \approx -120\text N, \\\\
\text{friend}_x &= 450\text N (\cos 135^\circ) \approx -318\text N, \\\\
\text{friend}_y &= 450\text N (\sin 135^\circ) \approx 318\text N.
\end{align*}$$
From this, we get the resultant force
$$\begin{align*}
\text{F}_x &\approx -329\text N -318\text N = -647\text N, \\\\
\text{F}_y &\approx -120\text N + 318\text N = 198\text N.
\end{align*}$$
Comparing this to the force of friction, we see that a 145 kg mass will produce a weight force of $145 \text{ kg}\left(9.8 \text{ m/s}^2\right) = 1421\text N$, so the maximun magnitude of our friction force will be $0.45(1421\text N) \approx 639 \text N$. Since this is less than our resultant force, $\boxed{\text{yes, it will move.}}$

## 2.
{{< details title="Problem Statement 2" closed="true" >}}
![problem 1 PNG](/archive/images/kin325r/chap2p2.png)
{{< /details >}}

**Solution:**

<div class="rounded-lg float-left max-w-xs mr-4">
    <img src="/archive/images/kin325r/c2p2.jpg" alt="FBD - Runner">
</div>
<br>
We are given the mass of the skier, the angle of the slope, the magnitude and direction of air resistance, the dynamic friction between skis and snow, and are asked to calculate the resultant force assuming the skier is sliding straight down the slope. Let the $x$-axis be parallel to the slope, with the skier traveling in the positive $x$ direction. Now, since we are told that the skier is traveling straight down the slope, we know that the acceleration (and therefore net force) along the $y$-axis must be zero, so we solve for the $x$ compontents of our three forces: gravity, friction, and air resistance.

The force of gravity on the skier will act $30^\circ$ from the $y$-axis in the fourth cartesian quadrant, or $300^\circ$ from the $x$-axis. Therefore, its $x$ component will be
$$60\text{ kg}\left(9.8\text{ m/s}^2\right)\left(\cos 300^\circ\right) = 294\text N.$$
and its $y$ component will be
$$60\text{ kg}\left(9.8\text{ m/s}^2\right)\left(\sin 300^\circ\right) \approx -509.2\text N.$$
Since the normal force from the ground will be equal in magnitude to this $y$ component, the force of surface friction will be
$$0.08(-509.2\text N) \approx -40.7.$$
Combining these with the air friction of $10\text N$ gives the resultant force
$$294\text N - 40.7\text N - 10\text N = 243.3\text N$$
in the positive $x$ direction, or $\boxed{243.3\text N \text{ at }30^\circ \text{ below horizontal}}$ .


## 3.
{{< details title="Problem Statement 3" closed="true" >}}
![problem 1 PNG](/archive/images/kin325r/chap2p3.png)
{{< /details >}}

**Solution:**

<div class="rounded-lg float-left max-w-xs mr-4">
    <img src="/archive/images/kin325r/c2p3-a.jpg" alt="FBD - Runner">
</div>
<br>
Let the $y$-axis be parallel to the tibia, so that the $x$ component of our resultant force equals the shear force, and the $y$ component equals the compressive force. Let the femur be $60^\circ$ counter-clockwise from the positive $x$-axis (so that positive $x$ is posterior and positive $y$ is superior). Now, in degrees counter-clockwise from positive $x$, we have the force of the gastroc at $87^\circ$, force of hamstrings at $60^\circ$, and force of quadriceps at $105^\circ$.<div class="clear-both"></div>
Therefore, the force of the gastroc is
$$\begin{align*}
\text{gastroc}_x &= 780\text N (\cos 87^\circ) \approx 40.8\text N, \\
\text{gastroc}_y &= 780\text N (\sin 87^\circ) \approx 778.9\text N,
\end{align*}$$
the force of the hamstrings is
$$\begin{align*}
\text{hamstrings}_x &= 790\text N (\cos 60^\circ) = 395\text N, \\
\text{hamstrings}_y &= 790\text N (\sin 60^\circ) \approx 684.2\text N,
\end{align*}$$
and the force of the quadriceps is
$$\begin{align*}
\text{quadriceps}_x &= 1650\text N (\cos 105^\circ) \approx -427.1\text N, \\
\text{quadriceps}_y &= 1650\text N (\sin 105^\circ) \approx 1593.8\text N.
\end{align*}$$
Summing these up, we have our resultant force as
$$\begin{align*}
\text{resultant}_x &\approx 40.8\text N + 395\text N - 427\text N = 8.8\text N, \\
\text{resultant}_y &\approx 778.9\text N + 684.2\text N + 1593.8\text N = 3056.9\text N.
\end{align*}$$
Therefore, we have $\boxed{\text{compression} = 3056.9\text N}$ and $\boxed{\text{shear} = 8.8\text N}$ .
