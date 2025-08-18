---
date: '2025-03-11T08:53:22-06:00'
draft: false
title: 'Chapter 5 Problems'
weight: 10
math: true
---

## 1.

### A.

Taking the average velocity between times $0.033$s and $0.066$s, we have
$$\frac{1.377\text{ units} - 1.221\text{ units}}{0.066\text{ s} - 0.033\text{ s}} = 4.875\text{ units/s}.$$
Assuming these vertical measurements were in meters, the jumper's velocity is $\boxed{4.875\text{ m/s}}.$

### B. 

With a given vertical velocity $v$, a projectile will take $v/g$ seconds to reach its maximum height, during which it will have an average velocity of $v/2$. Therefore, the projectile will rise a distance of
$$\frac vg \cdot \frac v2 = \frac{v^2}{2g}.$$
Plugging in $v = 4.875\text{ m/s}$ and $g = 9.81 \text{ m/s}^2$ gives
$$\frac{(4.875\text{ m/s})^2}{2(9.81 \text{ m/s}^2)} \approx \boxed{1.21\text{ m}}.$$

### C.

The jumper takes off with vertical position of $1.299$ (let's assume this is in meters), so the apex height will be $$1.299\text{ m} + 1.21\text{ m} \approx 2.51\text{ m}.$$
This is higher than the $2.27\text{ m}$ required to clear a $2.24\text{ m}$ bar by $3\text{ cm}$, so **they are high enough to possibly clear the bar.**

## 2.

### A.

Taking the average AP velocity between times $3.015$ and $3.025$, we get
$$\frac{0.86112 - 0.840685}{3.025 - 3.015} = 2.0435.$$
Similarly, in the vertical direction we have
$$\frac{0.917025 - 0.898041}{3.025 - 3.015} = 1.8984.$$
Therefore, the takeoff angle is
$$\arctan\left(\frac{1.8984}{2.0435}\right) = \boxed{42.89^\circ \text{ above horizontal}},$$
and the resultant velocity is
$$\sqrt{1.8984^2 + 2.0435^2} \approx \boxed{2.7892}.$$

### B.

#### What I think you're asking for:

Despite the the week 7 day 2 lecture on differentiating "distance covered" from "displacement" (slides 22 & 23), I assume you're looking for the COM *displacement* with your problem:
> Determine the distance the center of mass will travel during the flight phase, assuming the landing height is the same as the takeoff height.

The since the vertical displacement is given to be zero, we can find this by multiplying the AP velocity by the flight time. Also, the flight time will equal twice the vertical velocity divided by gravity. Therefore, (assuming our data was given in meters and seconds) we have
$$2.0435\text{ m/s} \cdot \frac{2\cdot 1.8984\text{ m/s}}{9.81\text{ m/s}^2} \approx \boxed{0.7909\text{ m}}.$$

#### The actual distance traveled:
The arc length $L$ of a continuously differentiable curve $y=f(x)$ from $x=a$ to $x=b$ is given by
$$L = \int_{a}^{b} \sqrt{1 + \left( \frac{dy}{dx} \right)^2}\ dx,$$
and the trajectory function of a projectile with takeoff speed $s$ and angle $\theta$ is
$$f(x) = x\tan\theta - \frac{gx^2}{2s^2\cos^2\theta}.$$
I'll spare myself the actually calculation since I don't think it's what you're looking for, but the question of "distance the center of mass will travel during the flight phase" could be found by setting $\theta=42.89^\circ$, $s=2.7892\text{ m/s}$, finding the appropriate function $f(x)$ as described, then solving the integral above.

### C.

Assuming that these takeoff distances are in opposite directions despite having the same sign (as advised by Issac's help), the feet will be 40.3 cm posterior to the COM at takeoff and 14.18 cm anterior to the COM on landing. Therefore, if we also maintain our assumption from part B that landing height is the same as the takeoff height, we have the jump distance as
$$0.403\text{ m} + 0.7909\text{ m} + 0.1418\text{ m} = \boxed{1.3357\text{ m}}.$$

