---
date: '2025-04-15T12:04:24-06:00'
draft: false
title: 'Chapter 8 HW'
weight: 13
math: true
---

## 1. Vocabulary

### A.

- **Inertia (Linear Term)**  
Inertia is the property of an object to resist changes in its state of motion (either at rest or moving at a constant velocity). It is directly related to mass—the greater the mass, the greater the inertia.

- **Moment of Inertia (Rotational Inertia)**  
The moment of inertia ($I$) is a measure of an object's resistance to changes in its rotational motion. It depends on both the mass of the object and the distribution of that mass relative to the axis of rotation:  
$$ I = \sum m_i r_i^2 $$  
where $m_i$ is a mass element and $r_i$ is its distance from the axis.

- **Radius of Gyration**  
The radius of gyration ($k$) is a simplified representation of an object's mass distribution about an axis. It is the distance from the axis at which the total mass could be concentrated to produce the same moment of inertia:  
$$ I = M k^2 $$  
where $M$ is the total mass.

- **Parallel Axis Theorem**  
This theorem relates the moment of inertia about any axis to the moment of inertia about a parallel axis passing through the object's center of mass:  
$$ I = I_{cm} + M d^2 $$  
where $I_{cm}$ is the moment of inertia about the center of mass, $M$ is the total mass, and $d$ is the perpendicular distance between the two axes . 

- **Angular Momentum**  
Angular momentum ($L$) is the rotational equivalent of linear momentum, representing the quantity of rotation of an object. It is given by:  
$$ L = I \omega $$  
where: $I$ is the moment of inertia and $\omega$ is the angular velocity.

- **Conservation of Momentum (Angular Version)**  
Just as linear momentum is conserved in a closed system with no external forces, angular momentum is conserved in a system with no external torques:  
$$ L_{initial} = L_{final} $$  
This principle explains phenomena like spinning ice skaters pulling their arms in to spin faster.

- **Force Equilibrium (Review from Linear Motion)**  
An object is in force equilibrium when the net force acting on it is zero, meaning no linear acceleration occurs:  
$$ \sum \vec{F} = 0 $$  
This implies the object is either at rest or moving at constant velocity (Newton’s First Law).

- **Torque Equilibrium**  
An object is in torque equilibrium when the net torque acting on it is zero, meaning no angular acceleration occurs:  
$$ \sum \vec{\tau} = 0 $$  
For complete static equilibrium, both force equilibrium and torque equilibrium must hold.

### B.

| **Linear Motion**               | **Angular Motion**                     | **Formula/Explanation**                          |
|----------------------------------|----------------------------------------|------------------------------------------------|
| **Force (F)**                | **Torque (τ)**                        | $ \tau = r \times F $ (r = lever arm)       |
| **Newton’s 1st Law (Inertia)** | **Rotational Inertia**               | Objects resist changes in rotational motion ($ \omega = \text{constant} $ if $ \sum \tau = 0 $). |
| **Momentum (p = mv)**         | **Angular Momentum (L = Iω)**         | $ L = I \omega $ (I = moment of inertia)     |
| **Conservation of Momentum**  | **Conservation of Angular Momentum**  | If $ \sum \tau_{ext} = 0 $, $ L_{initial} = L_{final} $. |
| **Newton’s 2nd Law (F = ma)** | **Rotational $ \tau = I \alpha $**  | $ \alpha $ = angular acceleration ($ \tau = I \frac{d\omega}{dt} $) |
| **Newton’s 3rd Law (Action-Reaction)** | **Rotational Action-Reaction** | For every torque applied by Object A on Object B, B applies an equal and opposite torque on A. |

## 2. The Moment of Inertia

### A.
The moment of inertia is a measure of an object's resistance to changes in its rotational motion. It depends on both the mass of the object and how that mass is distributed relative to the axis of rotation. The farther the mass is from the axis, the greater the moment of inertia, making it harder to start or stop rotation.

### B.
1. **Point Mass:**  
   $$
   I = mr^2
   $$
   (where $ m $ = mass, $ r $ = distance from axis)  

2. **Solid Cylinder (about central axis):**  
   $$
   I = \frac{1}{2}MR^2
   $$
   (where $ M $ = total mass, $ R $ = radius)  

3. **Thin Rod (rotating about one end):**  
   $$
   I = \frac{1}{3}ML^2
   $$
   (where $ M $ = mass, $ L $ = length of rod)  

### C.

- **a.** Rigid Sporting Implement (e.g., Tennis Racket, Baseball Bat, Golf Club)  
Increase $ I $: Distribute mass farther from the axis (e.g., use a longer or heavier racket head).  
Decrease $ I $: Make the object shorter or move mass closer to the handle (e.g., choke up on a bat).  

- **b.** Non-Rigid Object (e.g., Human Body)  
Increase $ I $: Extend limbs away from the body (e.g., spreading arms and legs).  
Decrease $ I $: Tuck limbs close to the axis (e.g., pulling arms in during a spin).

### D.
Angular momentum ($ L $) is conserved when no external torque acts on a system:  
$$
L = I \omega = \text{constant}.
$$
When the skater pulls their arms in, $ I $ **decreases** (mass is closer to the axis). To conserve $ L $, angular velocity ($ \omega $) must **increase**, causing the skater to spin faster. Therefore, reducing $ I $ by pulling arms in leads to a faster spin to maintain the same $ L $.

## 3. Application to Gymnastics

From the given frontal view, falling off the beam to her right would be a counterclockwise rotation around the beam. Therefore, she should apply a counter-clockwise torque to her arm (out of the page in this frontal plane) so that by action-reaction her arm applies a clock-wise torque back to her body. Given her position, this would likely be achieved via horizontal adduction of the shoulder.


### References
Week 11 Lecture: Angular Kinematics. Montana State University, Department of Food Systems, Nutrition, & Kinesiology, Fall 2024.
