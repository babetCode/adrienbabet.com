---
date: '2025-02-27T10:59:02-07:00'
draft: false
title: 'Chapter 4 HW'
weight: 7
math: true
---

## 1.

### A.
<div class="rounded-lg float-left max-w-xs mr-4">
    <img src="/archive/kin325r/images/hw4p1a.jpg" alt=" ">
</div>
<div class="clear-both"></div>

### B.

<div class="rounded-lg float-left max-w-xs mr-4">
    <img src="/archive/kin325r/images/hw4p1b.jpg" alt=" ">
</div>
<div class="clear-both"></div>

### C.

<div class="rounded-lg float-left max-w-s mr-4">
    <img src="/archive/kin325r/images/hw4p1c.jpg" alt=" ">
</div>
<div class="clear-both"></div>
As the person raises their arms, the COM rises. As person squats, the COM lowers, but it rises relative to the torso.

### D.

The gymnast keeps their body extended to maintain a consistent COM position relative to the bar, ensuring stability and momentum. If they bent at the hips, the COM would shift closer, altering the dynamics and probably making the move less elegant or harder to sustain over multiple rotations. A sprinter starting out of the blocks keeps a low and forward-leaning center of mass for a powerful initial push against the blocks with minimal net torque. For a high jumper going over the bar, the center of mass ideally passes just under the bar while their arched body clears it. If the COM were higher, they’d need to jump with even more momentum to avoid knocking the bar which would be inefficient. Lastly, a linebacker in a low defensive stance relies on a low center of mass to stay balanced and resist being pushed back by opponents. Similar to the sprinter, a lower COM allows them to create more horizontal force against the ground with less torque so they’re less likely to fall or be knocked over. 

## 2.

### A.

Given a set of $k$ point masses
$$\left\\{ \begin{pmatrix}x_1 \\\ y_1 \\\ m_1 \end{pmatrix},\ \begin{pmatrix}x_2 \\\ y_2 \\\ m_2 \end{pmatrix},\ ...,\ \begin{pmatrix}x_k \\\ y_k \\\ m_k \end{pmatrix} \right\\}$$
where each mass is located at it's respective coordinates $(x,y)$ with mass $m$, their total center of mass can be calculated using:
$$\left(\frac{\sum_{n=1}^k(x_n\cdot m_n)}{\sum_{n=1}^k(m_n)}, \frac{\sum_{n=1}^k(y_n\cdot m_n)}{\sum_{n=1}^k(m_n)}\right).$$
As we can see, these formulas for the $x$ and $y$ coordinates have the same form, so we can use this same form for any number of dimensions. In other words, the total center of mass is the weighted average of the point masses, with each point contributing proportionally to its mass.

### B.

To apply this to a 1 dimensional example with a set of three point masses
$$\left\\{ \begin{pmatrix}x_1 \\\ m_1 \end{pmatrix},\ \begin{pmatrix}x_2 \\\ m_2 \end{pmatrix},\ \begin{pmatrix}x_3 \\\ m_3 \end{pmatrix} \right\\},$$
we calculate
$$\frac{\sum_{n=1}^3(x_n\cdot m_n)}{\sum_{n=1}^3(m_n)},$$
which expands to
$$\frac{x_1\cdot m_1 + x_2\cdot m_2 + x_3\cdot m_3}{m_1 + m_2 + m_3}.$$

## 3.

### A.

This can be be done with varying degrees of fidelity depending on the application, but one option discussed in class for 2d sagittal plane analysis is a five segment model with a segment each for the: head, arms, torso, thighs, lower legs. So long as the movements analyzed are symetrical across the sagitall plane and require negligible elbow flexion (i.e. a two foot broad jump), this may be sufficient. 

### B.

<div class="rounded-lg float-left max-w-40 mr-4">
    <img src="/archive/kin325r/images/hw4p3b.jpg" alt=" ">
</div>
<div class="rounded-lg float-left max-w-xs mr-4">
    <img src="/archive/kin325r/images/hw4p3b.png" alt=" ">
</div>
<div class="clear-both"></div>


### C.

An individual’s stability depends on both the location of their COM and their base of support because:
- The ground reaction force holding them up must come from inside the base of support, meaning that the COM must be over the base of support for static equilibrium.
- A larger base of support allows for more COM movement while maintaining this stability.
- A lower COM will have a shorter moment arm to the ground when expering horizontal forces, resulting in less torque and more stability.

### D.

<div class="rounded-lg float-left max-w-xs mr-4">
    <img src="/archive/kin325r/images/hw4p3d.jpg" alt=" ">
</div>
<div class="clear-both"></div>

Dynamic stability is the ability of a system to resist perturbations, and at least in the context of sports, typcially includes the ability to maintain control of one's orientation.
