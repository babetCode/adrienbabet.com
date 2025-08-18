---
date: '2025-04-16T10:03:14-06:00'
draft: false
title: 'Chap 6 Problems'
weight: 12
math: true
---

## Problem 1

> ### A.
> You are concerned about your grandmother falling and breaking her hip on the hard wood floor in her house. Your grandmother has a mass of 55 kg. When standing upright her hip is 65 cm. above the ground. After doing some materials testing you conclude that objects come to rest in 0.15 seconds when impacting the floor. If a pelvis fractures at 1500 N of force, is she likely to break her hip if she falls on this floor? (hint: think about what you already know about projectile motion and then consider using impulse-momentum relationships)


Assuming an initial fall velocity of zero, the kinematic equation $ v_f^2 = v_i^2 + 2 \cdot a \cdot d $ becomes
$$ v_f^2 = 2 \cdot 9.81 \text{ m/s}^2 \cdot 0.65 \text{ m}. $$
Evaluating the right hand side and taking the square root gives
$$ v_f = \sqrt{12.753\ \text m^2 /\text s^2} \approx 3.57\text{ m/s}. $$

Therefore, we have an average force of
$$ \frac{55\text{ kg} \cdot 3.57\text{ m/s}}{0.15\text{ s}} = 1309\text{ N}. $$

Since this is less than 1500 N, she is unlikely to break her hip if we assume uniform acceleration. However, if peak force is much greater than average force (such  as the 2x assumed in problem 2 part C), then she is much more likely to sustain a break.

> ### B.
> Based on your answer to the above, what could you do to reduce the risk of her fracturing her pelvis in a fall?

Either decrease her mass, or increse that time over which she decelerates. This could be accomplished either by diet/exercise, or by padded flooring/hip pads.

## Problem 2
> Two people are riding their bikes around campus when suddenly someone walks out in front of them without looking. This causes the cyclists to swerve suddenly, resulting in them crashing their bikes. One individual is wearing a helmet while the other is not. The table below summarizes what you know about these individuals and the collision.
> |                                 | Subject 1 (no helmet) | Subject 2 (helmet)                     |
> |---------------------------------|------------------------|----------------------------------------|
> | **Mass of head**                | 5 kg                  | 6 kg (5 kg head + 1 kg helmet)         |
> | **Height of head before they fell off the bike** | 1.2 meters           | 1.5 meters                            |
> | **Time of impact with the ground (time until head comes to rest)** | 0.01 seconds         | 0.045 seconds                         |

> ### A.
> Based solely on the information in the table above (i.e. at first glance without doing any calculations) which of these cyclist do you hypothesize had a more serious head impact? Justify your answer.

Subject 1, because the ground contact time was very small.

> ### B.
> Determine the average force applied to the head of each cyclist during these impacts.

**For subject 1**, the velocity can be calculated by evaluating
$$ v_f^2 = 2 \cdot 9.81 \text{ m/s}^2 \cdot 1.2 \text{ m}. $$
Evaluating the right hand side and taking the square root gives
$$ v_f = \sqrt{23.544\ \text m^2 /\text s^2} \approx 4.85\text{ m/s}. $$
Therefore, we have
$$ \frac{5\text{ kg} \cdot 4.85\text{ m/s}}{0.01\text{ s}} = \boxed{2425\text{ N}}. $$

**For subject 2**, following the same steps gives
$$ v_f^2 = 2 \cdot 9.81 \text{ m/s}^2 \cdot 1.5 \text{ m}. $$
Evaluating the right hand side and taking the square root gives
$$ v_f = \sqrt{29.43\ \text m^2 /\text s^2} \approx 5.42\text{ m/s}. $$
Therefore, we have
$$ \frac{6\text{ kg} \cdot 5.42\text{ m/s}}{0.045\text{ s}} \approx \boxed{723\text{ N}}. $$

> ### C.
> Assume the peak force applied to the head is double the average force you calculated in part B. Using this peak force, determine the resulting accelerations of the head. Express you answer in g’s (multiples of acceleration due to gravity, i.e. 3g, 10g, 21g, etc...).

**For subject 1**, we have
$$ \frac{2 \cdot 2425\text{ N}}{5 \text{ kg}} = 970 \text{ m/s}^2 \approx \boxed{99 \text{ g}}. $$

**For subject 2**, we have
$$ \frac{2 \cdot 723\text{ N}}{6 \text{ kg}} = 241 \text{ m/s}^2 \approx \boxed{25 \text{ g}}. $$

> ### D.
> The threshold for sustaining a concussion is somewhere around 85g. Based on the values you calculated in part C, are either of these cyclists likely to sustain a concussion from this crash?

While subject 2 is below this 85 g threshold, subjuct 1 is above it. Therefore, subject 1 is likely to sustain a concussion.