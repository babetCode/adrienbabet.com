---
date: '2025-02-11T15:15:28-07:00'
draft: false
linkTitle: 'Lab3'
title: "Lab 3: Quantitative Video Analysis KIN 325"
weight: 3
---

## Learning Objectives
Upon completion of this lab, students will be able to:
- Analyze the strengths and limitations of 2D video analysis as conducted in common clinical or performance settings and describe alternative 3D video analysis solutions.
- Demonstrate the ability to use emerging markerless motion capture technology to conduct a quantitative video analysis.
- Use appropriate statistics to evaluate the relationship between hip muscle strength and pelvis and hip kinematics during a drop landing.

## Background
There are an estimated 200,000 ACL injuries in the United States each year, about half of which require surgical reconstruction. Once an individual has torn their ACL, they are at a higher risk of sustaining a second injury. Additionally, there is a well-documented progression from ACL repair to knee osteoarthritis. Given these consequences, there has been a large emphasis placed on identifying who is at risk for sustaining an ACL tear and developing strategies for mitigating injury risk.

One such strategy is to screen athletes ahead of time for injury risk using tests such as drop jumps or run-to-cut maneuvers. Poor performance on these screens (i.e., high knee valgus, hip adduction, or internal rotation) can be indicative of a high risk for ACL injury. Often, these poor movement patterns are due to poor neuromuscular control. Thus, it is not uncommon to see muscle weakness, especially in the hip abductor muscles, in individuals who injure their ACL.

In this lab, we will use emerging markerless motion capture technology to quantitatively analyze pelvis and hip angles during a drop jump as well as assess the relationship between hip abductor muscle strength and pelvis and hip movement during the drop jump.

## Before You Begin
Develop several hypotheses related to data collection:
1. Will there be a difference in hip abductor strength between the dominant and non-dominant limbs?  
<span style="color:blue">Yes</span>
2. Will peak contralateral pelvic drop, hip adduction, and hip internal rotation be different between dominant and non-dominant limbs?  
<span style="color:blue">Yes</span>
3. Will there be a relationship between hip abductor muscle strength and peak contralateral pelvic drop, hip adduction, and hip internal rotation during the landing?  
<span style="color:blue">Yes</span>

## Part 1: Assessing Muscle Strength
Use the handheld dynamometer to measure the strength of your participant's hip abductor muscles.
- Position the hip so the long axis of the femur is abducted 15° relative to the line connecting the left and right anterior superior iliac spines.
- Place the dynamometer just proximal to the lateral femoral epicondyle.
- Have your participant perform a maximal isometric contraction while you resist abduction of their hip. Hold the contraction for 3-5 seconds.
- Perform 4 trials on the dominant limb and four trials on the non-dominant limb. Make sure to have a minimum of 30 seconds of rest between trials.

## Part 2: Drop Jump Test – Film 4 Trials on Both Limbs
We will be using the open-source markerless motion capture technology called OpenCap to collect our data. Information can be found here: [OpenCap](https://www.opencap.ai).

1. Click on the best practices link and quickly read through the guidelines for best practices in recording data. Pay particular attention to the sections about camera setup, camera calibration, and data collection.
2. Create an account and log in. You will need to enter a verification code sent to the email address you use to create the account.
3. Click “New session” to start a new capture session.
4. Scan the provided QR code with both iPads. Then place the iPads on the tripods and ensure they are held in place tightly and will not move during the data capture.
5. Follow the provided instructions to calibrate the cameras. You will need to enter your participant’s height and mass and then perform a static calibration trial. This will scale the musculoskeletal model appropriately.
6. Once you have a scaled model, perform four trials of the drop jump on each limb. When the circles next to each trial turn green, they are finished processing. At that point, you will want to download the data from the session to the laptop.

## Part 3: Data Analysis
1. Each file will contain a .sto file that has the 3D joint angle data for that trial. Load each trial and find the peak pelvis contralateral drop, the peak hip adduction, and the peak hip internal rotation. These are the variables you will use for the data analysis.

<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vS2C1p366f9zb1HukcZO5JEChhFZGDnraZRbtqZgR2KG5HCbkctBZH8pAocy57WybVGRVmuNJsnAlLi/pubhtml?gid=0&amp;single=true&amp;widget=true&amp;headers=false" width=100% height=200em></iframe>
