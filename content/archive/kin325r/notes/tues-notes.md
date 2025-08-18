---
title: "Tuesday Notes (week 2)"
linkTitle: "21 Jan."
date: 2025-01-19T23:15:41-07:00
draft: false
weight: 1
---

## Announcements
- Office hours Mondays 10am - Noon  
    **& Wed. 12-4?**

- Lab due fri. 31st 5pm

- "By zombies" test for passive voice --- if you can add "by zombies" at the end of a sentence, it's probable in the passive voice.

## Plan for day

- Forces
- Adding vectors

leg extension gets bad rep for patella-femoral forces --- what are patella femoral-forces?

- what is a joint reaction force?
- what is a joint contact force?

## What is a force?
> a push or pull on an object

## My attempt to add vectors in [asymptote](http://asymptote.ualberta.ca/)

```
// Define pens and settings
pen forcePen = black + 1.5bp;
pen resultPen = blue + 1.5bp;
real scale = 15; // Scale factor for vector lengths

// Function to draw a single vector with magnitude and angle
void drawVector(pair start, real magnitude, real angle, string label, pen p=forcePen) {
    pair end = start + scale*magnitude*dir(angle);
    draw(start--end, p, Arrow(DefaultHead));
    write(scale*magnitude*dir(angle));
    label("$" + label + "$", end, dir(angle+90));
}

// Function to calculate resultant vector
pair getResultant(real[] magnitudes, real[] directions) {
    pair result = (0,0);
    for(int i = 0; i < magnitudes.length; ++i) {
        result += scale*magnitudes[i]*dir(directions[i]);
    }
    return result;
}

// Example forces with resultant label
real[] magnitudes = {3, 12};
real[] directions = {-90, -100};
string[] labels = {"\mathrm{vector\ 1}", "\mathrm{vector\ 2}"};
string resultLabel = "\mathrm{result}";

// Fixed spacing values
real spacing = 20;  // Space between vectors/operators
real xpos = 50;     // Starting x position for first vector

// Draw individual vectors with plus signs
for(int i = 0; i < magnitudes.length; ++i) {
    // Draw current vector
    drawVector((xpos, 100), magnitudes[i], directions[i], labels[i]);
    
    // Add plus sign if not last vector
    if(i < magnitudes.length - 1) {
         
        xpos += spacing;
             
        if(magnitudes[i] * cos(radians(directions[i])) > 0) {
            xpos += scale * magnitudes[i] * cos(radians(directions[i]));
        }
      
        label("$+$", (xpos, 100));
      
        xpos += spacing;
 
        // Add space for next vector
        if(magnitudes[i + 1] * cos(radians(directions[i + 1])) < 0) {
            xpos += scale * magnitudes[i + 1] * abs(cos(radians(directions[i + 1]))); 
        }
    }
    else if(magnitudes[i] * cos(radians(directions[i])) > 0) {
    	xpos += scale * magnitudes[i] * cos(radians(directions[i]));
    }
}

// Add equals sign
xpos += spacing;
label("$=$", (xpos, 100));

// Calculate and draw the resultant vector
pair result = getResultant(magnitudes, directions);
drawVector((xpos + spacing, 100), length(result)/scale, degrees(angle(result)), resultLabel, resultPen);
```
