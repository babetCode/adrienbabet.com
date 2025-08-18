---
date: '2025-01-31T17:07:03-07:00'
draft: false
title: 'Vector Addition'
weight: 100
---

```c
pen forcePen = black + 1.5bp;
pen resultPen = blue + 1.5bp;

real scale = 10;
real spacing = 20;  // Space between vectors/operators
real xpos = 0;

real drawVec(real xleft, real magnitude, real angle, string label, pen p=forcePen) {
    real startx;
    if (magnitude * cos(radians(angle)) < 0) {
        startx = xleft - scale*magnitude*dir(angle).x;
    }
    else {
        startx = xleft;
    }
    real starty = -scale*magnitude*dir(angle).y/2;
    pair start = (startx, starty);
    pair end = start + scale*magnitude*dir(angle);
    draw(start--end, p, Arrow(DefaultHead));
    write("start: (" + format("%.2f", startx) + "," + format("%.2f", starty) + ")");
    label(rotate(angle)*("$" + label + "$"), (start+end)/2, dir(angle+90));
    real width = abs(end.x - start.x);
    return width;
}

// Forces (magnitude, direction in deg, label)
real[] magnitudes = {4, 2, 1};
real[] directions = {170, 25, -45};
string[] labels = {"F_1", "F_2", "F_3"};

for (int i = 0; i < magnitudes.length; ++i) {
    real width = drawVec(xpos, magnitudes[i] * scale, directions[i], labels[i]);
    xpos += width + spacing;
    if (i < (magnitudes.length - 1)) {
        label("$+$", (xpos,0));
    }
    else {
        label("$=$", (xpos,0));
    }
    xpos += spacing;
    write("xpos: " + string(xpos));
}

pair getResultant(real[] magnitudes, real[] directions) {
    pair result = (0,0);
    for(int i = 0; i < magnitudes.length; ++i) {
        result += scale * magnitudes[i] * dir(directions[i]);
    }

    // Calculate the magnitude of the resultant vector
    real magnitude = length(result);

    // Calculate the direction (angle) of the resultant vector in degrees
    real direction = degrees(atan2(result.y, result.x));

    // Return the magnitude and direction as a pair
    return (magnitude, direction);
}

pair resultant = getResultant(magnitudes, directions);
real resultantMagnitude = resultant.x;
real resultantDirection = resultant.y;

drawVec(xpos, resultantMagnitude, resultantDirection, "result", p=resultPen);
```

*This code can be copied & pasted into the [asymptote web app](http://asymptote.ualberta.ca/).*