---
date: '2025-08-01T21:15:24-06:00'
draft: false
title: 'Websat'
tags:
  - Programming
---

In the summer of 2025, I worked with Wake Forest University School of Medicine [Center for Injury Biomechanics](https://school.wakehealth.edu/departments/biomedical-engineering/center-for-injury-biomechanics) under Dr. Scott Gayzik and in collaboration with [Elemance](https://www.elemance.com/) on a novel web based signal analysis tool for vehicle safety testing.

<!--more-->

![Summer Project Group Photo](/images/of-me/wfu-bme/AdrienBabet_AwardWinner.jpg)

## Overview
Newly developed vehicles must undergo and pass a series of rigorous testing procedures to comply with the Federal Motor Vehicle Safety Standards (FMVSS). The National Highway Traffic Safety Administration (NHTSA) is responsible for verifying results and approving vehicles to ensure public safety. As part of this mission, NHTSA is developing a web-based platform for industry and govenment professionals titled WebSAT, for efficient visualization and analysis of vehicle safety testing data.

My project contributed to WebSAT by building a "Hood Top Visualizer"—a component in the website that graphically displays the top surface of a vehicle's hood along with overlaid pedestrian safety test data. These tests evaluate the potential injury outcomes for pedestrians in the event of a collision, making spatial representation of impact zones and injury metrics crucial for interpreting results. The visualizer plots relevant testing metrics such as Head Injury Criterion (HIC) scores at specified impact locations, enabling engineers and analysts to assess vehicle compliance at a glance.

## Project Architecture
The WebSAT project used a containerized architechture with four key tools:
![WebSAT Architechture](/images/websat.png)

## Implementation
At a high level, my implemetation comprised a "Hood Plot" button and an API call cascade: 
```mermaid
flowchart TD
    A[User clicks button] -->|triggers handleClickHoodPlot from:| bhp(ButtonHoodPlot.vue)
    bhp -->|calls loadPedestrianHood from:| ms(MetricsStore.js)
    ms -->|calls loadPedestrianHood from:| api(api.js)
    api -->|calls getAsyncComputationResult to the backend, which is routed by:| url(urls.py)
    url -->|to the pedestrian_hood view in:| views(views.py)
    views -->|calls calculate_pedestrian_hood_task from:| tasks(tasks.py)
    tasks -->|calls calculate_pedestrian_hood from:| services(services.py)
    tasks -->|creates input and output serializer instances from:| serializers(serializers.py)
    services -->|calls dispatch_pedestrian_hood from:| sat(sat.py)
    sat -->|calls calculate_pedestrian_hood from backend| ph(pedestrian_hood.py)
    bhp -->|Creates tab with type 'PedestrianHood'| mc(MainContent.vue)
    mc -->|Maps this tab type to:| phc(PedestrianHoodContent.vue)
    phc -->|Renders| hp(HoodPlot.vue)
    

    classDef yellowNode fill:#fffeca
    classDef greenNode fill:#deebd1
    class ms,api yellowNode
    class bhp,mc,phc,hp greenNode
```
The end result would include a dynamically generated plot similar to this example, containing line landmarks, area landmarks, and impact landmarks:
{{< rawhtml >}}
<!-- Load Plotly.js -->
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<!-- Container for the plot -->
<div id="pedestrian-hood-plot" style="width: 100%; height: 600px;"></div>

<!-- Load our custom script -->
<script src="/js/pedestrian-hood.js"></script>

<script>
// Initialize the plot when the DOM is ready
document.addEventListener('DOMContentLoaded', function() {
  // Load the actual hood data from JSON file
  fetch('/data/test-hood-data.json')
    .then(response => response.json())
    .then(hoodOutput => {
      // Create the plot with the loaded data
      const plot = createPedestrianHoodPlot(
        'pedestrian-hood-plot',
        hoodOutput,
        {
          title: 'Ford F-250 Hood Top',
          xAxisTitle: 'X Position (mm)',
          yAxisTitle: 'Y Position (mm)'
        }
      );
    })
    .catch(error => {
      console.error('Error loading hood data:', error);
      // Fallback to a simple message if data fails to load
      document.getElementById('pedestrian-hood-plot').innerHTML = 
        '<div style="text-align: center; padding: 50px;">Error loading hood data. Please check the console for details.</div>';
    });
});
</script>
{{< /rawhtml >}}
