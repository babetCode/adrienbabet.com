---
date: '2025-07-17T15:34:59-04:00'
draft: false
title: 'Hood Plot Example'
---

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

## Analysis Description

The pedestrian hood plot above shows:

- **Line Landmarks**: Key structural lines like the hood edge, windshield line, and A-pillars
- **Area Landmarks**: Impact zones highlighted as filled areas
- **Impact Landmarks**: Critical impact points marked in red
