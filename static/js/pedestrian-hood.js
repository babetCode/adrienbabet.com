class PedestrianHoodPlot {
  constructor(containerId, hoodOutput, options = {}) {
    this.containerId = containerId;
    this.hoodOutput = hoodOutput;
    this.title = options.title || "Pedestrian Hood";
    this.xAxisTitle = options.xAxisTitle || "x (mm)";
    this.yAxisTitle = options.yAxisTitle || "y (mm)";
    
    // Default plotly config (simplified version)
    this.config = {
      responsive: true,
      displayModeBar: true,
      scrollZoom: true,
      toImageButtonOptions: {
        filename: "pedestrian_hood_2d",
        format: "png",
        width: 1000,
        height: 600,
        scale: 1
      }
    };
    
    this.init();
  }

  init() {
    if (typeof Plotly === 'undefined') {
      console.error('Plotly.js is required but not loaded');
      return;
    }
    
    this.render();
  }

  generateLineLandmarks() {
    if (!this.hoodOutput || !this.hoodOutput.line_landmarks) {
      return [];
    }

    const traces = [];
    for (const [landmarkName, coordinates] of Object.entries(this.hoodOutput.line_landmarks)) {
      traces.push({
        type: "scatter",
        x: coordinates[0],
        y: coordinates[1],
        mode: "lines+markers",
        name: landmarkName,
        showlegend: true,
        hoverinfo: "name",
      });
    }
    return traces;
  }

  generateAreaLandmarks() {
    if (!this.hoodOutput || !this.hoodOutput.area_landmarks) {
      return [];
    }

    const traces = [];
    for (const [landmarkName, coordinates] of Object.entries(this.hoodOutput.area_landmarks)) {
      traces.push({
        type: "scatter",
        x: coordinates[0],
        y: coordinates[1],
        mode: "lines+markers",
        fill: "toself",
        name: landmarkName,
        showlegend: true,
        hoverinfo: "name",
      });
    }
    return traces;
  }

  generateImpactLandmarks() {
    if (!this.hoodOutput || !this.hoodOutput.impact_landmarks) {
      return [];
    }

    const traces = [];
    for (const [landmarkName, coordinates] of Object.entries(this.hoodOutput.impact_landmarks)) {
      traces.push({
        type: "scatter",
        x: coordinates[0],
        y: coordinates[1],
        mode: "markers",
        name: landmarkName,
        showlegend: true,
        hoverinfo: "name",
        marker: {
          color: "red",
          size: 5,
        },
      });
    }
    return traces;
  }

  getData() {
    return [
      ...this.generateLineLandmarks(),
      ...this.generateAreaLandmarks(),
      ...this.generateImpactLandmarks(),
    ];
  }

  getLayout() {
    return {
      title: this.title,
      xaxis: {
        title: this.xAxisTitle,
        type: "linear",
        scaleanchor: "y",
      },
      yaxis: {
        title: this.yAxisTitle,
      },
      template: "plotly_white",
      autosize: true,
    };
  }

  render() {
    const data = this.getData();
    const layout = this.getLayout();
    
    Plotly.newPlot(this.containerId, data, layout, this.config);
  }

  update(newHoodOutput) {
    this.hoodOutput = newHoodOutput;
    this.render();
  }
}

// Example usage function
function createPedestrianHoodPlot(containerId, hoodOutput, options = {}) {
  return new PedestrianHoodPlot(containerId, hoodOutput, options);
}

// Example data structure for testing
const exampleHoodOutput = {
  line_landmarks: {
    "Hood Edge": [
      [0, 100, 200, 300, 400, 500],
      [0, 50, 75, 100, 125, 150]
    ],
    "Windshield Line": [
      [0, 100, 200, 300, 400, 500],
      [150, 175, 200, 225, 250, 275]
    ]
  },
  area_landmarks: {
    "Impact Zone": [
      [50, 150, 150, 50, 50],
      [50, 50, 150, 150, 50]
    ]
  },
  impact_landmarks: {
    "Impact Point 1": [
      [100],
      [100]
    ],
    "Impact Point 2": [
      [200],
      [125]
    ]
  }
};
