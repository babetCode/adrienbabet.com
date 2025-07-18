---
date: '2025-07-17T15:21:50-04:00'
draft: false
title: 'API Cascade Example:'
---

{{< details-html title="ButtonIntrusion.vue → Backend API Call" closed="true" >}}
{{< md >}}

### 1. Frontend Component Layer
**`ButtonIntrusion.vue`**
- User clicks button → `handleClickVehicleIntrusion()`
- Calls `this.loadIntrusion()` (mapped from `useMetricsStore`)

### 2. Pinia Store Layer
**`MetricsStore.js`**
- `loadIntrusion()` method at line 719
- Calls `this.api.loadIntrusion()` via WebSAT API plugin

### 3. API Client Layer
**`index.js`**
- `WebSatApiPlugin()` provides API access
- API client makes HTTP request to backend

### 4. Django URL Routing
**`urls.py`**
- Routes to `include("core.urls")`

**urls.py**
- Line 97: `path("intrusion", views.IntrusionAnalysis.as_view(), name="intrusion")`

### 5. Django View Layer
**`views.py`**
- `IntrusionAnalysis` class at line 848
- Uses `serializers.IntrusionInputSerializer` for input validation
- Uses `serializers.VehicleIntrusionOutputSerializer` for output formatting
- Calls `tasks.calculate_intrusion_task`

### 6. Django Serializer Layer
**`serializers.py`**
- Input serialization with `IntrusionInputSerializer`
- Output serialization with `VehicleIntrusionOutputSerializer`
- Field validation using `IntrusionPointDataField`

### 7. Celery Task Layer
**`tasks.py`**
- `calculate_intrusion_task()` at line 868
- Calls `services.calculate_intrusion()`
- Returns formatted result at line 883

### 8. Business Logic Layer
**`services.py`**
- `services.calculate_intrusion()` processes the actual intrusion calculations
- Interfaces with the SAT (Signal Analysis Toolkit) library

### 9. WebSocket Response Layer
**`consumers.py`**
- `TaskConsumer` handles WebSocket communication
- `send_chunked_json()` for large responses
- `send_websockets_task_result()` sends results back to frontend

### 10. Frontend Result Processing
**`MetricsStore.js`**
- `addMetric()` stores the result
- Returns processed metric data to component

### 11. UI Display Layer
**`ButtonIntrusion.vue`**
- `gatherResultantIntrusionSummary()` processes results
- `makeIntrusionTab()` creates new tab

**`IntrusionContent.vue`**
- Displays intrusion data in plots and tables
- Uses plotting components for data visualization

### Flow Summary:
**Content → MetricsStore → API Client → URLs → Views → Serializers → Tasks → Services → SAT Library**

The flow goes through Django's URL routing, view classes, serializers for data validation, Celery tasks for async processing, and business logic services that interface with the SAT library for signal analysis calculations.

{{< /md >}}
{{< /details-html >}}

## ButtonHoodPlot.vue → Backend API Call

To create a similar cascade for the ButtonHoodPlot.vue to make a backend call and display PedestrianHoodContent.vue, implement the following components following the same pattern:

### 1. Update MetricsStore.js

Add a `loadPedestrianHood` method to the MetricsStore:

````javascript
// ...existing code...

async loadPedestrianHood(hoodData, testNumber, vehicleNumber) {
  const messageStore = useMessageStore();

  const loadPedestrianHoodAndAddMetric = async (description) => {
    try {
      const pedestrianHoodResult = await messageStore.fetchAsync(
        this.api.loadPedestrianHood(hoodData, vehicleNumber),
        description,
      );

      const metric = this.addMetric(
        "pedestrianHood",
        "vehicle",
        testNumber,
        pedestrianHoodResult,
        null,
        null,
        vehicleNumber,
      );
      return { description, result: metric };
    } catch (error) {
      throw new ErrorInputDescription(
        error.name,
        error.message,
        description,
      );
    }
  };
  return loadPedestrianHoodAndAddMetric(`Pedestrian Hood Vehicle ${vehicleNumber}`);
},
````

### 2. Update urls.py

Add the URL pattern for pedestrian hood analysis:

````python
# ...existing code...

sat_patterns = [
    # ...existing patterns...
    path("pedestrian_hood", views.PedestrianHoodAnalysis.as_view(), name="pedestrian_hood"),
    # ...existing patterns...
]
````

### 3. Add views.py class

Create the view class for pedestrian hood analysis:

````python
# ...existing code...

class PedestrianHoodAnalysis(SATView):
    description = "Performs the pedestrian hood analysis."
    input_serializer = serializers.PedestrianHoodInputSerializer
    output_serializer = serializers.PedestrianHoodOutputSerializer
    task = tasks.calculate_pedestrian_hood_task

    @_sat_schema(description, input_serializer, output_serializer)
    def post(self, request):
        return super().post(request)
````

### 4. Update get_endpoint_configuration

Add the endpoint to the configuration:

````python
# ...existing code...

def get_endpoint_configuration(request):
    context = {}
    # ...existing endpoints...
    
    context.update({
        # ...existing endpoints...
        "loadPedestrianHood": reverse("signal-methods:pedestrian_hood"),
        # ...existing endpoints...
    })

    return JsonResponse(context)
````

### 5. Add serializers.py classes

Create input and output serializers:

````python
# ...existing code...

class PedestrianHoodInputSerializer(serializers.Serializer):
    hood_data = serializers.ListField(child=serializers.DictField())
    vehicle_number = serializers.IntegerField()

class PedestrianHoodOutputSerializer(serializers.Serializer):
    vehicle_number = serializers.IntegerField()
    hood_plot_data = serializers.DictField()
    analysis_results = serializers.DictField()
````

### 6. Add tasks.py function

Create the Celery task:

````python
# ...existing code...

@shared_task
@registry.register_function(
    name="pedestrian_hood",
    input_serializer=serializers.PedestrianHoodInputSerializer,
    output_serializer=serializers.PedestrianHoodOutputSerializer,
)
def calculate_pedestrian_hood_task(validated_data):
    logger.debug("Calculating pedestrian hood: %s", validated_data)

    result = services.calculate_pedestrian_hood(
        validated_data["hood_data"],
        validated_data["vehicle_number"],
    )

    return {
        "vehicle_number": result.vehicle_number,
        "hood_plot_data": result.hood_plot_data,
        "analysis_results": result.analysis_results,
    }
````

### 7. Add services.py function

Create the service function:

````python
# ...existing code...

def calculate_pedestrian_hood(
    hood_data: typing.List[dict],
    vehicle_number: int,
) -> types.PedestrianHoodResult:
    logger.debug(
        "Calculating pedestrian hood for vehicle number %d",
        vehicle_number,
    )

    # Call the SAT library function here
    pedestrian_hood_result = sat.dispatch_pedestrian_hood(
        hood_data=hood_data, 
        vehicle_number=vehicle_number
    )

    return pedestrian_hood_result
````

### 8. Update ButtonHoodPlot.vue

Modify the button to make the backend call:

````vue
<script>
// ...existing imports...
import { useMetricsStore } from "@stores/MetricsStore";
import { useMessageStore } from "@stores/MessageStore";

export default {
    // ...existing code...
    methods: {
        ...mapActions(useTabsUIStore, ["addTab"]),
        ...mapActions(useMetricsStore, ["loadPedestrianHood"]),

        async handleClickHoodPlot() {
            const selectedTestId = this.activeTab.testNumber;
            const vehicleInfo = useVehicleTestsStore()?.vehicleInfoLookup[selectedTestId]?.vehicle_information || [];
            const vehicleNumbers = _.map(vehicleInfo, "vehicle_number");

            // Filter vehicles that have hood data
            const vehicleNumbersWithHoodData = vehicleNumbers.filter(
                (vehicleNumber) => {
                    const vehicleData = useVehicleTestsStore()?.vehicleHoodData[selectedTestId]?.[vehicleNumber];
                    return vehicleData?.data && Array.isArray(vehicleData.data.hood_data) && vehicleData.data.hood_data.length > 0;
                }
            );

            // Create promises for each vehicle
            const hoodMetricsPromises = _.map(vehicleNumbersWithHoodData, (vehicleNumber) => {
                const vehicleData = useVehicleTestsStore()?.vehicleHoodData[selectedTestId]?.[vehicleNumber];
                const hoodDataProvenance = vehicleData.provenance;
                return this.loadPedestrianHood(hoodDataProvenance, selectedTestId, vehicleNumber);
            });

            const hoodMetrics = await Promise.allSettled(hoodMetricsPromises)
                .then((results) => useMessageStore().setCompletionMessage(results));

            const hoodDataSummary = hoodMetrics
                .filter((item) => item.status === "fulfilled")
                .map((item) => {
                    const metricData = item.value?.result?.metricData;
                    const metricID = item.value?.result?.metricID;
                    if (metricData?.hood_plot_data) {
                        return { metricData, metricID };
                    }
                    return null;
                })
                .filter(Boolean);

            this.makeHoodPlotTab(hoodDataSummary);
        },

        makeHoodPlotTab(hoodDataSummary) {
            const tabContent = {
                contentHoodPlotData: hoodDataSummary,
                selectedHoodMetricId: null,
            };

            const tabName = `${this.activeTab.tabContentType[0]}${this.activeTab.testNumber}: Hood Plot`;
            const tabTitle = `${_.capitalize(this.activeTab.tabContentType)} Test ${
                this.activeTab.testNumber
            }: Pedestrian Hood`;

            this.addTab(
                tabName,
                tabTitle,
                "pedestrianHood",
                tabContent,
                this.activeTab.testNumber,
            );

            const injButtonsUIStore = useInjButtonsUIStore();
            injButtonsUIStore.handleTabAdded();
        },
    },
};
</script>
````

### 9. Add types.py (if needed)

Create the result type:

````python
# ...existing code...

@dataclass
class PedestrianHoodResult:
    vehicle_number: int
    hood_plot_data: dict
    analysis_results: dict
````

### 10. Complete the `dispatch_pedestrian_hood` function in sat.py

Make sure `pedestrian_hood` is imported at the top of sat.py

````python
// ...existing code...

def dispatch_pedestrian_hood(
    hood_data: typing.List[dict],
    vehicle_number: int,
) -> types.PedestrianHoodResult:
    pedestrian_hood_input = pedestrian_hood.PedestrianHoodInput(
        hood_data=hood_data,
        vehicle_number=vehicle_number,
    )

    pedestrian_hood_output = pedestrian_hood.calculate_pedestrian_hood(pedestrian_hood_input)

    pedestrian_hood_result = types.PedestrianHoodResult(
        vehicle_number=pedestrian_hood_output.vehicle_number,
        hood_plot_data=pedestrian_hood_output.hood_plot_data,
        analysis_results=pedestrian_hood_output.analysis_results,
    )

    return pedestrian_hood_result

// ...existing code...
````

### 11. Add the API method to api.js

````javascript
// ...existing code...

class WebSatApi {
  constructor(
    csrftoken,
    {
      // ...existing endpoints...
      loadIntrusion,
      loadPedestrianHood,  // Add this line
      loadLoadCellAnalysis,
      // ...existing endpoints...
    },
  ) {
    this.csrftoken = csrftoken;
    this.endpoints = {
      // ...existing endpoints...
      loadIntrusion,
      loadPedestrianHood,  // Add this line
      loadLoadCellAnalysis,
      // ...existing endpoints...
    };
  }

  // ...existing methods...

  // Add this method in the VEHICLE METRICS section
  async loadPedestrianHood(hoodData, vehicleNumber) {
    const data = {
      hood_data: hoodData,
      vehicle_number: vehicleNumber,
    };
    return await this.sendAsyncComputationRequest(
      this.endpoints.loadPedestrianHood,
      data,
    );
  }

  // ...existing code...
}
````

### 12. Update TestsVehicleStore.js to handle hood data

````javascript
// ...existing code...

export const useVehicleTestsStore = defineStore("testsVehicleStore", {
  state: () => ({
    // ...existing state...
    vehicleHoodData: {},  // Add this line
    // ...existing state...
  }),

  actions: {
    // ...existing actions...
    
    addVehicleHoodData(testNumber, vehicleNumber, vehicleHoodData) {
      if (!this.vehicleHoodData[testNumber]) {
        this.vehicleHoodData[testNumber] = {};
      }
      this.vehicleHoodData[testNumber][vehicleNumber] = vehicleHoodData;
    },
    
    // ...existing actions...
  },
});
````

### 13. Update the tab handling in your UI store

Make sure TabsUIStore can handle the new "pedestrianHood" tab type, similar to how it handles "intrusion" tabs.

This implementation follows the exact same pattern as the intrusion analysis, creating a complete pipeline from the frontend button click through to the backend processing and display in the content component. The key differences are:

1. **Data structure**: Adapted for hood plot data instead of intrusion data
2. **Component names**: Changed to reflect pedestrian hood functionality
3. **Backend processing**: Calls the specific SAT library function for hood analysis

The flow will be: ButtonHoodPlot.vue → `MetricsStore.loadPedestrianHood()` → Backend API → PedestrianHoodContent.vue display.