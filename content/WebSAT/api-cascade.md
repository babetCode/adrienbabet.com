---
date: '2025-07-17T15:21:50-04:00'
draft: false
title: 'API Cascade Example:'
---

## ButtonIntrusion.vue → Backend API Call

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

## Flow Summary:
**Content → MetricsStore → API Client → URLs → Views → Serializers → Tasks → Services → SAT Library**

The flow goes through Django's URL routing, view classes, serializers for data validation, Celery tasks for async processing, and business logic services that interface with the SAT library for signal analysis calculations.