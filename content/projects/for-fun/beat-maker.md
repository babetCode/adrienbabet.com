+++
date = '2024-12-26T19:08:56-07:00'
draft = false
title = 'Beat Maker'
+++

{{< rawhtml >}}
<div id="beat-maker" class="mx-auto">
    <div class="flex justify-between items-center mb-6">
        <div class="flex items-center space-x-4">
            <button id="play-pause" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                Play
            </button>
            <div class="flex items-center space-x-2">
                <button id="bpm-decrease" class="px-2 rounded border border-gray-400">-</button>
                <span id="bpm-display">120 BPM</span>
                <button id="bpm-increase" class="px-2 rounded border border-gray-400">+</button>
            </div>
        </div>
    </div>
    
    <div id="sequencer-grid" class="space-y-4">
        <!-- Grid will be dynamically generated -->
    </div>
</div>
<style>
    .step-active { background-color: #3B82F6; }
    .step-inactive { background-color: #E5E7EB; }
    .step-current { border: 2px solid #10B981; }
</style>
<script src="/js/beat-maker.js"></script>
{{< /rawhtml >}}