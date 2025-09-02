---
draft: false
date: '2024-10-03'
title: 'Beat Maker'
tags:
  - Miscellaneous
---

Make sick beats and get groovin'

{{< rawhtml >}}
<div id="beat-maker">
    <div class="header">
        <div class="controls">
            <button id="play-pause" class="play-pause-btn">
                Play
            </button>
            <div class="bpm-controls">
                <button id="bpm-decrease" class="bpm-btn">-</button>
                <span id="bpm-display">120 BPM</span>
                <button id="bpm-increase" class="bpm-btn">+</button>
            </div>
        </div>
    </div>
    <div id="sequencer-grid" class="sequencer-grid">
        <!-- Grid will be dynamically generated -->
    </div>
</div>
<style>
    #beat-maker {
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    .controls {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .play-pause-btn {
        background-color: #3B82F6;
        color: #fff;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        border: none;
        cursor: pointer;
    }
    .play-pause-btn:hover {
        background-color: #2563EB;
    }
    .bpm-controls {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .bpm-btn {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #9CA3AF;
        background: #fff;
        cursor: pointer;
        font-size: 1rem;
    }
    .sequencer-grid {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .track-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .track-label {
        width: 60px;
        text-align: right;
        margin-right: 1rem;
        font-weight: 500;
        color: #374151;
        font-size: 1rem;
    }
    .step-btn {
        width: 2rem;
        height: 2rem;
        margin: 0 0.125rem;
        border: 1px solid #d1d5db;
        border-radius: 0.25rem;
        background: #E5E7EB;
        cursor: pointer;
        transition: background-color 0.2s, border-color 0.2s;
        outline: none;
    }
    .step-btn.step-bar {
        border-color: #9CA3AF;
    }
    .step-active {
        background-color: #3B82F6;
    }
    .step-inactive {
        background-color: #E5E7EB;
    }
    .step-current {
        border: 2px solid #10B981;
    }
    .step-active { background-color: #3B82F6; }
    .step-inactive { background-color: #E5E7EB; }
    .step-current { border: 2px solid #10B981; }
</style>
<script src="/js/beat-maker.js"></script>
{{< /rawhtml >}}