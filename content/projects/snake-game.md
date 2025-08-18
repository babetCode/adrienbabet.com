---
draft: true
date: '2025-01-02T18:24:22-07:00'
title: 'Snake Game'
tags:
  - Games
  - Misc.
---

Play snake, the classic apple eating game!

{{< rawhtml >}}
<div class="flex flex-col items-center">
    <p>Use ←↑↓→ or WASD to control the snake</p>
    
    <!-- Game Canvas -->
    <canvas id="gameCanvas" 
            width="400" 
            height="400" 
            class="border-2 border-grey-500 rounded-lg mb-4">
    </canvas>

    <!-- Game Controls -->
    <div class="flex space-x-4">
        <button id="startButton" 
                class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-400 transition">
            Start Game
        </button>
    </div>

    <!-- Score Display -->
    <div class="mt-4 text-lg font-semibold">
        Score: <span id="scoreDisplay">0</span>
    </div>
</div>

<script defer src="/js/snake.js"></script>
{{< /rawhtml >}}


{{< callout type="warning" >}}
  This game currently requires a keyboard (touch screen swipes are not recognized to control the snake).
{{< /callout >}}
