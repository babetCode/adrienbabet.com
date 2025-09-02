---
date: '2025-08-28T15:51:29-06:00'
draft: true
title: 'Cla Games'
---

{{< rawhtml >}}

    <form class="flex flex-col p-4">
        <textarea id="title" name="title" rows="1" class="mb-4 border border-gray-600 rounded-md p-2" placeholder="Title"></textarea>

        <select id="startingPosition" name="startingPosition" class="mb-4 border border-gray-600 rounded-md p-2" onchange="toggleCustomInput(this)">
            <option value="" selected disabled>Select a starting position</option>
            <optgroup label="Standing">
                <option value="standing-disconnected">Disconnected</option>
                <option value="standing-chest-to-back">Chest to back</option>
                <option value="single-leg">Single leg</option>
                <option value="standing-headlock">Front headlock</option>
            </optgroup>
            <optgroup label="One Standing, One Grounded">
                <option value="one-standing-disconnected">Disconnected</option>
                <option value="one-standing-inside-hooks">Inside hooks</option>
                <option value="one-standing-slx">Single leg x</option>
                <option value="one-standing-one-leg-in">One leg in</option
            </optgroup>
            <optgroup label="Both Grounded">
                <option value="sitting-mount">Sitting mount</option>
                <option value="Chest-to-chest-mount">Chest to chest mount</option>
                <option value="side-control">Side control</option>
            </optgroup>
            <optgroup label="Enter Custom">
                <option value="custom">Custom</option>
            </optgroup>
        </select>
        <input type="text" id="customStartingPosition" name="customStartingPosition" class="mb-4 border border-gray-600 rounded-md p-2 hidden" placeholder="Enter custom starting position">

        <textarea id="player-1-goal" name="player-1-goal" rows="2" class="mb-4 border border-gray-600 rounded-md p-2" placeholder="Player 1 Goal"></textarea>

        <textarea id="player-2-goal" name="player-2-goal" rows="2" class="mb-4 border border-gray-600 rounded-md p-2" placeholder="Player 2 Goal"></textarea>

        <textarea id="combined-goal" name="combined-goal" rows="2" class="mb-4 border border-gray-600 rounded-md p-2 hidden" placeholder="Goal"></textarea>

        <div class="mb-4">
            <input type="checkbox" id="sameGoal" name="sameGoal" onchange="togglePlayerGoals(this)">
            <label for="sameGoal">Same goal for both players</label>
        </div>

        <textarea id="notes" name="notes" rows="2" class="mb-4 border border-gray-600 rounded-md p-2" placeholder="Notes"></textarea>

        <script>
            function togglePlayerGoals(checkbox) {
                const player1Goal = document.getElementById('player-1-goal');
                const player2Goal = document.getElementById('player-2-goal');
                const combinedGoal = document.getElementById('combined-goal');

                if (checkbox.checked) {
                    player1Goal.classList.add('hidden');
                    player2Goal.classList.add('hidden');
                    combinedGoal.classList.remove('hidden');
                } else {
                    player1Goal.classList.remove('hidden');
                    player2Goal.classList.remove('hidden');
                    combinedGoal.classList.add('hidden');
                }
            }

            function toggleCustomInput(select) {
                const customInput = document.getElementById('customStartingPosition');
                if (select.value === 'custom') {
                    customInput.classList.remove('hidden');
                } else {
                    customInput.classList.add('hidden');
                }
            }
        </script>
    </form>

{{< /rawhtml >}}
