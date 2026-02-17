#!/usr/bin/env python3
"""
Gamified Neurofeedback for Motor-Imagery BCI
===========================================
Plug this into your *online_mi_pipeline* by importing **GameScene** and calling
`scene.update(cue_label, predicted_label)` right after you compute the
classifier output.

The game:
---------
* A spaceship sits centre-screen.
* Each trial, two portals appear - **LEFT** (blue) and **RIGHT** (red).
* When the classifier fires, the ship zips toward the portal that matches the
  predicted label.
    * If it matches the *cue*, you score +1.
    * If it's wrong, you lose a point and the ship flashes red.
* The current score is always visible top-left.  Nice and simple, but much more
  engaging than a static bar.

How to embed
------------
```python
from gamified_feedback import GameScene
scene = GameScene(window_ref)
...
scene.update(cue_label, pred_label)   # LEFT=1, RIGHT=2
```
The scene lives in its own PsychoPy *Layer* so you can keep your cue text on
 Layer 0 and let the game draw on Layer 1.

Stand-alone demo
----------------
Run this file directly to see a keyboard-driven mock-up:
*Press ← or → to simulate the classifier, ESC to quit.*
"""
from __future__ import annotations

from typing import Tuple

from psychopy import visual, event, core

# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------

class GameScene:
    """Handles gamified MI feedback inside a PsychoPy window."""

    def __init__(self, win: visual.Window):
        self.win = win
        # layers: cue on 0, game on 1
        self.layer = 1
        # spaceship (white triangle)
        self.ship = visual.ShapeStim(
            win,
            vertices=[(-20, -20), (0, 20), (20, -20)],
            fillColor="white",
            lineColor="black",
            pos=(0, 0),
            autoDraw=False,
            opacity=1.0,
            ori=0,
            lineWidth=2,
            depth=self.layer,
        )
        # portals
        self.portal_left = visual.Circle(
            win, radius=40, pos=(-250, 0), fillColor="#4f83ff", lineColor="black",
            autoDraw=False, depth=self.layer
        )
        self.portal_right = visual.Circle(
            win, radius=40, pos=(250, 0), fillColor="#ff4f4f", lineColor="black",
            autoDraw=False, depth=self.layer
        )
        # green highlight background for target portal (hidden by default)
        self.highlight = visual.Circle(
            win,
            radius=60,
            pos=(0, 0),
            fillColor="green",
            lineColor=None,
            autoDraw=False,
            depth=self.layer
        )
        # score text
        self.score = 0
        self.txt_score = visual.TextStim(
            win,
            text="Score: 0",
            pos=(-320, 280),
            color="black",
            height=32,
            bold=True,
            autoDraw=False,
            depth=self.layer,
        )
        # outcome flash timer
        self.flash_timer = core.CountdownTimer(0)
        # fixation cross for baseline periods
        self.fixation = visual.TextStim(
            win,
            text='+',
            height=80,
            color='black',
            autoDraw=False,
            depth=self.layer,
        )
        # percentage texts above portals
        self.txt_left_proba = visual.TextStim(
            win,
            text='0%',
            pos=(-248, 80),
            color='black',
            height=24,
            autoDraw=False,
            depth=self.layer,
        )
        self.txt_right_proba = visual.TextStim(
            win,
            text='0%',
            pos=(252, 80),
            color='black',
            height=24,
            autoDraw=False,
            depth=self.layer,
        )

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------

    def update(self, cue: int, pred: int, proba=None):
        """Move ship, update score, schedule a flash if wrong, and update percentage texts if provided."""
        # update percentage texts if probabilities provided
        if proba is not None:
            self.txt_left_proba.text = f'{proba[0]*100:.1f}%'
            self.txt_right_proba.text = f'{proba[1]*100:.1f}%'
        target_pos = self._pos_from_label(pred)

        if cue == pred:
            self.score += 1
            self.ship.fillColor = "white"
        else:
            # self.score -= 1
            self.ship.fillColor = "red"
            self.flash_timer.reset(0.3)   # flash for 300 ms

        self.txt_score.text = f"Score: {self.score}"
        # reset ship to center for movement animation
        self.ship.pos = (0, 0)

        target_pos = self._pos_from_label(cue)
        self.highlight.pos = target_pos

        # animate movement from center to target over 0.2 seconds
        num_steps = 10
        duration = 0.2  # seconds
        dt = duration / num_steps
        for i in range(1, num_steps + 1):
            frac = i / num_steps
            self.ship.pos = (frac * target_pos[0], frac * target_pos[1])
            self.highlight.draw()
            self.draw()
            self.txt_left_proba.draw()
            self.txt_right_proba.draw()
            self.win.flip()
            core.wait(dt)

        self.highlight.draw()
        self.draw()
        self.txt_left_proba.draw()
        self.txt_right_proba.draw()
        self.win.flip()

    def draw(self):
        """Draw every frame (call once per display flip)."""
        # restore colour after flash period
        if self.flash_timer.getTime() <= 0:
            self.ship.fillColor = "white"

        for stim in (
            self.portal_left,
            self.portal_right,
            self.ship,
            self.txt_score,
        ):
            stim.draw()

    def get_ready(self, cue: int):
        """Reset ship and highlight the target portal before each trial."""
        # reset ship to center
        self.ship.pos = (0, 0)
        # position and draw highlight behind the target portal
        target_pos = self._pos_from_label(cue)
        self.highlight.pos = target_pos
        self.highlight.draw()
        # draw portals, ship, score
        self.draw()
        self.win.flip()

    def baseline(self, duration: float = 1.0):
        """Show fixation cross with current score for the specified duration."""
        # draw fixation and score
        self.fixation.draw()
        self.txt_score.draw()
        self.win.flip()
        core.wait(duration)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pos_from_label(label: int) -> Tuple[int, int]:
        if label == 1:  # LEFT
            return (-250, 0)
        elif label == 2:  # RIGHT
            return (250, 0)
        else:
            return (0, 0)


# -----------------------------------------------------------------------------
# Stand-alone demo (keyboard-driven)
# -----------------------------------------------------------------------------

def demo():
    win = visual.Window((800, 600), color="white", units="pix")

    scene = GameScene(win)

    while True:
        scene.draw()
        win.flip()
        for key in event.getKeys():
            if key in ("escape", "q"):
                win.close(); core.quit()
            elif key in ("left", "right"):
                # simulate cue == LEFT (1)
                pred = 1 if key=="left" else 2
                scene.update(cue=1, pred=pred)
        core.wait(0.01)


if __name__ == "__main__":
    demo()
