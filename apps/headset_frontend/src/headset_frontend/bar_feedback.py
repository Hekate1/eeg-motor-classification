from psychopy import visual

class BarFeedback:
    """Handles non-game bar feedback with fixed baseline, axis, ticks, and drawing."""

    def __init__(self, win, bar_width=50, max_height=400, baseline_y=-250, bar_x_offset=200):
        self.win = win
        self.bar_width = bar_width
        self.max_height = max_height
        self.baseline_y = baseline_y
        self.bar_x_offset = bar_x_offset
        # compute bar edges (left_edge outside left bar, right_edge inside left side of right bar)
        self.left_edge = -bar_x_offset - bar_width/2
        self.right_edge = bar_x_offset - bar_width/2
        # create bars
        self.bar_left = visual.Rect(
            win,
            width=bar_width,
            height=0,
            pos=(-bar_x_offset, baseline_y),
            fillColor="#4f83ff",
            lineColor="black"
        )
        self.bar_right = visual.Rect(
            win,
            width=bar_width,
            height=0,
            pos=(bar_x_offset, baseline_y),
            fillColor="#ff4f4f",
            lineColor="black"
        )
        # create axes
        self.axis_left = visual.Line(
            win,
            start=(self.left_edge, baseline_y),
            end=(self.left_edge, baseline_y + max_height),
            lineColor="black"
        )
        # axis for right bar at its left edge
        self.axis_right = visual.Line(
            win,
            start=(self.right_edge, baseline_y),
            end=(self.right_edge, baseline_y + self.max_height),
            lineColor="black"
        )
        # create ticks and labels
        tick_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.ticks_left = []
        self.ticks_right = []
        for frac in tick_fracs:
            y = baseline_y + frac * max_height
            # left tick and label
            tick_left = visual.Line(
                win,
                start=(self.left_edge, y),
                end=(self.left_edge - 10, y),
                lineColor="black"
            )

            self.ticks_left.append(tick_left)
            # right tick and label moved to left side of right bar
            tick_right = visual.Line(
                win,
                start=(self.right_edge, y),
                end=(self.right_edge - 10, y),
                lineColor="black"
            )

            self.ticks_right.append(tick_right)
        # percentage text for bars
        self.txt_left = visual.TextStim(
            win,
            text='0%',
            pos=(-self.bar_x_offset + 10, self.baseline_y),
            color='black',
            height=24
        )
        self.txt_right = visual.TextStim(
            win,
            text='0%',
            pos=(self.bar_x_offset + 10, self.baseline_y),
            color='black',
            height=24
        )

    def update(self, proba):
        """Update bar heights based on probability array [left, right]."""
        left_h = self.max_height * proba[0]
        self.bar_left.height = left_h
        self.bar_left.pos = (
            -self.bar_x_offset,
            self.baseline_y + left_h / 2
        )
        right_h = self.max_height * proba[1]
        self.bar_right.height = right_h
        self.bar_right.pos = (
            self.bar_x_offset,
            self.baseline_y + right_h / 2
        )
        # update percentage text
        self.txt_left.text = f"{proba[0]*100:.1f}%"
        self.txt_left.pos = (
            -self.bar_x_offset + 10,
            self.baseline_y + left_h + 20
        )
        self.txt_right.text = f"{proba[1]*100:.1f}%"
        self.txt_right.pos = (
            self.bar_x_offset + 10,
            self.baseline_y + right_h + 20
        )

    def draw(self):
        """Draw axes, bars, ticks, and labels."""
        # draw axes behind bars
        self.axis_left.draw()
        self.axis_right.draw()
        # draw bars
        self.bar_left.draw()
        self.bar_right.draw()
        # draw ticks and labels
        for t in self.ticks_left:
            t.draw()
        for t in self.ticks_right:
            t.draw()
        # draw percentage texts
        self.txt_left.draw()
        self.txt_right.draw()