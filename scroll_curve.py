# scroll response curve
#
# this maps "how fast your hand is moving" to "how much the page scrolls"
# and it does it with a fancy nonlinear curve so it feels natural
#
# move your hand slow = precise pixel level scrolling
# move your hand fast = zooming through pages
# the transition between those two is buttery smooth
#
# also has three zones based on how far your hand is from the sensor:
#   close (5-15cm) = fine control for reading
#   normal (15-35cm) = regular scrolling
#   far (35-60cm) = turbo mode

import math
from dataclasses import dataclass
from collections import deque


@dataclass
class ScrollOutput:
    """what to actually send to the os"""
    ticks: int                # scroll wheel ticks this frame
    smooth_delta: float       # high res scroll for windows sendinput
    zone: str                 # what zone the hand is in
    curve_multiplier: float   # how much the curve amplified things


class ScrollCurve:
    """
    turns raw velocity into scroll commands

    uses a 4 zone response curve:
    1. dead zone: tiny velocities = nothing (kills jitter)
    2. micro scroll: small movements = single tick precision
    3. linear: medium speed = proportional and predictable
    4. power: fast = accelerated scrolling for speed readers
    """

    def __init__(self):
        # dead zone (anything below this is jitter, ignore it)
        self.DEAD_ZONE_VELOCITY = 1.2

        # distance zones
        self.FINE_ZONE_MAX = 15.0
        self.NORMAL_ZONE_MAX = 35.0

        # zone multipliers
        self.FINE_MULTIPLIER = 0.35      # close = precision mode
        self.NORMAL_MULTIPLIER = 1.0     # normal
        self.FAST_MULTIPLIER = 2.5       # far = turbo

        # response curve breakpoints
        self.MICRO_THRESHOLD = 3.0
        self.LINEAR_THRESHOLD = 20.0
        self.POWER_THRESHOLD = 45.0
        self.MICRO_SCALE = 0.5
        self.LINEAR_SCALE = 1.0
        self.POWER_EXPONENT = 1.4

        # sub pixel accumulator (keeps track of fractional scroll)
        self._accumulator = 0.0
        self._accumulator_decay = 0.92

        # prevents stutter from big bursts
        self._tick_budget: deque[int] = deque(maxlen=5)
        self.MAX_TICKS_PER_FRAME = 60

        # windows scroll resolution
        self.SMOOTH_SCROLL_SCALE = 120.0

        # base conversion rate
        self.BASE_TICKS_PER_CM_S = 4.5

        # output smoothing
        self._last_smooth_delta = 0.0
        self.OUTPUT_SMOOTHING = 0.3

    def _hermite_interpolate(self, t: float, p0: float, p1: float,
                              m0: float, m1: float) -> float:
        """smooth interpolation between two points with tangent control"""
        t2 = t * t
        t3 = t2 * t

        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2

        return h00*p0 + h10*m0 + h01*p1 + h11*m1

    def _response_curve(self, velocity: float) -> float:
        """the nonlinear curve that makes everything feel right"""
        speed = abs(velocity)
        direction = 1.0 if velocity >= 0 else -1.0

        # dead zone
        if speed < self.DEAD_ZONE_VELOCITY:
            return 0.0

        # micro scroll: gentle ramp from dead zone
        if speed < self.MICRO_THRESHOLD:
            t = (speed - self.DEAD_ZONE_VELOCITY) / (self.MICRO_THRESHOLD - self.DEAD_ZONE_VELOCITY)
            t_smooth = t * t * (3.0 - 2.0 * t)
            output = t_smooth * self.MICRO_SCALE * self.MICRO_THRESHOLD
            return output * direction

        # linear zone: predictable and proportional
        if speed < self.LINEAR_THRESHOLD:
            t = (speed - self.MICRO_THRESHOLD) / (self.LINEAR_THRESHOLD - self.MICRO_THRESHOLD)
            p0 = self.MICRO_SCALE * self.MICRO_THRESHOLD
            p1 = self.LINEAR_SCALE * self.LINEAR_THRESHOLD
            m0 = self.MICRO_SCALE * (self.LINEAR_THRESHOLD - self.MICRO_THRESHOLD) * 0.5
            m1 = self.LINEAR_SCALE * (self.LINEAR_THRESHOLD - self.MICRO_THRESHOLD)
            output = self._hermite_interpolate(t, p0, p1, m0, m1)
            return output * direction

        # power zone: accelerated for fast scrolling
        excess = speed - self.LINEAR_THRESHOLD
        linear_output = self.LINEAR_SCALE * self.LINEAR_THRESHOLD
        power_output = linear_output + excess ** self.POWER_EXPONENT
        return power_output * direction

    def _get_zone(self, distance: float) -> tuple[str, float]:
        """figure out which sensitivity zone the hand is in
        uses cosine blending so the transition is invisible"""
        if distance <= self.FINE_ZONE_MAX:
            return "fine", self.FINE_MULTIPLIER

        elif distance <= self.NORMAL_ZONE_MAX:
            t = (distance - self.FINE_ZONE_MAX) / (self.NORMAL_ZONE_MAX - self.FINE_ZONE_MAX)
            blend = 0.5 * (1.0 - math.cos(t * math.pi))
            multiplier = self.FINE_MULTIPLIER + blend * (self.NORMAL_MULTIPLIER - self.FINE_MULTIPLIER)
            return "normal", multiplier

        else:
            t = min(1.0, (distance - self.NORMAL_ZONE_MAX) / 25.0)
            blend = 0.5 * (1.0 - math.cos(t * math.pi))
            multiplier = self.NORMAL_MULTIPLIER + blend * (self.FAST_MULTIPLIER - self.NORMAL_MULTIPLIER)
            return "fast", multiplier

    def process(self, scroll_velocity: float, scroll_delta: float,
                hand_distance: float, dt: float) -> ScrollOutput:
        """take physics output and turn it into actual scroll commands"""
        # which zone
        zone_name, zone_multiplier = self._get_zone(hand_distance)

        # apply the response curve
        curved_velocity = self._response_curve(scroll_velocity)

        # how much to scroll this frame
        frame_scroll = curved_velocity * dt * self.BASE_TICKS_PER_CM_S * zone_multiplier

        # accumulate (keeps fractional scrolling working)
        if abs(frame_scroll) < 0.01:
            self._accumulator *= self._accumulator_decay
        self._accumulator += frame_scroll

        # get whole number ticks
        ticks = int(self._accumulator)

        # cap it
        if abs(ticks) > self.MAX_TICKS_PER_FRAME:
            ticks = self.MAX_TICKS_PER_FRAME if ticks > 0 else -self.MAX_TICKS_PER_FRAME

        # smooth out bursts
        if abs(ticks) > 0:
            self._tick_budget.append(abs(ticks))
            if len(self._tick_budget) >= 3:
                avg_recent = sum(self._tick_budget) / len(self._tick_budget)
                if abs(ticks) > avg_recent * 2.5 and avg_recent > 1:
                    max_ticks = max(2, int(avg_recent * 2.0))
                    if ticks > max_ticks:
                        ticks = max_ticks
                    elif ticks < -max_ticks:
                        ticks = -max_ticks

        if ticks != 0:
            self._accumulator -= ticks

        # prevent infinite buildup
        if abs(self._accumulator) > 5.0:
            self._accumulator = 5.0 if self._accumulator > 0 else -5.0

        # high res scroll for windows
        raw_smooth = frame_scroll * self.SMOOTH_SCROLL_SCALE
        self._last_smooth_delta = (
            (1.0 - self.OUTPUT_SMOOTHING) * self._last_smooth_delta
            + self.OUTPUT_SMOOTHING * raw_smooth
        )
        smooth_delta = self._last_smooth_delta

        if abs(scroll_velocity) > 0.01:
            curve_mult = abs(curved_velocity / scroll_velocity)
        else:
            curve_mult = 0.0

        return ScrollOutput(
            ticks=ticks,
            smooth_delta=smooth_delta,
            zone=zone_name,
            curve_multiplier=curve_mult
        )

    def reset(self):
        """wipe everything"""
        self._accumulator = 0.0
        self._last_smooth_delta = 0.0
        self._tick_budget.clear()
