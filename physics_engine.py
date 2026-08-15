# physics engine
#
# this is what makes the scroll feel like an iphone
# instead of feeling like a 2005 dell trackpad
#
# it simulates actual physics: momentum, friction, inertia
# when you flick your hand away the scroll keeps going
# and slowly decelerates to a stop
# just like when you swipe on your phone
#
# two phase deceleration: fast drop first then a long smooth tail
# thats the secret sauce that makes it feel premium

import math
from dataclasses import dataclass
from enum import Enum
from collections import deque


class ScrollPhase(Enum):
    """what the scroller is doing rn"""
    IDLE = "idle"            # chillin, not scrolling
    TRACKING = "tracking"    # hand is moving, following it
    COASTING = "coasting"    # hand lifted, riding the momentum
    SNAPPING = "snapping"    # smoothly stopping


@dataclass
class PhysicsOutput:
    """what the physics engine spits out each frame"""
    scroll_velocity: float    # how fast we scrolling
    scroll_delta: float       # how much to scroll this frame
    phase: ScrollPhase
    momentum: float           # how much energy left
    energy: float             # for the nerds (kinetic energy)


class PhysicsEngine:
    """
    real physics simulation that makes scrolling feel like butter

    the big idea: when you scroll on an iphone and lift your finger
    it doesnt just stop. it coasts and slowly decelerates
    thats exactly what this does but with your hand in the air
    """

    def __init__(self):
        self._velocity = 0.0
        self._phase = ScrollPhase.IDLE

        # two phase deceleration (the premium feel)
        # phase 1: quick slowdown right after you stop
        # phase 2: long gentle coast thats satisfying af
        self.FAST_DECAY = 1.8
        self.SLOW_DECAY = 0.9
        self.DECAY_TRANSITION_TIME = 0.25
        self._current_decay = self.FAST_DECAY

        # faster scrolls feel heavier (more momentum)
        self.BASE_MASS = 1.0
        self.MASS_VELOCITY_SCALE = 0.008
        self.MAX_MASS = 3.0

        # friction
        self.STATIC_FRICTION = 0.5
        self.KINETIC_COEFF = 0.012

        # hand input smoothing
        self._hand_velocity_smooth = 0.0
        self._hand_velocity_raw = 0.0

        # adaptive: smooth when slow (precision) responsive when fast
        self.HAND_SMOOTHING_SLOW = 0.20
        self.HAND_SMOOTHING_FAST = 0.55
        self.SMOOTHING_SPEED_THRESHOLD = 15.0

        # inertia (how much velocity carries between frames)
        self.INERTIA_SLOW = 0.95
        self.INERTIA_FAST = 0.80
        self.INERTIA_SPEED_THRESHOLD = 20.0

        # velocity history for flick detection
        self._velocity_samples: deque[tuple[float, float]] = deque(maxlen=20)
        self._total_time = 0.0

        # coasting state
        self._coast_start_velocity = 0.0
        self._coast_time = 0.0
        self._coast_initial_energy = 0.0

        # smooth stop (cosine curve, not abrupt)
        self.COAST_SNAP_VELOCITY = 1.2
        self._snap_start_velocity = 0.0
        self._snap_time = 0.0
        self._snap_duration = 0.15
        self._is_snapping = False

        # speed limiter (soft cap, not hard wall)
        self.MAX_VELOCITY = 250.0
        self.SPRING_K = 0.4
        self.SPRING_DAMPING = 0.88

        self._hand_active = False

        # flick detection
        self.FLICK_VELOCITY_THRESHOLD = 15.0
        self.FLICK_ACCELERATION_THRESHOLD = 50.0
        self._flick_energy_boost = 1.0

    @property
    def _mass(self) -> float:
        """heavier at higher speeds for that premium weight"""
        speed = abs(self._velocity)
        return min(self.MAX_MASS, self.BASE_MASS + speed * self.MASS_VELOCITY_SCALE)

    def _adaptive_smoothing(self, speed: float) -> float:
        """more smoothing when slow, less when fast"""
        t = min(1.0, speed / self.SMOOTHING_SPEED_THRESHOLD)
        t = t * t * (3.0 - 2.0 * t)
        return self.HAND_SMOOTHING_SLOW + t * (self.HAND_SMOOTHING_FAST - self.HAND_SMOOTHING_SLOW)

    def _adaptive_inertia(self, speed: float) -> float:
        """more inertia when slow (smooth) less when fast (responsive)"""
        t = min(1.0, speed / self.INERTIA_SPEED_THRESHOLD)
        t = t * t * (3.0 - 2.0 * t)
        return self.INERTIA_SLOW + t * (self.INERTIA_FAST - self.INERTIA_SLOW)

    def _apply_friction(self, velocity: float, dt: float) -> float:
        """slows things down like real friction would"""
        if abs(velocity) < 0.01:
            return 0.0

        speed = abs(velocity)
        direction = 1.0 if velocity > 0 else -1.0

        friction_force = self.STATIC_FRICTION + self.KINETIC_COEFF * speed
        friction_decel = friction_force * dt / self._mass
        new_speed = max(0.0, speed - friction_decel)

        return new_speed * direction

    def _apply_spring_damper(self, velocity: float, dt: float) -> float:
        """soft speed limit so it doesnt go crazy"""
        if abs(velocity) <= self.MAX_VELOCITY:
            return velocity

        overshoot = abs(velocity) - self.MAX_VELOCITY
        direction = 1.0 if velocity > 0 else -1.0

        spring_force = -self.SPRING_K * overshoot * dt
        damped = velocity * self.SPRING_DAMPING

        return damped + spring_force * direction

    def _two_phase_decay(self, v0: float, time: float) -> float:
        """the magic decay curve
        fast drop first then long smooth tail
        faster initial speeds get proportionally more decay
        so everything stops in about 1 to 3 seconds"""
        speed_factor = 1.0 + abs(v0) / 25.0

        fast_decay = self.FAST_DECAY * speed_factor
        slow_decay = self.SLOW_DECAY * speed_factor

        blend = min(1.0, time / self.DECAY_TRANSITION_TIME)
        blend = blend * blend * (3.0 - 2.0 * blend)

        decay = fast_decay + blend * (slow_decay - fast_decay)

        return v0 * math.exp(-decay * time)

    def _detect_flick(self) -> float:
        """did you just flick your hand? if yes, boost the coast energy"""
        if len(self._velocity_samples) < 5:
            return 1.0

        samples = list(self._velocity_samples)
        recent = samples[-5:]
        velocities = [v for _, v in recent]
        times = [t for t, _ in recent]

        peak_speed = max(abs(v) for v in velocities)

        if len(recent) >= 2:
            dt_window = times[-1] - times[0]
            if dt_window > 0.001:
                accel = abs(velocities[-1] - velocities[0]) / dt_window
            else:
                accel = 0.0
        else:
            accel = 0.0

        if peak_speed > self.FLICK_VELOCITY_THRESHOLD and accel > self.FLICK_ACCELERATION_THRESHOLD:
            intensity = min(3.0, accel / self.FLICK_ACCELERATION_THRESHOLD)
            return 1.0 + intensity * 0.4
        elif peak_speed > self.FLICK_VELOCITY_THRESHOLD * 2:
            return 1.3

        return 1.0

    def update_tracking(self, hand_velocity: float, dt: float) -> PhysicsOutput:
        """hand is moving, follow it with physics"""
        self._total_time += dt
        self._hand_active = True
        self._phase = ScrollPhase.TRACKING
        self._is_snapping = False

        self._hand_velocity_raw = hand_velocity
        self._velocity_samples.append((self._total_time, hand_velocity))

        current_speed = abs(self._hand_velocity_smooth)
        smoothing = self._adaptive_smoothing(current_speed)

        self._hand_velocity_smooth = (
            (1.0 - smoothing) * self._hand_velocity_smooth
            + smoothing * hand_velocity
        )

        target_velocity = self._hand_velocity_smooth
        inertia = self._adaptive_inertia(abs(target_velocity))

        self._velocity = (
            inertia * self._velocity
            + (1.0 - inertia) * target_velocity
        )

        self._velocity = self._apply_spring_damper(self._velocity, dt)

        scroll_delta = self._velocity * dt

        mass = self._mass
        return PhysicsOutput(
            scroll_velocity=self._velocity,
            scroll_delta=scroll_delta,
            phase=self._phase,
            momentum=abs(self._velocity) * mass,
            energy=0.5 * mass * self._velocity ** 2
        )

    def update_coasting(self, dt: float) -> PhysicsOutput:
        """hand lifted, coast to a stop like an iphone"""
        self._total_time += dt

        # first frame of coasting
        if self._hand_active:
            self._hand_active = False
            self._coast_time = 0.0

            self._flick_energy_boost = self._detect_flick()
            self._coast_start_velocity = self._velocity * self._flick_energy_boost

            # slow movements dont need much coast
            speed = abs(self._coast_start_velocity)
            if speed < self.FLICK_VELOCITY_THRESHOLD:
                scale = (speed / self.FLICK_VELOCITY_THRESHOLD) ** 2
                self._coast_start_velocity *= scale

            self._coast_initial_energy = 0.5 * self._mass * self._coast_start_velocity ** 2
            self._phase = ScrollPhase.COASTING
            self._is_snapping = False

        self._coast_time += dt

        # smooth stop phase (cosine curve, no abrupt halt)
        if self._is_snapping:
            self._snap_time += dt
            progress = min(1.0, self._snap_time / self._snap_duration)

            ease = 0.5 * (1.0 + math.cos(progress * math.pi))
            self._velocity = self._snap_start_velocity * ease

            if progress >= 1.0:
                self._velocity = 0.0
                self._phase = ScrollPhase.IDLE
                self._is_snapping = False

                return PhysicsOutput(
                    scroll_velocity=0.0,
                    scroll_delta=0.0,
                    phase=ScrollPhase.IDLE,
                    momentum=0.0,
                    energy=0.0
                )

            scroll_delta = self._velocity * dt
            mass = self._mass
            return PhysicsOutput(
                scroll_velocity=self._velocity,
                scroll_delta=scroll_delta,
                phase=ScrollPhase.SNAPPING,
                momentum=abs(self._velocity) * mass,
                energy=0.5 * mass * self._velocity ** 2
            )

        # two phase decay
        self._velocity = self._two_phase_decay(
            self._coast_start_velocity, self._coast_time
        )

        self._velocity = self._apply_friction(self._velocity, dt)

        # time to start the smooth stop?
        if abs(self._velocity) < self.COAST_SNAP_VELOCITY:
            self._is_snapping = True
            self._snap_start_velocity = self._velocity
            self._snap_time = 0.0
            self._phase = ScrollPhase.SNAPPING

        scroll_delta = self._velocity * dt
        mass = self._mass

        return PhysicsOutput(
            scroll_velocity=self._velocity,
            scroll_delta=scroll_delta,
            phase=self._phase,
            momentum=abs(self._velocity) * mass,
            energy=0.5 * mass * self._velocity ** 2
        )

    def force_stop(self):
        """emergency brake"""
        self._velocity = 0.0
        self._hand_velocity_smooth = 0.0
        self._hand_velocity_raw = 0.0
        self._coast_start_velocity = 0.0
        self._phase = ScrollPhase.IDLE
        self._hand_active = False
        self._is_snapping = False
        self._velocity_samples.clear()

    def is_idle(self) -> bool:
        return self._phase == ScrollPhase.IDLE

    def is_coasting(self) -> bool:
        return self._phase in (ScrollPhase.COASTING, ScrollPhase.SNAPPING)

    @property
    def phase(self) -> ScrollPhase:
        return self._phase

    @property
    def velocity(self) -> float:
        return self._velocity

    def reset(self):
        """full reset"""
        self.force_stop()
        self._coast_time = 0.0
        self._total_time = 0.0
        self._flick_energy_boost = 1.0
        self._coast_initial_energy = 0.0
