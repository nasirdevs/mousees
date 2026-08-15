# kalman filter for cleaning up noisy ultrasonic readings
#
# the ultrasonic sensor is kinda mid on its own ngl
# it spits out random spikes and jitters like crazy
# this kalman filter tracks position, velocity, and acceleration
# and figures out whats real movement vs sensor being dumb
#
# it also auto calibrates itself when you first start up
# so it learns how noisy YOUR specific sensor is

import numpy as np
from dataclasses import dataclass


@dataclass
class KalmanState:
    """whats the filter currently thinking"""
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    confidence: float = 0.0


class AdaptiveKalmanFilter:
    """
    cleans up the raw sensor data and tracks where your hand actually is

    tracks three things: position, velocity, acceleration
    adapts to how noisy the sensor is in real time
    rejects readings that are clearly wrong (statistical outliers)
    """

    def __init__(self):
        # state: position, velocity, acceleration
        self.x = np.zeros(3)

        # how uncertain we are (starts very uncertain)
        self.P = np.eye(3) * 100.0

        # sensor noise (gets calibrated automatically)
        self.R = np.array([[1.5]])

        # process noise (how much we expect things to change)
        # lower = less jitter, higher = more responsive
        self._q_position = 0.005
        self._q_velocity = 0.3
        self._q_acceleration = 8.0

        # motion detection (asymmetric on purpose)
        # ramps up fast when you start moving, decays slowly when you stop
        # this is what makes it feel smooth
        self._motion_regime = 0.0
        self._regime_up_rate = 0.35    # quick to react
        self._regime_down_rate = 0.92  # slow to settle

        # outlier rejection (adaptive threshold)
        self._mahalanobis_base = 3.0
        self._mahalanobis_threshold = 3.0
        self._consecutive_outliers = 0
        self._max_consecutive_outliers = 5

        # tracks how accurate the filter is being
        self._innovation_history: list[float] = []
        self._innovation_window = 30
        self._r_adaptation_rate = 0.05

        # drift killer: when your hand is still, this slowly
        # zeroes out any phantom velocity from sensor noise
        self._stillness_counter = 0
        self._stillness_threshold = 0.8
        self._stillness_frames_required = 8
        self._velocity_damping = 0.0

        # calibration stuff
        self._calibration_readings: list[float] = []
        self._calibration_count = 60
        self._is_calibrated = False
        self._calibrated_noise_floor = 0.0

        # we can only measure position directly
        self.H = np.array([[1.0, 0.0, 0.0]])

        # stats
        self.readings_processed = 0
        self.outliers_rejected = 0

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def _build_F(self, dt: float) -> np.ndarray:
        """physics model: how position/velocity/acceleration relate over time"""
        dt2 = 0.5 * dt * dt
        return np.array([
            [1.0, dt, dt2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ])

    def _build_Q(self, dt: float) -> np.ndarray:
        """how much uncertainty to add each step
        more when moving (so it stays responsive)
        less when still (so it doesnt jitter)"""
        regime_cubed = self._motion_regime ** 2
        motion_scale = 1.0 + regime_cubed * 15.0

        q_p = self._q_position * (1.0 + self._motion_regime * 2.0)
        q_v = self._q_velocity * motion_scale
        q_a = self._q_acceleration * motion_scale

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        Q = np.array([
            [dt4 / 4.0 * q_a + dt2 * q_p, dt3 / 2.0 * q_a, dt2 / 2.0 * q_a],
            [dt3 / 2.0 * q_a, dt2 * q_a + q_v, dt * q_a],
            [dt2 / 2.0 * q_a, dt * q_a, q_a]
        ])

        return Q

    def calibrate_step(self, measurement: float) -> bool:
        """feed it readings during startup to learn the sensor noise"""
        self._calibration_readings.append(measurement)

        if len(self._calibration_readings) >= self._calibration_count:
            readings = np.array(self._calibration_readings)

            # use median absolute deviation (handles spikes during calibration)
            median = np.median(readings)
            mad = np.median(np.abs(readings - median))
            robust_std = 1.4826 * mad

            variance = np.var(readings)
            noise_estimate = max(robust_std ** 2, variance, 0.1)

            self.R = np.array([[noise_estimate]])
            self._calibrated_noise_floor = np.sqrt(noise_estimate)

            self.x[0] = median
            self.x[1] = 0.0
            self.x[2] = 0.0
            self.P = np.diag([noise_estimate, 0.5, 2.0])

            self._is_calibrated = True
            return True

        return False

    def predict(self, dt: float):
        """predict where the hand will be based on physics"""
        F = self._build_F(dt)
        Q = self._build_Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # drift killer kicks in when hand is still
        if self._velocity_damping > 0:
            damping_matrix = np.diag([1.0, 1.0 - self._velocity_damping * 0.15, 1.0 - self._velocity_damping * 0.25])
            self.x = damping_matrix @ self.x

    def _mahalanobis_distance(self, measurement: float) -> float:
        """how many standard deviations is this reading from what we expected
        big number = probably a garbage reading"""
        innovation = np.array([measurement]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        d_squared = float(innovation @ np.linalg.inv(S) @ innovation.T)
        return np.sqrt(max(d_squared, 0.0))

    def _adapt_measurement_noise(self, innovation: float):
        """learns how noisy the sensor actually is over time"""
        self._innovation_history.append(innovation ** 2)
        if len(self._innovation_history) > self._innovation_window:
            self._innovation_history.pop(0)

        if len(self._innovation_history) >= 10:
            S_matrix = self.H @ self.P @ self.H.T + self.R
            expected_var = float(S_matrix[0, 0])
            actual_var = float(np.mean(self._innovation_history))

            ratio = actual_var / max(expected_var, 0.01)
            r_adjustment = 1.0 + (ratio - 1.0) * self._r_adaptation_rate
            r_adjustment = max(0.9, min(1.1, r_adjustment))

            new_r = float(self.R[0, 0]) * r_adjustment
            self.R = np.array([[max(new_r, 0.05)]])

    def update(self, measurement: float, dt: float) -> KalmanState | None:
        """the main thing. give it a raw reading, get back clean data"""
        self.readings_processed += 1

        # still calibrating
        if not self._is_calibrated:
            done = self.calibrate_step(measurement)
            if done:
                return KalmanState(
                    position=self.x[0],
                    velocity=self.x[1],
                    acceleration=self.x[2],
                    confidence=0.3
                )
            return None

        # predict step
        self.predict(dt)

        # if we keep getting outliers, relax the threshold
        # (maybe the hand actually moved and we need to catch up)
        if self._consecutive_outliers >= self._max_consecutive_outliers:
            self._mahalanobis_threshold = self._mahalanobis_base * 2.0
        else:
            self._mahalanobis_threshold = self._mahalanobis_base

        # check if this reading is sus
        mahal_dist = self._mahalanobis_distance(measurement)

        if mahal_dist > self._mahalanobis_threshold:
            self.outliers_rejected += 1
            self._consecutive_outliers += 1

            confidence = min(1.0, self.readings_processed / 100.0) * 0.4
            return KalmanState(
                position=self.x[0],
                velocity=self.x[1],
                acceleration=self.x[2],
                confidence=confidence
            )

        self._consecutive_outliers = 0

        # kalman math
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        innovation = np.array([measurement]) - self.H @ self.x
        self.x = self.x + (K @ innovation).flatten()

        # joseph form update (numerically stable)
        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self._adapt_measurement_noise(float(innovation[0]))

        # motion regime: ramps up fast, decays slowly
        speed = abs(self.x[1])
        instant_regime = min(speed / 25.0, 1.0)

        if instant_regime > self._motion_regime:
            self._motion_regime = (
                (1.0 - self._regime_up_rate) * self._motion_regime
                + self._regime_up_rate * instant_regime
            )
        else:
            self._motion_regime = (
                self._regime_down_rate * self._motion_regime
                + (1.0 - self._regime_down_rate) * instant_regime
            )

        # drift killer
        if speed < self._stillness_threshold:
            self._stillness_counter += 1
            if self._stillness_counter > self._stillness_frames_required:
                self._velocity_damping = min(1.0, self._velocity_damping + 0.15)
        else:
            self._stillness_counter = 0
            self._velocity_damping = max(0.0, self._velocity_damping - 0.3)

        # how confident are we rn
        trace = np.trace(self.P)
        base_conf = min(1.0, self.readings_processed / 60.0)
        noise_conf = max(0.0, 1.0 - trace / 30.0)
        confidence = max(0.0, min(1.0, base_conf * noise_conf))

        return KalmanState(
            position=self.x[0],
            velocity=self.x[1],
            acceleration=self.x[2],
            confidence=confidence
        )

    def reset(self):
        """start fresh"""
        self.x = np.zeros(3)
        self.P = np.eye(3) * 100.0
        self._motion_regime = 0.0
        self._velocity_damping = 0.0
        self._stillness_counter = 0
        self._consecutive_outliers = 0
        self._innovation_history.clear()
        self.readings_processed = 0
        self._calibration_readings.clear()
        self._is_calibrated = False
