# ml velocity predictor
#
# a tiny neural network that learns how YOU scroll
# it watches the kalman filter output and learns patterns
# then predicts where your hand is going next
#
# also classifies your hand gesture in real time:
#   still, slow scroll, fast scroll, flick, or stopping
#
# pure numpy, no tensorflow or pytorch needed
# trains itself while you use it (online learning)

import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import deque


class GestureType(Enum):
    """what your hand is doing"""
    STILL = "still"
    SLOW_SCROLL = "slow_scroll"
    FAST_SCROLL = "fast_scroll"
    FLICK = "flick"
    STOPPING = "stopping"


@dataclass
class Prediction:
    """what the neural net thinks"""
    predicted_velocity: float
    gesture: GestureType
    confidence: float     # 0 to 1, how sure it is
    blend_factor: float   # how much to trust this vs the kalman filter


class ReplayBuffer:
    """saves past experiences so the network can learn from them"""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.features: deque[np.ndarray] = deque(maxlen=capacity)
        self.targets: deque[float] = deque(maxlen=capacity)

    def add(self, features: np.ndarray, target: float):
        self.features.append(features.copy())
        self.targets.append(target)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        size = min(batch_size, len(self.features))
        indices = np.random.choice(len(self.features), size, replace=False)
        X = np.array([self.features[i] for i in indices])
        y = np.array([self.targets[i] for i in indices])
        return X, y

    def __len__(self):
        return len(self.features)


def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """relu but doesnt completely kill negative values"""
    return np.where(x > 0, x, alpha * x)


def _leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)


class VelocityPredictor:
    """
    smol neural network: 60 inputs > 32 neurons > 16 neurons > 1 output

    looks at the last 10 readings and predicts where your hand is going
    each reading has 6 features: position, velocity, acceleration,
    jerk, speed, and velocity change
    """

    WINDOW_SIZE = 10
    FEATURES_PER_STEP = 6
    INPUT_DIM = WINDOW_SIZE * FEATURES_PER_STEP  # 60

    def __init__(self):
        # sliding window of recent readings
        self._history: deque[np.ndarray] = deque(maxlen=self.WINDOW_SIZE)

        # network weights (he initialization for leaky relu)
        rng = np.random.RandomState(42)

        # 60 > 32
        self.W1 = rng.randn(self.INPUT_DIM, 32) * np.sqrt(2.0 / self.INPUT_DIM)
        self.b1 = np.zeros(32)

        # 32 > 16
        self.W2 = rng.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros(16)

        # 16 > 1
        self.W3 = rng.randn(16, 1) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros(1)

        # normalizes inputs so the network doesnt freak out
        self._feature_mean = np.zeros(self.INPUT_DIM)
        self._feature_var = np.ones(self.INPUT_DIM)
        self._norm_count = 0
        self._norm_alpha = 0.02

        # learning stuff
        self.replay_buffer = ReplayBuffer(capacity=5000)
        self._base_learning_rate = 0.002
        self._min_learning_rate = 0.0002
        self._lr_decay = 0.9999
        self.learning_rate = self._base_learning_rate
        self._train_interval = 15
        self._prediction_count = 0
        self._min_training_samples = 30

        # gesture classification (with hysteresis so it doesnt flicker)
        self._velocity_history: deque[float] = deque(maxlen=20)
        self._accel_history: deque[float] = deque(maxlen=15)
        self._jerk_history: deque[float] = deque(maxlen=10)
        self._last_gesture = GestureType.STILL
        self._gesture_hold_frames = 0
        self._gesture_change_threshold = 3

        # tracks prediction accuracy
        self._prediction_errors: deque[float] = deque(maxlen=80)
        self._warmup_predictions = 60

        self._prev_velocity = 0.0

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self._norm_count < 2:
            return features
        return (features - self._feature_mean) / (np.sqrt(self._feature_var) + 1e-8)

    def _update_normalization(self, features: np.ndarray):
        self._norm_count += 1
        if self._norm_count == 1:
            self._feature_mean = features.copy()
            self._feature_var = np.ones_like(features) * 0.1
        else:
            self._feature_mean = (
                (1 - self._norm_alpha) * self._feature_mean
                + self._norm_alpha * features
            )
            diff = features - self._feature_mean
            self._feature_var = (
                (1 - self._norm_alpha) * self._feature_var
                + self._norm_alpha * diff * diff
            )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """run data through the network"""
        z1 = x @ self.W1 + self.b1
        a1 = _leaky_relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = _leaky_relu(z2)

        z3 = a2 @ self.W3 + self.b3
        output = np.tanh(z3) * 80.0

        return output

    def _backward(self, X: np.ndarray, y: np.ndarray):
        """learn from mistakes (backpropagation)"""
        batch_size = X.shape[0]

        z1 = X @ self.W1 + self.b1
        a1 = _leaky_relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _leaky_relu(z2)
        z3 = a2 @ self.W3 + self.b3
        output = np.tanh(z3) * 80.0

        y_reshaped = y.reshape(-1, 1)
        error = output - y_reshaped
        huber_threshold = 10.0
        d_output = np.where(
            np.abs(error) < huber_threshold,
            2.0 * error / batch_size,
            2.0 * huber_threshold * np.sign(error) / batch_size
        )

        d_z3 = d_output * 80.0 * (1 - np.tanh(z3) ** 2)

        d_W3 = a2.T @ d_z3
        d_b3 = d_z3.sum(axis=0)

        d_a2 = d_z3 @ self.W3.T
        d_z2 = d_a2 * _leaky_relu_derivative(z2)

        d_W2 = a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)

        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * _leaky_relu_derivative(z1)

        d_W1 = X.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        # gradient clipping (keeps things stable)
        all_grads = [d_W1, d_W2, d_W3, d_b1, d_b2, d_b3]
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in all_grads))
        max_norm = 2.0
        if total_norm > max_norm:
            scale = max_norm / total_norm
            for g in all_grads:
                g *= scale

        lr = self.learning_rate
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3

        self.learning_rate = max(
            self._min_learning_rate,
            self.learning_rate * self._lr_decay
        )

    def _classify_gesture(self) -> GestureType:
        """figures out what your hand is doing based on recent history
        has hysteresis so it doesnt flip flop between states"""
        if len(self._velocity_history) < 5:
            return GestureType.STILL

        recent_vel = list(self._velocity_history)[-7:]
        avg_speed = np.mean(np.abs(recent_vel))

        if len(recent_vel) >= 4:
            first_half_speed = np.mean(np.abs(recent_vel[:len(recent_vel)//2]))
            second_half_speed = np.mean(np.abs(recent_vel[len(recent_vel)//2:]))
            speed_trend = second_half_speed - first_half_speed
        else:
            speed_trend = 0.0

        if len(self._accel_history) >= 3:
            recent_accel = list(self._accel_history)[-5:]
            avg_accel = np.mean(np.abs(recent_accel))
            peak_accel = max(np.abs(recent_accel))
        else:
            avg_accel = 0.0
            peak_accel = 0.0

        if len(self._jerk_history) >= 3:
            recent_jerk = list(self._jerk_history)[-3:]
            avg_jerk = np.mean(np.abs(recent_jerk))
        else:
            avg_jerk = 0.0

        candidate = GestureType.STILL

        if avg_speed > 15.0 and (peak_accel > 40.0 or avg_jerk > 50.0):
            candidate = GestureType.FLICK
        elif avg_speed > 2.0 and speed_trend < -2.0:
            candidate = GestureType.STOPPING
        elif avg_speed < 1.0:
            candidate = GestureType.STILL
        elif avg_speed < 10.0:
            candidate = GestureType.SLOW_SCROLL
        else:
            candidate = GestureType.FAST_SCROLL

        # hysteresis: need to see the new gesture for a few frames
        # before we actually switch (prevents flickering)
        # exception: flicks override immediately cuz they're time sensitive
        if candidate == self._last_gesture:
            self._gesture_hold_frames = 0
            return candidate

        self._gesture_hold_frames += 1
        if self._gesture_hold_frames >= self._gesture_change_threshold:
            self._last_gesture = candidate
            self._gesture_hold_frames = 0
            return candidate

        if candidate == GestureType.FLICK:
            self._last_gesture = candidate
            self._gesture_hold_frames = 0
            return candidate

        return self._last_gesture

    def feed(self, position: float, velocity: float, acceleration: float, dt: float):
        """give it new data from the kalman filter"""
        if len(self._accel_history) > 0:
            jerk = (acceleration - self._accel_history[-1]) / max(dt, 0.001)
        else:
            jerk = 0.0

        speed = abs(velocity)
        vel_delta = velocity - self._prev_velocity
        self._prev_velocity = velocity

        step_features = np.array([position, velocity, acceleration, jerk, speed, vel_delta])
        self._history.append(step_features)

        self._velocity_history.append(velocity)
        self._accel_history.append(acceleration)
        self._jerk_history.append(jerk)

        if len(self._history) == self.WINDOW_SIZE:
            features = np.concatenate(list(self._history))
            self._update_normalization(features)
            self.replay_buffer.add(self._normalize_features(features), velocity)

    def predict(self) -> Prediction | None:
        """predict velocity and classify gesture"""
        if len(self._history) < self.WINDOW_SIZE:
            return None

        self._prediction_count += 1

        features = np.concatenate(list(self._history))
        normalized = self._normalize_features(features)

        output = self._forward(normalized.reshape(1, -1))
        predicted_velocity = float(output[0, 0])

        gesture = self._classify_gesture()

        if len(self._prediction_errors) > 15:
            errors = np.array(list(self._prediction_errors))
            median_error = np.median(errors)
            raw_conf = max(0.0, 1.0 - median_error / 15.0)
        else:
            raw_conf = 0.15

        warmup_scale = min(1.0, self._prediction_count / self._warmup_predictions)
        confidence = raw_conf * warmup_scale

        # trust the ml more for fast movements and flicks
        gesture_blend_boost = {
            GestureType.STILL: 0.3,
            GestureType.SLOW_SCROLL: 0.5,
            GestureType.FAST_SCROLL: 0.7,
            GestureType.FLICK: 0.8,
            GestureType.STOPPING: 0.6,
        }.get(gesture, 0.5)

        blend_factor = confidence * gesture_blend_boost

        if len(self._velocity_history) > 0:
            actual = self._velocity_history[-1]
            error = abs(predicted_velocity - actual)
            self._prediction_errors.append(error)

        # train on random batch from replay buffer
        if (self._prediction_count % self._train_interval == 0
                and len(self.replay_buffer) >= self._min_training_samples):
            X, y = self.replay_buffer.sample(batch_size=48)
            self._backward(X, y)

        return Prediction(
            predicted_velocity=predicted_velocity,
            gesture=gesture,
            confidence=confidence,
            blend_factor=blend_factor
        )

    def reset(self):
        """clear state but keep what it learned"""
        self._history.clear()
        self._velocity_history.clear()
        self._accel_history.clear()
        self._jerk_history.clear()
        self._prediction_count = 0
        self._prev_velocity = 0.0
        self._last_gesture = GestureType.STILL
        self._gesture_hold_frames = 0
