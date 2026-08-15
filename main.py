# mousees
# the coolest scroller you have ever seen
#
# wave your hand over an ultrasonic sensor
# and it scrolls your screen like magic
#
# the pipeline:
#   sensor reading > noise filter > neural net > physics sim > scroll output
#
# works over bluetooth or usb, your choice

import serial
import serial.tools.list_ports
import time
import ctypes
import ctypes.wintypes
import sys
import math
from collections import deque

from kalman_filter import AdaptiveKalmanFilter
from ml_predictor import VelocityPredictor, GestureType
from physics_engine import PhysicsEngine, ScrollPhase
from scroll_curve import ScrollCurve


# windows scroll api stuff
# we talk directly to the os for buttery smooth scrolling

INPUT_MOUSE = 0
MOUSEEVENTF_WHEEL = 0x0800


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


# pre allocated so the hot loop doesnt waste time on memory
_scroll_input = INPUT()
_scroll_input.type = INPUT_MOUSE
_scroll_input.union.mi.dx = 0
_scroll_input.union.mi.dy = 0
_scroll_input.union.mi.dwFlags = MOUSEEVENTF_WHEEL
_scroll_input.union.mi.time = 0
_scroll_extra = ctypes.c_ulong(0)
_scroll_input.union.mi.dwExtraInfo = ctypes.pointer(_scroll_extra)

_send_input = ctypes.windll.user32.SendInput
_input_size = ctypes.sizeof(INPUT)


def send_smooth_scroll(delta: int):
    """yeet a scroll event to windows"""
    if delta == 0:
        return

    _scroll_input.union.mi.mouseData = ctypes.wintypes.DWORD(delta & 0xFFFFFFFF)
    _send_input(1, ctypes.byref(_scroll_input), _input_size)


def send_distributed_scroll(total_delta: int, chunks: int = 1):
    """split a big scroll into smaller ones so it feels smooth instead of chunky"""
    if total_delta == 0 or chunks <= 0:
        return

    chunks = min(chunks, 8)
    chunk_size = total_delta // chunks
    remainder = total_delta - chunk_size * chunks

    for i in range(chunks):
        delta = chunk_size + (1 if i < abs(remainder) else 0)
        if remainder < 0:
            delta = chunk_size - (1 if i < abs(remainder) else 0)
        if delta != 0:
            send_smooth_scroll(delta)


# config

PORT = "COM3"
BAUD_RATE = 115200

MIN_DISTANCE = 5.0      # closer than this = too close
MAX_DISTANCE = 60.0      # further than this = too far

TARGET_LOOP_HZ = 250     # how fast the main loop runs

HAND_ABSENT_TIMEOUT = 0.12    # how long before we decide hand is gone
HAND_GLITCH_TOLERANCE = 0.04  # ignore tiny sensor dropouts

SCROLL_MULTIPLIER = 3.0
SMOOTH_SCROLL_DIVISOR = 0.8

DISTRIBUTE_THRESHOLD = 180
DISTRIBUTE_CHUNKS = 3

CALIBRATION_SECONDS = 1.5

SHOW_DEBUG = True
DEBUG_INTERVAL = 0.12

# different gestures get different scroll speeds
GESTURE_SCROLL_SCALE = {
    GestureType.STILL: 0.0,
    GestureType.SLOW_SCROLL: 0.85,
    GestureType.FAST_SCROLL: 1.15,
    GestureType.FLICK: 1.3,
    GestureType.STOPPING: 0.6,
}


def find_arduino_port() -> str | None:
    """sniff out the arduino or bluetooth port automatically"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = (port.description or "").lower()
        mfr = (port.manufacturer or "").lower()
        if any(keyword in desc for keyword in ["arduino", "ch340", "cp210", "ftdi", "usb serial", "usb-serial", "bluetooth", "hc-05", "standard serial over bluetooth link"]):
            return port.device
        if any(keyword in mfr for keyword in ["arduino", "wch", "silicon labs", "ftdi"]):
            return port.device
    return None


def main():
    print()
    print("  +--------------------------------------------------+")
    print("  |                                                    |")
    print("  |        m o u s e e s                               |")
    print("  |        the coolest scroller u have ever seen       |")
    print("  |                                                    |")
    print("  |        wave ur hand. stuff scrolls. its magic.     |")
    print("  |                                                    |")
    print("  +--------------------------------------------------+")
    print()

    # find the port
    port = PORT
    auto_port = find_arduino_port()
    if auto_port:
        port = auto_port
        print(f"  found it on {port}")
    else:
        print(f"  using {port} (change PORT in the code if thats wrong)")

    # connect (tries different baud rates for bluetooth compatibility)
    arduino = None
    for baud in [9600, 115200, 38400]:
        try:
            print(f"  trying {port} at {baud} baud...")
            arduino = serial.Serial(port, baud, timeout=0.01)
            BAUD_RATE = baud
            break
        except serial.SerialException:
            continue

    if not arduino:
        print(f"\n  couldnt connect to {port}")
        print(f"  if bluetooth: pair it first (pin 1234 or 0000)")
        print(f"  if usb: make sure its plugged in lol")
        sys.exit(1)

    time.sleep(1.5)
    arduino.reset_input_buffer()
    print(f"  connected to {port} ({BAUD_RATE} baud)")

    # fire up all the modules
    kalman = AdaptiveKalmanFilter()
    predictor = VelocityPredictor()
    physics = PhysicsEngine()
    curve = ScrollCurve()

    print()
    print("  calibrating... hold ur hand still or keep it away")

    # state
    last_valid_time = time.perf_counter()
    last_debug_time = time.perf_counter()
    last_reading_time = time.perf_counter()
    last_scroll_time = time.perf_counter()
    calibration_start = time.perf_counter()
    is_calibrating = True

    hand_present = False
    hand_tentatively_absent = False
    tentative_absent_start = 0.0
    last_hand_distance = 30.0

    loop_times: deque[float] = deque(maxlen=200)
    total_scroll_ticks = 0
    frames_processed = 0
    scroll_events_sent = 0

    smooth_scroll_accumulator = 0.0
    MIN_SCROLL_INTERVAL = 0.004

    last_gesture = GestureType.STILL
    last_blended_velocity = 0.0
    last_kalman_state = None

    try:
        while True:
            loop_start = time.perf_counter()

            # read from serial
            line = ""
            try:
                raw = arduino.readline()
                if raw:
                    line = raw.decode("utf-8", errors="ignore").strip()
            except (serial.SerialException, OSError):
                time.sleep(0.05)
                continue

            now = time.perf_counter()

            # parse the distance
            distance = -1.0
            if line:
                try:
                    distance = float(line)
                except ValueError:
                    distance = -1.0

            valid_reading = MIN_DISTANCE <= distance <= MAX_DISTANCE

            if valid_reading:
                dt = now - last_reading_time
                last_reading_time = now
                dt = max(dt, 0.001)
                dt = min(dt, 0.1)

                last_valid_time = now
                last_hand_distance = distance
                hand_present = True
                hand_tentatively_absent = False

                # clean up the reading
                kalman_state = kalman.update(distance, dt)
                if kalman_state is None:
                    continue

                last_kalman_state = kalman_state

                # still calibrating
                if is_calibrating:
                    elapsed = now - calibration_start

                    if not kalman.is_calibrated or elapsed < CALIBRATION_SECONDS:
                        pct = min(100, int(elapsed / CALIBRATION_SECONDS * 100))
                        bar_len = 30
                        filled = int(bar_len * pct / 100)
                        bar = "#" * filled + "." * (bar_len - filled)
                        print(f"\r  [{bar}] {pct}%", end="", flush=True)
                        continue

                    is_calibrating = False
                    print(f"\r  [{'#' * 30}] 100%")
                    print()
                    print(f"  calibrated. noise floor: +/-{kalman._calibrated_noise_floor:.2f} cm")
                    print()
                    print("  +--------------------------------------------------+")
                    print("  |  READY. move ur hand to scroll                    |")
                    print("  |                                                    |")
                    print("  |  move closer/further = scroll down/up             |")
                    print("  |  hold still = stop                                |")
                    print("  |  flick = inertial coast                           |")
                    print("  |  remove hand = coast to stop                      |")
                    print("  |                                                    |")
                    print("  |  ctrl+c to quit                                   |")
                    print("  +--------------------------------------------------+")
                    print()
                    continue

                # feed the neural net
                predictor.feed(
                    kalman_state.position,
                    kalman_state.velocity,
                    kalman_state.acceleration,
                    dt
                )

                prediction = predictor.predict()

                # blend kalman and ml predictions
                if prediction and prediction.confidence > 0.2:
                    blend = prediction.blend_factor
                    blended_velocity = (
                        (1.0 - blend) * kalman_state.velocity
                        + blend * prediction.predicted_velocity
                    )
                    gesture = prediction.gesture
                else:
                    blended_velocity = kalman_state.velocity
                    gesture = GestureType.STILL if abs(kalman_state.velocity) < 1.5 else GestureType.SLOW_SCROLL

                last_gesture = gesture
                last_blended_velocity = blended_velocity

                gesture_scale = GESTURE_SCROLL_SCALE.get(gesture, 1.0)

                # physics
                physics_out = physics.update_tracking(
                    blended_velocity * SCROLL_MULTIPLIER * gesture_scale,
                    dt
                )

                # scroll curve
                scroll_out = curve.process(
                    physics_out.scroll_velocity,
                    physics_out.scroll_delta,
                    last_hand_distance,
                    dt
                )

                # send scroll to windows
                if now - last_scroll_time >= MIN_SCROLL_INTERVAL:
                    smooth_scroll_accumulator += scroll_out.smooth_delta / SMOOTH_SCROLL_DIVISOR
                    scroll_value = int(smooth_scroll_accumulator)

                    if scroll_value != 0:
                        if abs(scroll_value) > DISTRIBUTE_THRESHOLD:
                            send_distributed_scroll(scroll_value, DISTRIBUTE_CHUNKS)
                        else:
                            send_smooth_scroll(scroll_value)

                        smooth_scroll_accumulator -= scroll_value
                        total_scroll_ticks += abs(scroll_out.ticks)
                        scroll_events_sent += 1
                        last_scroll_time = now

                frames_processed += 1

                # debug output
                if SHOW_DEBUG and now - last_debug_time >= DEBUG_INTERVAL:
                    last_debug_time = now

                    if loop_times:
                        avg_loop_ms = sum(loop_times) / len(loop_times) * 1000
                        hz = 1000.0 / avg_loop_ms if avg_loop_ms > 0 else 0
                    else:
                        avg_loop_ms = 0
                        hz = 0

                    phase_tag = {
                        ScrollPhase.IDLE: "IDLE",
                        ScrollPhase.TRACKING: "TRAK",
                        ScrollPhase.COASTING: "COAS",
                        ScrollPhase.SNAPPING: "SNAP",
                    }.get(physics_out.phase, "????")

                    gesture_tag = {
                        GestureType.STILL: "STILL",
                        GestureType.SLOW_SCROLL: "SLOW ",
                        GestureType.FAST_SCROLL: "FAST ",
                        GestureType.FLICK: "FLICK",
                        GestureType.STOPPING: "STOP ",
                    }.get(gesture, "?????")

                    conf_bar = "#" * int(kalman_state.confidence * 5) + "." * (5 - int(kalman_state.confidence * 5))

                    print(
                        f"  {phase_tag}"
                        f" D:{kalman_state.position:5.1f}cm"
                        f" V:{blended_velocity:+6.1f}"
                        f" > Scroll:{scroll_out.ticks:+4d}"
                        f" [{scroll_out.zone:6s}]"
                        f" {gesture_tag}"
                        f" [{conf_bar}]"
                        f" {hz:.0f}hz"
                    )

            else:
                # no valid reading

                if is_calibrating:
                    pass

                elif hand_present:
                    time_since_valid = now - last_valid_time

                    # ignore tiny glitches
                    if time_since_valid < HAND_GLITCH_TOLERANCE:
                        pass
                    elif time_since_valid < HAND_ABSENT_TIMEOUT:
                        if not hand_tentatively_absent:
                            hand_tentatively_absent = True
                            tentative_absent_start = now
                    else:
                        # hand is gone for real
                        hand_present = False
                        hand_tentatively_absent = False

                # inertial coasting (hand lifted, scroll keeps going)
                if not is_calibrating and not hand_present and not physics.is_idle():
                    dt_coast = now - last_reading_time
                    last_reading_time = now
                    dt_coast = max(dt_coast, 0.001)
                    dt_coast = min(dt_coast, 0.05)

                    physics_out = physics.update_coasting(dt_coast)

                    if not physics.is_idle():
                        scroll_out = curve.process(
                            physics_out.scroll_velocity,
                            physics_out.scroll_delta,
                            last_hand_distance,
                            dt_coast
                        )

                        if now - last_scroll_time >= MIN_SCROLL_INTERVAL:
                            smooth_scroll_accumulator += scroll_out.smooth_delta / SMOOTH_SCROLL_DIVISOR
                            scroll_value = int(smooth_scroll_accumulator)

                            if scroll_value != 0:
                                send_smooth_scroll(scroll_value)
                                smooth_scroll_accumulator -= scroll_value
                                total_scroll_ticks += abs(scroll_out.ticks)
                                scroll_events_sent += 1
                                last_scroll_time = now

                        if SHOW_DEBUG and now - last_debug_time >= DEBUG_INTERVAL:
                            last_debug_time = now
                            coast_pct = abs(physics_out.scroll_velocity / max(abs(physics._coast_start_velocity), 0.1)) * 100
                            print(
                                f"  COAST"
                                f" V:{physics_out.scroll_velocity:+6.1f}"
                                f" > {scroll_out.ticks:+4d}"
                                f" Energy:{coast_pct:.0f}%"
                                f" Phase:{physics_out.phase.value}"
                            )

                    else:
                        smooth_scroll_accumulator = 0.0
                        if SHOW_DEBUG:
                            print("  stopped")

                elif not is_calibrating and not hand_present and physics.is_idle():
                    predictor.reset()
                    smooth_scroll_accumulator = 0.0
                    if kalman.is_calibrated:
                        kalman.x[1] = 0.0
                        kalman.x[2] = 0.0

            # timing
            loop_end = time.perf_counter()
            loop_duration = loop_end - loop_start
            loop_times.append(loop_duration)

            target_period = 1.0 / TARGET_LOOP_HZ
            sleep_time = target_period - loop_duration
            if sleep_time > 0.0003:
                time.sleep(sleep_time * 0.75)

    except KeyboardInterrupt:
        print()
        print()
        print("  +--------------------------------------------------+")
        print("  |  sesh over                                        |")
        print("  +--------------------------------------------------+")
        print(f"  frames:       {frames_processed:>10,}")
        print(f"  scroll events:{scroll_events_sent:>10,}")
        print(f"  total ticks:  {total_scroll_ticks:>10,}")
        print(f"  outliers:     {kalman.outliers_rejected:>10,}")

        if loop_times:
            avg_ms = sum(loop_times) / len(loop_times) * 1000
            print(f"  avg loop:     {avg_ms:>7.2f} ms")
            print(f"  frequency:    {1000/avg_ms:>7.0f} hz")

        print()

    finally:
        arduino.close()


if __name__ == "__main__":
    main()