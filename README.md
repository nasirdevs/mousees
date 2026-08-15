# mousees

**the coolest scroller you have ever seen**

wave your hand over an ultrasonic sensor and it scrolls your screen. no touching anything. just vibes.

it uses a neural network that learns how you scroll, a kalman filter that makes the sensor not garbage, and a physics engine that makes it feel like an iphone. its wireless over bluetooth. its open source. its kinda insane ngl.

## ok but what does it actually do

you put an ultrasonic sensor on your desk. you wave your hand over it. your screen scrolls. thats it. thats the project.

but the scroll feels GOOD. like really good. like iphone smooth. because theres a whole pipeline of math running on your computer thats processing 200 sensor readings per second through:

1. **kalman filter** that cleans up the noisy sensor data and tracks your hand position, velocity, and acceleration
2. **neural network** that learns YOUR scroll patterns and predicts where your hand is going
3. **physics engine** with real momentum, friction, and two phase deceleration so when you flick your hand away it coasts to a stop like scrolling on your phone
4. **scroll curve** with a nonlinear response so slow = precise and fast = zooming through pages

the arduino just reads the sensor and yeets data over bluetooth. the python side does all the heavy lifting.

## what you need

| thing | what it is | cost |
|-------|-----------|------|
| arduino uno/nano | the brain on the hardware side | ~$5 |
| HC-SR04 | ultrasonic distance sensor (the eyes) | ~$1 |
| HC-05 | bluetooth module (so its wireless) | ~$3 |
| breadboard | to connect everything without soldering | ~$2 |
| jumper wires | the cables | ~$1 |

total: like $12 bro

you also need a computer running windows with python installed. thats it.

## wiring it up

```
ULTRASONIC SENSOR (HC-SR04)          ARDUINO
        VCC  ────────────────────>  5V
        GND  ────────────────────>  GND
        TRIG ────────────────────>  Pin 9
        ECHO ────────────────────>  Pin 10


BLUETOOTH (HC-05)                    ARDUINO
        VCC  ────────────────────>  5V
        GND  ────────────────────>  GND
        TXD  ────────────────────>  Pin 2
        RXD  ────────────────────>  Pin 3
```

thats literally it. 8 wires. no resistors, no capacitors, no extra components. just plug and go.

### pin summary

| component | component pin | arduino pin |
|-----------|--------------|-------------|
| HC-SR04 | VCC | 5V |
| HC-SR04 | GND | GND |
| HC-SR04 | TRIG | Pin 9 |
| HC-SR04 | ECHO | Pin 10 |
| HC-05 | VCC | 5V |
| HC-05 | GND | GND |
| HC-05 | TXD | Pin 2 |
| HC-05 | RXD | Pin 3 |

## how to set it up

### step 1: wire it

connect everything like the diagram above. double check the pins. if it doesnt work its probably a loose wire lol

### step 2: upload the arduino code

1. open `firmware.ino` in the arduino IDE
2. select your board (uno or nano)
3. select the port
4. hit upload

the arduino code is dead simple. it reads the sensor 200 times a second and sends the distance over both usb and bluetooth.

### step 3: pair the bluetooth

1. go to windows settings > bluetooth & devices > add device
2. find HC-05
3. enter pin: `1234` (or `0000` on some modules)
4. windows will create a virtual COM port for it

you can skip this step and just use usb if you dont want wireless

### step 4: install python dependencies

```bash
pip install pyserial numpy
```

thats it. just two packages. no tensorflow. no pytorch. the neural net is pure numpy.

### step 5: run it

```bash
python main.py
```

it auto detects the arduino (usb or bluetooth), calibrates for 1.5 seconds, then youre good to go. wave your hand over the sensor and watch your screen scroll.

## how it works (the nerdy part)

the whole system is a pipeline:

```
ultrasonic sensor
       |
       v
   arduino (200hz sampling)
       |
       v  (bluetooth or usb)
   python on your computer
       |
       +---> kalman filter (cleans up noise, tracks position/velocity/acceleration)
       |         |
       |         v
       +---> neural network (predicts velocity, classifies gesture)
       |         |
       |         v
       +---> physics engine (momentum, friction, two phase deceleration)
       |         |
       |         v
       +---> scroll curve (nonlinear response mapping)
       |         |
       |         v
       +---> windows SendInput API (smooth scroll output)
```

### the kalman filter (`kalman_filter.py`)

the ultrasonic sensor is noisy af. it spits out random spikes and jitter constantly. the kalman filter:

- tracks position, velocity, AND acceleration
- learns how noisy your specific sensor is during calibration
- rejects readings that are statistically impossible (outlier rejection)
- has a "drift killer" that zeros out phantom velocity when your hand is still
- adapts its noise model in real time based on how accurate its predictions are

its a 3 state adaptive kalman filter with asymmetric regime detection. sounds fancy but basically: it reacts fast when you start moving and settles slow when you stop. thats what makes it feel natural.

### the neural network (`ml_predictor.py`)

a tiny MLP (60 > 32 > 16 > 1) that runs in pure numpy:

- looks at the last 10 readings (position, velocity, acceleration, jerk, speed, velocity change = 6 features each = 60 inputs)
- predicts where your hand velocity is going
- also classifies your gesture: still, slow scroll, fast scroll, flick, or stopping
- trains itself while you use it (online learning with a replay buffer)
- uses leaky relu activations and huber loss for stable training
- has gesture hysteresis so it doesnt flip flop between states

it gets better the more you use it because it literally learns YOUR scroll patterns

### the physics engine (`physics_engine.py`)

this is what makes it feel like an iphone. real physics simulation:

- **momentum**: fast scrolls have more inertia
- **friction**: static + kinetic friction model
- **two phase deceleration**: when you lift your hand, the scroll does a fast initial slowdown then a long gentle coast (this is the premium feel)
- **flick detection**: quick hand movements get an energy boost for longer coasts
- **cosine snap to stop**: instead of abruptly stopping, it uses a cosine curve to ease to zero (no jarring halt)
- **velocity dependent mass**: faster scrolls feel "heavier" with more momentum

coast times are tuned to 1-3 seconds depending on how fast you were going. just like a real phone.

### the scroll curve (`scroll_curve.py`)

maps velocity to scroll output with 4 zones:

- **dead zone** (v < 1.2 cm/s): nothing happens. kills jitter completely
- **micro scroll** (1.2 - 3 cm/s): single tick precision for careful scrolling
- **linear** (3 - 20 cm/s): proportional and predictable
- **power** (> 20 cm/s): accelerated for speed readers

transitions between zones use cosine interpolation so they're completely invisible. also has 3 distance based sensitivity zones: close = precision, normal = regular, far = turbo.

## project structure

```
mousees/
  firmware.ino        # arduino code (reads sensor, sends over bluetooth/usb)
  main.py             # the orchestrator (runs the whole pipeline)
  kalman_filter.py    # noise filtering and state estimation
  ml_predictor.py     # neural network velocity predictor and gesture classifier
  physics_engine.py   # momentum, friction, deceleration simulation
  scroll_curve.py     # nonlinear response curve and scroll output
  README.md           # you are here
```

## controls

| gesture | what happens |
|---------|-------------|
| move hand closer | scroll down |
| move hand further | scroll up |
| hold still | stop scrolling |
| quick flick then lift | inertial coast (like swiping on phone) |
| remove hand | coast to a smooth stop |

## tuning guide (make it YOUR scroller)

ok so the defaults work pretty good out of the box but everyones sensor is a lil different and everyones hands move different. heres how to dial it in so it feels perfect for YOU.

### the vibe check (start here)

run it and just scroll around for a minute. pay attention to:

- does it jitter when your hand is still? (dead zone too small)
- does it feel sluggish when you move? (multiplier too low)
- does the coast feel too long or too short? (decay needs tweaking)
- is it scrolling when you dont want it to? (dead zone or distance range)

once you know whats off, go mess with the right knob below.

### sensitivity and speed (`main.py`)

these are the big ones. start here.

| variable | default | what it does | turn it up | turn it down |
|----------|---------|-------------|-----------|-------------|
| `SCROLL_MULTIPLIER` | 3.0 | overall scroll speed | scrolls faster, more responsive | scrolls slower, more precise |
| `MIN_DISTANCE` | 5.0 | closest hand distance (cm) | ignores more close readings | picks up closer hand positions |
| `MAX_DISTANCE` | 60.0 | furthest hand distance (cm) | bigger range | smaller more focused range |
| `HAND_ABSENT_TIMEOUT` | 0.12 | seconds before it decides ur hand is gone | more forgiving with sensor dropouts | faster to start coasting |
| `HAND_GLITCH_TOLERANCE` | 0.04 | ignores sensor glitches shorter than this | smoother but slightly less responsive | more responsive but might stutter |

**pro tip**: if your sensor has a lot of dead spots or weird angles, bump `HAND_GLITCH_TOLERANCE` up to 0.06 or 0.08. it makes a huge difference.

### killing the jitter (`scroll_curve.py`)

if its scrolling when your hand is perfectly still, this is what you want.

| variable | default | what it does |
|----------|---------|-------------|
| `DEAD_ZONE_VELOCITY` | 1.2 | velocities below this = ignored completely |
| `MICRO_THRESHOLD` | 3.0 | below this = micro scroll mode (super precise single ticks) |

if your sensor is extra noisy try bumping `DEAD_ZONE_VELOCITY` to 1.5 or even 2.0. if its super clean you can drop it to 0.8 for more sensitivity.

### scroll speed curve (`scroll_curve.py`)

this is where you make slow movements precise and fast movements zoom.

| variable | default | what it does |
|----------|---------|-------------|
| `BASE_TICKS_PER_CM_S` | 4.5 | how many scroll ticks per cm/s of hand velocity |
| `MICRO_SCALE` | 0.5 | how much output in micro scroll mode |
| `LINEAR_SCALE` | 1.0 | output multiplier in normal mode |
| `POWER_EXPONENT` | 1.4 | how aggressively fast scrolling accelerates (1.0 = linear, 2.0 = quadratic) |

want it to scroll more per hand movement? bump `BASE_TICKS_PER_CM_S` to 6 or 7. want more acceleration when going fast? try `POWER_EXPONENT` at 1.6 or 1.8. want micro scroll to be even more precise? drop `MICRO_SCALE` to 0.3.

### distance zones (`scroll_curve.py`)

the sensor range is split into three zones. close = fine control, mid = normal, far = turbo. you can change where these boundaries are and how much each zone amplifies the scroll.

| variable | default | what it does |
|----------|---------|-------------|
| `FINE_ZONE_MAX` | 15.0 | anything closer than this cm = precision mode |
| `NORMAL_ZONE_MAX` | 35.0 | between fine and this = normal mode |
| `FINE_MULTIPLIER` | 0.35 | how much to scale scroll in precision zone |
| `NORMAL_MULTIPLIER` | 1.0 | normal zone scale (baseline) |
| `FAST_MULTIPLIER` | 2.5 | turbo zone scale |

if you mostly scroll with your hand close to the sensor, maybe bump `FINE_MULTIPLIER` to 0.5 so it doesnt feel too slow. if you want turbo mode to be even crazier try `FAST_MULTIPLIER` at 3.0 or 4.0.

### the physics feel (`physics_engine.py`)

this is what makes it feel premium vs feeling like a $2 mouse wheel.

| variable | default | what it does |
|----------|---------|-------------|
| `FAST_DECAY` | 1.8 | how hard the initial deceleration hits when you lift your hand |
| `SLOW_DECAY` | 0.9 | how gentle the long coast tail is |
| `DECAY_TRANSITION_TIME` | 0.25 | seconds to blend from fast to slow decay |
| `COAST_SNAP_VELOCITY` | 1.2 | when velocity drops below this, start the smooth stop |
| `STATIC_FRICTION` | 0.5 | base friction force |
| `KINETIC_COEFF` | 0.012 | speed dependent friction |
| `FLICK_VELOCITY_THRESHOLD` | 15.0 | hand speed above this = potential flick gesture |
| `FLICK_ACCELERATION_THRESHOLD` | 50.0 | acceleration above this = its definitely a flick |

**want longer coasts?** lower `FAST_DECAY` to 1.2 and `SLOW_DECAY` to 0.5. the scroll will glide for longer after you lift your hand.

**want snappier stops?** raise `FAST_DECAY` to 2.5 and `COAST_SNAP_VELOCITY` to 2.0. it'll stop quicker.

**want flicks to go further?** lower `FLICK_VELOCITY_THRESHOLD` to 10.0 so slower movements count as flicks too.

**want it to feel heavier?** raise `BASE_MASS` to 1.5 and `MASS_VELOCITY_SCALE` to 0.015. everything will have more momentum.

### the kalman filter (`kalman_filter.py`)

you probably dont need to touch this but if youre a nerd:

| variable | default | what it does |
|----------|---------|-------------|
| `_q_position` | 0.005 | how much position uncertainty grows each step |
| `_q_velocity` | 0.3 | velocity uncertainty growth |
| `_q_acceleration` | 8.0 | acceleration uncertainty growth |
| `_stillness_threshold` | 0.8 | velocity below this = hand is still |
| `_mahalanobis_base` | 3.0 | how many standard deviations before a reading is rejected as outlier |

if the filter is rejecting too many readings (you see a lot of outliers in the debug output) try bumping `_mahalanobis_base` to 4.0 or 5.0. if its not rejecting enough noise try 2.5.

### the neural net (`ml_predictor.py`)

the network trains itself while you use it so theres not much to tune here. but:

| variable | default | what it does |
|----------|---------|-------------|
| `_base_learning_rate` | 0.002 | how fast it learns (higher = faster but less stable) |
| `_train_interval` | 15 | trains every N predictions |
| `_warmup_predictions` | 60 | predictions before it starts trusting itself |

if you want it to adapt faster to your scroll style, bump `_base_learning_rate` to 0.005. if its being weird and unstable, drop it to 0.001.

### tuning recipes

heres some preset vibes you can try:

**"i want it buttery smooth and slow"**
```
SCROLL_MULTIPLIER = 2.0
BASE_TICKS_PER_CM_S = 3.0
FAST_DECAY = 1.0
SLOW_DECAY = 0.4
DEAD_ZONE_VELOCITY = 1.5
```

**"i want speed. raw unfiltered speed."**
```
SCROLL_MULTIPLIER = 5.0
BASE_TICKS_PER_CM_S = 7.0
POWER_EXPONENT = 1.8
FAST_MULTIPLIER = 4.0
FAST_DECAY = 2.5
```

**"i want it to feel like an iphone"**
```
SCROLL_MULTIPLIER = 3.0
BASE_TICKS_PER_CM_S = 4.5
FAST_DECAY = 1.8
SLOW_DECAY = 0.9
COAST_SNAP_VELOCITY = 1.2
```
(this is basically the default lol)

**"my sensor is garbage and super noisy"**
```
DEAD_ZONE_VELOCITY = 2.5
HAND_GLITCH_TOLERANCE = 0.08
_mahalanobis_base = 4.0
_stillness_threshold = 1.2
_q_position = 0.002
```

## troubleshooting

**nothing happens when i move my hand**
- check the wiring, especially trig/echo pins
- make sure the sensor is pointing UP and your hand is above it
- check the COM port in the python output

**its super jittery**
- try holding your hand still during calibration
- increase `DEAD_ZONE_VELOCITY` in scroll_curve.py
- make sure nothing is interfering with the ultrasonic sensor (reflective surfaces, other objects)

**bluetooth wont connect**
- default pin is `1234` or `0000`
- make sure the HC-05 is blinking (not solid) which means its in pairing mode
- try the other COM port (windows creates two, use the outgoing one)

**scrolling is too fast/slow**
- adjust `SCROLL_MULTIPLIER` in main.py
- adjust `BASE_TICKS_PER_CM_S` in scroll_curve.py

**coast feels weird**
- too long: raise `FAST_DECAY` and `SLOW_DECAY`
- too short: lower them
- stops abruptly: lower `COAST_SNAP_VELOCITY`
- never fully stops: raise `COAST_SNAP_VELOCITY`

**the calibration never finishes**
- make sure the sensor can see at least 60 valid readings
- check that nothing is moving in front of the sensor during calibration
- check your wiring

## built with

- python (numpy, pyserial)
- arduino
- an ultrasonic sensor that costs less than a coffee
- too much free time
- love

## license

do whatever you want with it. MIT license. fork it, mod it, put it in a toaster, i dont care. just build cool stuff.
