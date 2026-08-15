// mousees firmware
// this is the arduino side. it literally just goes brr
// with the ultrasonic sensor and yeets distance readings
// over bluetooth + usb at like 200 times a second
//
// the python side does all the big brain stuff
// kalman filters, neural nets, physics, the whole shebang
// arduino just vibes and sends numbers
//
// pins:
//   ultrasonic trig > pin 9
//   ultrasonic echo > pin 10
//   hc05 tx > pin 2
//   hc05 rx > pin 3

#include <SoftwareSerial.h>

const int TRIG_PIN = 9;
const int ECHO_PIN = 10;

// bluetooth pins (hc-05)
const int BT_RX_PIN = 2; // hc05 tx goes here
const int BT_TX_PIN = 3; // hc05 rx goes here

SoftwareSerial BTSerial(BT_RX_PIN, BT_TX_PIN);

// if the reading is outside these bounds its literally impossible
const float ABS_MIN_CM = 2.0;
const float ABS_MAX_CM = 400.0;

// how fast we sample (5000 microseconds = 200hz)
const unsigned long SAMPLE_INTERVAL_US = 5000;

unsigned long lastSampleTime = 0;

void setup() {
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);

    Serial.begin(115200);    // usb
    BTSerial.begin(9600);    // bluetooth

    delay(100);
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(5);
}

float readDistanceCm() {
    // ping the sensor
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);

    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);

    digitalWrite(TRIG_PIN, LOW);

    // wait for echo (30ms timeout)
    long duration = pulseIn(ECHO_PIN, HIGH, 30000);

    // got nothing back
    if (duration == 0) {
        return -1.0;
    }

    // speed of sound math to get centimeters
    float distance = duration * 0.0343 / 2.0;

    // nah thats not real
    if (distance < ABS_MIN_CM || distance > ABS_MAX_CM) {
        return -1.0;
    }

    return distance;
}

void loop() {
    unsigned long now = micros();

    // keep it steady at 200hz
    if (now - lastSampleTime < SAMPLE_INTERVAL_US) {
        return;
    }

    lastSampleTime = now;

    float distance = readDistanceCm();

    // blast it out over usb and bluetooth
    Serial.println(distance, 2);
    BTSerial.println(distance, 2);
}
