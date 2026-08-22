# Where the USB control step goes

Measured on a Nano 33 BLE by `tinyml-board`, 200-300 frames a run, against the shipped
61-16-8-2 model. One control step is one request: `'R'`, 61 float32 observations and an
xor8, 246 bytes out, then a 15-byte reply. The board also reports `us_read`, its own time
from the first payload byte to the verified checksum, so the host's round trip and the
device's share of it are separable.

## The chain

| step                                        | round trip | device `us_read` |
| ------------------------------------------- | ---------- | ---------------- |
| `Serial.readBytes` into a paced 64-byte host write | 5.86 ms | 5.35 ms      |
| `Serial.read()` under one deadline          | 2.87 ms    | 2.13 ms          |

Inference is 138 us throughout, unchanged and never the point.

## It was never the bus

Three host write strategies -- 64-byte chunks 50 us apart, 64-byte chunks back to back,
and one 246-byte write -- all landed within 0.03 ms of each other, and the board's own
`us_read` accounted for 5.35 ms of the 5.86. So the time was being spent on the device,
after the bytes had arrived.

A scratch sketch that reads L bytes and reports how long it waited pinned it exactly:

| payload | OUT packets | device wait |
| ------- | ----------- | ----------- |
| 1 B     | 1           | 0.06 ms     |
| 32 B    | 1           | 0.69 ms     |
| 60 B    | 1           | 1.25 ms     |
| 244 B   | 4           | 4.96 ms     |
| 500 B   | 8           | 10.92 ms    |

Linear in bytes at ~21 us each, with no step at the 64-byte packet boundaries. Bulk
endpoints have no polling interval and the packets were already in the host's URB, so
this is not USB pacing and not the 500 kbaud line rate CDC ignores anyway. It is per-byte
software cost.

## Two clocks and a mutex per byte

Timed on the device, 1,000 calls each:

| call                        | cost     |
| --------------------------- | -------- |
| `Serial.read()`             | 8.3 us   |
| `Serial.available()`        | 8.3 us   |
| `micros()`                  | 9.4 us   |
| `USBCDC::receive_nb()`      | 0.48 us  |

`USBSerial::read()` takes the USB lock and calls `onInterrupt()` -- a 256-byte stack
buffer, a second lock pair inside `_available()`, and a `receive_nb` -- for every single
byte. `Stream::readBytes` then adds a `millis()` per byte in `timedRead`, and the clock
is not cheap here either. 8.3 + 9.4 is the 21 us.

Dropping `readBytes` for a `read()` loop with one deadline check per starved poll removes
the per-byte clock: 20.6 us a byte becomes 8.7. That is the change, and it is five lines
in `read_exact`.

## The bulk path, and why it is not used

`USBCDC::receive_nb` moves a whole packet for 0.48 us and is public. Claiming packets with
it in an RX callback measured **0.65 ms** round trip for the same 246 bytes, 1.4 us a byte,
a 9x improvement over the original.

It is not shipped because it cannot be made deterministic. The core's `main` calls
`_SerialUSB.begin()` before `setup()` runs, `begin()` attaches the core's own drain
callback, and `data_rx()` runs callbacks in registration order with no detach: a sketch
callback is always second, and by then the packet is in the core's 256-byte ring. The
probe only worked because nothing ever read that ring, so it saturated and every later
packet fell through to the callback. A protocol whose speed depends on a core buffer
staying full is not a protocol, and one `Serial.read()` anywhere else would silently undo
it.

## What is left

At 8.3 us a byte the wire size is the remaining lever, and the request is 244 bytes of
float32 carrying a vector the kernel quantizes to 61 int8 values as its first act. Sending
those bytes instead measures 0.93 ms round trip. It is bit-exact -- `quantize._quantize` is
what `tinyml_quantize` reproduces, and the equality is already a test -- but it moves the
model's input quantization to the host, so the board would no longer evaluate the whole
network. Not taken.

## Why this is enough

2.87 ms is 14% of the 20 ms control interval at `CarParams.dt`, down from 29%. The link
still costs 20x the inference it carries, which is the honest shape of this deployment:
the network is small and the transport is a full-speed CDC stack tuned for terminals.
Nothing in the car's behaviour depends on it, because `tinyml-board` and `evaluate.py`
drive the board step by step rather than in real time.
