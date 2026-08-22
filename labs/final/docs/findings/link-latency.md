# Where the USB control step goes

Measured on a Nano 33 BLE by `tinyml-board`, 200-2,000 frames a run, against the shipped
61-16-8-2 model. One control step is one request: `'R'`, 61 float32 observations and an
xor8, 246 bytes out, then a 15-byte reply. The board also reports `us_read`, its own time
from the first payload byte to the verified checksum, so the host's round trip and the
device's share of it are separable.

## The chain

| read path                                          | round trip | device `us_read` |
| -------------------------------------------------- | ---------- | ---------------- |
| `Serial.readBytes`                                 | 5.86 ms    | 5.35 ms          |
| `Serial.read()` under one deadline                 | 2.87 ms    | 2.13 ms          |
| `USBCDC::receive_nb` from an RX callback           | 0.85 ms    | 0.41 ms          |

Inference is ~138 us throughout, unchanged and never the point. Every row is bit-identical
to the emulator at 5.960e-08, because none of this touches the model.

## It was never the bus

Three host write strategies -- 64-byte chunks 50 us apart, 64-byte chunks back to back,
and one 246-byte write -- all landed within 0.03 ms of each other, and the board's own
`us_read` accounted for 5.35 ms of the 5.86. The time was going on the device, after the
bytes had arrived.

A scratch sketch that reads L bytes and reports how long it waited pinned it: linear in
bytes at ~21 us each (1 B / 0.06 ms, 60 B / 1.25 ms, 244 B / 4.96 ms, 500 B / 10.92 ms),
with no step at the 64-byte packet boundaries. Bulk endpoints have no polling interval and
the packets were already in the host's URB, so this is not USB pacing and not the 500 kbaud
line rate CDC ignores anyway. It is per-byte software cost.

## Two clocks and a mutex per byte

Timed on the device, 1,000 calls each:

| call                   | cost    |
| ---------------------- | ------- |
| `Serial.read()`        | 8.3 us  |
| `Serial.available()`   | 8.3 us  |
| `micros()`             | 9.4 us  |
| `USBCDC::receive_nb()` | 0.48 us |

`USBSerial::read()` takes the USB lock and calls `onInterrupt()` -- a 256-byte stack
buffer, a second lock pair inside `_available()`, and a `receive_nb` -- for every single
byte. `Stream::readBytes` then adds a `millis()` per byte in `timedRead`, and the clock is
not cheap here either. 8.3 + 9.4 is the 21 us; dropping `readBytes` for a `read()` loop
with one deadline check per starved poll gets 8.7, and 5.86 ms becomes 2.87.

## Taking the packet instead of the byte

`receive_nb` moves a whole packet for 0.48 us, 17x cheaper per call and ~500x per byte.
The obstacle is that the core gets to the packet first: `cores/arduino/main.cpp` calls
`_SerialUSB.begin()` before `setup()` runs, `begin()` attaches the core's own drain
callback, `data_rx()` runs callbacks in registration order, and there is no detach. A
sketch callback is always second, and normally finds the endpoint already emptied into the
core's 256-byte ring.

The way through is that the core's drain copies
`min(packet, rx_buffer.availableForStore())` bytes, and the only things that ever free
space in that ring are `Serial.read`, `available` and `peek`. A sketch that calls none of
them fills the ring exactly once and then keeps it full forever, at which point the core's
callback copies nothing, leaves the endpoint un-drained, and every packet reaches the
sketch's own callback whole. `link.h` owns the receive path outright: a 512-byte ring
filled by `link_on_rx` and drained by `read_exact`, with the 256 bytes wedged in the core
the only cost.

Priming is the handshake. `Board._identify` writes 256 bytes of `\0` -- one write, whole
packets, exactly the ring -- then `'?'` on its own, and repeats up to eight times. On a
cold board the first round answers. The answer is not merely a handshake: reaching the
sketch at all is *only* possible once the ring is full, so an identified board is a proof
that the fast path is live. A board that answers early (an older sketch, or a core that
orders the callbacks the other way) is correct and merely slower, and one that never
answers fails loudly in `Board.__init__`.

Two details make it safe rather than clever. Copying less than the endpoint holds is the
backpressure: mbed re-arms only once its buffer is empty, so a full ring simply NAKs the
host -- 738 bytes written in one go (three pipelined requests) are served in order, at
0.70 ms each. And if the invariant ever broke, the split would land mid-frame and the xor8
would catch it: the failure is a `ProtocolError`, not a wrong action.

Measured cold, through `tinyml-board`: **0.85 ms mean, 1.03 ms max**, `us_read` 411 us,
and 2,000 consecutive frames with zero rejections. The reject paths were re-checked
against the new buffer: a truncated payload reports its short count, a missing checksum
byte and a corrupted one report their sentinels, and the link resyncs on the next frame.

## The pacing came out

`Board._write_paced` split the request into 64-byte writes 50 us apart, because the old
sketch could not drain the core's ring fast enough to keep a single write from losing its
tail. Nothing about that survives: the sketch now takes whole packets and NAKs when its
own ring is full, which is backpressure the host cannot outrun however it writes.

Interleaved, 3,000 frames each way, paced against one unbroken `write`:

| host write | mean     | p99      | p99.9    | max      |
| ---------- | -------- | -------- | -------- | -------- |
| 64 B paced | 0.905 ms | 1.101 ms | 1.548 ms | 3.517 ms |
| one write  | 0.837 ms | 0.988 ms | 1.091 ms | 1.343 ms |

Unpaced wins on every statistic, including the tail the pacing was latterly justified by,
and 6,000 unpaced frames plus the reject cases ran without a failure. So `infer` writes
the frame and flushes, and the constants, the method and its 50 us busy-wait are gone.

## What is left

0.85 ms is 4.3% of the 20 ms control interval, down from 29%. Of it, ~0.4 ms is host
syscalls and USB scheduling -- a 1-byte command round trip is 0.36 ms -- and 0.41 ms is
the device, mostly the 246 bytes still being memcpy'd twice. The remaining lever is the
wire itself: the request spends 244 bytes of float32 on a vector the kernel quantizes to
61 int8 values as its first act, and sending those instead measured 0.93 ms back when a
byte cost 8.7 us, so it would be worth ~0.2 ms now. It is bit-exact, but it moves the
model's input quantization to the host, and the board would no longer evaluate the whole
network. Not taken, and at this point not worth taking.
