#pragma once

#include <Arduino.h>

#include "tinyml.h"

constexpr uint32_t READ_TIMEOUT_MS = 500;

// `MODEL_DIGEST` keeps the `u` suffix of its C literal; `board.py` strips it.
#define STR_(x) #x
#define STR(x) STR_(x)
constexpr char IDENTITY[] =
    "tinyml arch=" MODEL_ARCH " act=" TINYML_ACT_NAME " n_in=" STR(
        MODEL_N_IN) " n_out=" STR(MODEL_N_OUT) " digest=" STR(MODEL_DIGEST);

// `board.py` unpacks this as `<{n_out}fHH`. `aligned(4)` is not cosmetic:
// `packed` alone drops alignment to 1, and `&reply.action[0]` reaches
// `tinyml_infer` as a `float *` whose aligned VSTR would UsageFault on the M4's
// VFP. Free either way, the struct being 12 bytes regardless.
struct __attribute__((packed, aligned(4))) Reply {
  float action[MODEL_N_OUT];
  uint16_t us_read;
  uint16_t us_infer;
};

static_assert(sizeof(Reply) == 4 * MODEL_N_OUT + 4,
              "Reply must stay byte-identical to board.py's `<{n_out}fHH`");

static float obs[MODEL_N_IN];

static uint8_t xor8(const uint8_t *data, size_t n) {
  uint8_t c = 0;
  for (size_t i = 0; i < n; ++i)
    c ^= data[i];
  return c;
}

// Saturating, not wrapping: past 65 ms the step has missed its deadline anyway.
constexpr uint16_t clamp_us(uint32_t us) {
  return us > 0xFFFFu ? 0xFFFFu : (uint16_t)us;
}

// Exactly `n` bytes, or as many as arrived before the deadline.
//
// `Serial.read()` rather than `Serial.readBytes`, which is `Stream`'s and
// spends a `millis()` per byte inside `timedRead`. `millis()` costs ~9 us on
// this core and `USBSerial::read()` another ~8 (it takes the USB lock and runs
// the core's drain on every call), so readBytes ran at ~21 us a byte and the
// 246-byte request alone was 5.2 ms of a 5.9 ms control step. One clock read
// per starved poll instead of one per byte halves that. See
// docs/findings/link-latency.md; the remaining ~8 us a byte is
// `USBSerial::read()` and there is no bulk accessor for its ring.
static size_t read_exact(uint8_t *dst, size_t n) {
  const uint32_t deadline = millis() + READ_TIMEOUT_MS;
  size_t got = 0;

  while (got < n) {
    int avail = Serial.available();
    if (avail <= 0) {
      if ((int32_t)(millis() - deadline) >= 0)
        break;
      yield(); // Hands the RTOS the scheduling slot it needs to accept the rest
      continue;
    }

    size_t chunk = (size_t)avail;
    if (chunk > (n - got))
      chunk = n - got;
    for (size_t i = 0; i < chunk; ++i)
      dst[got + i] = (uint8_t)Serial.read();
    got += chunk;
  }
  return got;
}

// Sentinels in the error frame's `received` field. A real short read reports at
// most sizeof(obs), so it cannot collide. `board.py` renders both by name.
constexpr uint16_t REJECT_BAD_CHECKSUM = 0xFFFFu;
constexpr uint16_t REJECT_NO_CHECKSUM = 0xFFFEu;
static_assert(sizeof(obs) < REJECT_NO_CHECKSUM,
              "a short-read count must not reach the reject sentinels");

// Two USB full-speed frames, so a mid-frame gap in a refused request cannot
// pass for the end of one.
constexpr uint32_t IDLE_GAP_MS = 2;

// loop() dispatches on single bytes and would read a payload 'R' as a command,
// so the tail of a refused frame has to go first. An empty ring is not idle,
// the rest of the request still being in flight, hence the quiet gap; and a
// host that never stops talking would park the board here forever, hence the
// absolute deadline.
static void drain_to_idle() {
  const uint32_t deadline = millis() + READ_TIMEOUT_MS;
  uint32_t quiet_since = millis();
  while ((uint32_t)(millis() - quiet_since) < IDLE_GAP_MS &&
         (int32_t)(millis() - deadline) < 0) {
    if (Serial.available() > 0) {
      Serial.read();
      quiet_since = millis();
      continue;
    }
    yield();
  }
}

// Drain first, then answer: the host may send its next command the instant it
// sees the 'E', and that byte has to survive.
static void reject(uint16_t received) {
  drain_to_idle();

  uint8_t err[3] = {'E'};
  memcpy(&err[1], &received, sizeof(received));
  Serial.write(err, sizeof(err));
  Serial.flush();
}

// `Serial` is already begun by the core's `main` before `setup` runs; this pins
// the baud the host opens with. No `setTimeout`: nothing here uses `Stream`'s
// timeout any more, `read_exact` keeping its own deadline.
static void link_begin() { Serial.begin(500000); }

// Flushed like a reply: the host is blocked in `readline`, and a line left in
// mbed's USB buffer costs it the whole timeout.
static void link_identify() {
  Serial.println(IDENTITY);
  Serial.flush();
}

static void link_serve() {
  const size_t want = sizeof(obs);
  uint8_t checksum = 0;

  const uint32_t t_read = micros();
  const size_t got = read_exact((uint8_t *)obs, want);
  if (got != want) {
    reject((uint16_t)got);
    return;
  }
  // A separate check: `got == want` here, so reporting the count would tell
  // the host nothing was wrong.
  if (read_exact(&checksum, 1) != 1) {
    reject(REJECT_NO_CHECKSUM);
    return;
  }

  if (checksum != xor8((const uint8_t *)obs, want)) {
    reject(REJECT_BAD_CHECKSUM);
    return;
  }

  Reply reply;
  // The spans tile the frame: `us_read` up to the verified observation,
  // `us_infer` the net alone.
  const uint32_t t_infer = micros();
  tinyml_infer(obs, reply.action);
  const uint32_t t_done = micros();
  reply.us_read = clamp_us(t_infer - t_read);
  reply.us_infer = clamp_us(t_done - t_infer);

  uint8_t frame[1 + sizeof(Reply) + 1];
  frame[0] = 'A';
  memcpy(&frame[1], &reply, sizeof(reply));
  frame[sizeof(frame) - 1] = xor8(&frame[1], sizeof(reply));

  Serial.write(frame, sizeof(frame));
  // The host is blocked reading it, so a reply parked in mbed's USB buffer
  // stalls the control step.
  Serial.flush();
}
