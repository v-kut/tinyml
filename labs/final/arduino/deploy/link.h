// The wire protocol, and the only thing here that talks to the host. USB CDC is an
// unframed byte stream, so every frame carries its own tag and xor8.
// `tinyml_racing/deploy/board.py` is the other half.

#pragma once

#include "tinyml.h"

// `constexpr`, so no storage unless someone takes an address.
constexpr uint32_t READ_TIMEOUT_MS = 500;

// Built by the preprocessor, so there is one spelling of it. `MODEL_DIGEST` keeps
// the `u` suffix of the C literal; the host strips it.
#define STR_(x) #x
#define STR(x) STR_(x)
constexpr char IDENTITY[] =
    "tinyml arch=" MODEL_ARCH " act=" TINYML_ACT_NAME " n_in=" STR(
        MODEL_N_IN) " n_out=" STR(MODEL_N_OUT) " digest=" STR(MODEL_DIGEST);

// The actions, then the two timers the board keeps. The host unpacks this layout as
// `<{n_out}fHH`. `packed` pins the wire layout; `aligned(4)` keeps it safe, since
// packed alone drops alignment to 1 and `&reply.action[0]` reaches `tinyml_infer` as
// a `float *` written with an aligned VSTR, a UsageFault on the M4's VFP rather than
// a slow path. Free: the struct is 12 bytes either way.
struct __attribute__((packed, aligned(4))) Reply {
  float action[MODEL_N_OUT];
  uint16_t us_read;
  uint16_t us_infer;
};

// The host's `<{n_out}fHH` is a literal, so a field added here would surface as
// noise rather than an error. This is the only guard.
static_assert(sizeof(Reply) == 4 * MODEL_N_OUT + 4,
              "Reply must stay byte-identical to board.py's `<{n_out}fHH`");

static float obs[MODEL_N_IN];

static uint8_t xor8(const uint8_t *data, size_t n) {
  uint8_t c = 0;
  for (size_t i = 0; i < n; ++i)
    c ^= data[i];
  return c;
}

// Saturating, not wrapping: past 65 ms the step has missed its deadline anyway, so
// the host only needs to see "pegged".
constexpr uint16_t clamp_us(uint32_t us) {
  return us > 0xFFFFu ? 0xFFFFu : (uint16_t)us;
}

// Drains exactly `n` bytes, or as many as arrive before the deadline.
static size_t read_exact(uint8_t *dst, size_t n) {
  const uint32_t deadline = millis() + READ_TIMEOUT_MS;
  size_t got = 0;

  while (got < n && (int32_t)(millis() - deadline) < 0) {
    int avail = Serial.available();
    if (avail <= 0) {
      yield(); // Hands the RTOS the scheduling slot it needs to accept the rest
      continue;
    }

    // Bulk read from the ring buffer. The clamp to `avail` keeps the deadline
    // honest: without it `readBytes` blocks on `Stream::_timeout` for bytes that
    // never arrived, which `link_begin` pins to READ_TIMEOUT_MS as well.
    size_t chunk = (size_t)avail;
    if (chunk > (n - got)) {
      chunk = n - got;
    }

    got += Serial.readBytes((char *)(dst + got), chunk);
  }
  return got;
}

// Sentinels in the error frame's `received` field. A real short read reports at most
// sizeof(obs), so it cannot collide. `board.py` renders both by name.
constexpr uint16_t REJECT_BAD_CHECKSUM = 0xFFFFu;
constexpr uint16_t REJECT_NO_CHECKSUM = 0xFFFEu;
static_assert(sizeof(obs) < REJECT_NO_CHECKSUM,
              "a short-read count must not reach the reject sentinels");

// Silence that counts as the end of a refused frame. Two milliseconds is two USB
// full-speed frames and 40x `board.py`'s pacing, so a mid-frame gap cannot fool it.
constexpr uint32_t IDLE_GAP_MS = 2;

// The tail of a refused frame must go before loop() runs again, since loop()
// dispatches on single bytes and would read a payload 'R' as a command. An empty ring
// is not idle, because the rest of the request is still in flight, so this waits for
// a quiet gap. The gap alone is no bound: a host that never stops talking would park
// the board here and never let its 'E' frame out, hence the absolute deadline.
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

// Draining before the error frame, not after: the host may send its next command the
// instant it sees the 'E', and that byte has to survive.
static void reject(uint16_t received) {
  drain_to_idle();

  uint8_t err[3] = {'E'};
  memcpy(&err[1], &received, sizeof(received));
  Serial.write(err, sizeof(err));
  Serial.flush();
}

// `Serial.setTimeout` bounds `readBytes` in `read_exact`; `Stream`'s 1000 ms default
// would outlast READ_TIMEOUT_MS.
static void link_begin() {
  Serial.begin(500000);
  Serial.setTimeout(READ_TIMEOUT_MS);
}

// Flushed like a reply: the host is blocked in `readline`, and a line left in the USB
// buffer costs it the whole timeout.
static void link_identify() {
  Serial.println(IDENTITY);
  Serial.flush();
}

// One control step: read the observation, run the net, answer with the actions and
// the two timings the host cannot measure from its side.
static void link_serve() {
  const size_t want = sizeof(obs);
  uint8_t checksum = 0;

  const uint32_t t_read = micros();
  const size_t got = read_exact((uint8_t *)obs, want);
  if (got != want) {
    reject((uint16_t)got);
    return;
  }
  // A separate sentinel: `got == want` here, so reporting the count would tell the
  // host nothing was wrong.
  if (read_exact(&checksum, 1) != 1) {
    reject(REJECT_NO_CHECKSUM);
    return;
  }

  if (checksum != xor8((const uint8_t *)obs, want)) {
    reject(REJECT_BAD_CHECKSUM);
    return;
  }

  Reply reply;
  // The spans tile the frame: us_read covers request to verified observation,
  // us_infer the net alone.
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
  // On the wire before loop() runs again: the host is blocked reading it, so a reply
  // parked in mbed's USB buffer stalls the control step.
  Serial.flush();
}
