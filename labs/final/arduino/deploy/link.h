// The wire protocol, and the only thing here that talks to the host. USB CDC is
// an unframed byte stream, so every frame carries its own tag and xor8.
// `tinyml_racing/deploy/board.py` is the other half.

#pragma once

// `Serial`, `millis`, `micros`, `yield`. The sketch preamble arduino-cli generates
// includes this too; spelling it out is what lets an editor resolve the header alone.
#include <Arduino.h>
// `_SerialUSB`, the `USBSerial` object behind the `Serial` shim. Nothing else in
// this sketch may read from `Serial`: see `link_on_rx`.
#include "USB/PluggableUSBSerial.h"

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

// The core's `main` calls `_SerialUSB.begin()` before `setup()`, `begin()` attaches
// the core's own RX drain, and `data_rx()` runs callbacks in registration order with
// no detach, so ours is always second. But that drain copies
// `min(packet, rx_buffer.availableForStore())` bytes, and nothing except
// `Serial.read/available/peek` ever frees space in that 256-byte ring. Never touch it
// and it stays full, the core's drain becomes a no-op, and every packet reaches this
// callback whole -- one `receive_nb` per packet at 0.48 us instead of ~8.3 us a byte.
//
// The host fills it during the handshake (`Board._identify`): the '?' that comes back
// answered is proof the ring is full, because that is the only way it reached here.
constexpr uint32_t RX_RING = 512;
static_assert((RX_RING & (RX_RING - 1)) == 0, "RX_RING must be a power of two");
static_assert(RX_RING >= 2 * (2 + sizeof(obs)),
              "RX_RING must hold two requests, so a pipelining host never stalls");
static uint8_t rx_ring[RX_RING];
static volatile uint32_t rx_head = 0;
static volatile uint32_t rx_tail = 0;

static void link_on_rx() {
  for (;;) {
    const uint32_t space = RX_RING - (rx_head - rx_tail);
    if (space == 0)
      return;  // backpressure: mbed re-arms only once its buffer is empty, so we NAK
    const uint32_t off = rx_head & (RX_RING - 1);
    uint32_t span = RX_RING - off;
    if (span > space)
      span = space;
    uint32_t got = 0;
    _SerialUSB.receive_nb(rx_ring + off, span, &got);
    if (got == 0)
      return;
    rx_head += got;
  }
}

static uint32_t rx_available() { return rx_head - rx_tail; }

static void rx_take(uint8_t *dst, uint32_t n) {
  const uint32_t off = rx_tail & (RX_RING - 1);
  const uint32_t span = RX_RING - off;
  if (n <= span) {
    memcpy(dst, rx_ring + off, n);
  } else {
    memcpy(dst, rx_ring + off, span);
    memcpy(dst + span, rx_ring, n - span);
  }
  rx_tail += n;
}

static bool link_poll(uint8_t *cmd) {
  if (rx_available() == 0)
    return false;
  rx_take(cmd, 1);
  return true;
}

static size_t read_exact(uint8_t *dst, size_t n) {
  const uint32_t deadline = millis() + READ_TIMEOUT_MS;
  while (rx_available() < n) {
    if ((int32_t)(millis() - deadline) >= 0) {
      const uint32_t partial = rx_available();
      rx_take(dst, partial);
      return partial;
    }
    yield();
  }
  rx_take(dst, (uint32_t)n);
  return n;
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
    if (rx_available() > 0) {
      rx_tail = rx_head;
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
static void link_begin() {
  _SerialUSB.attach(link_on_rx);
  Serial.begin(500000);
}

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
