// Dispatch loop and nothing else: `link.h` owns the wire protocol, `tinyml.h` the
// kernel, `model.h` the weights.

#include "link.h"

// No `while (!Serial)`: CDC enumerates asynchronously, so waiting for a host hangs a
// battery-powered board. The handshake is the host's, `Board.__init__` sleeping 1.5 s
// after opening the port (which resets the Nano). Shorten it and `?` lands before the
// sketch is listening.
void setup() { link_begin(); }

void loop() {
  // The only place the sketch is idle. Without the yield the poll spins at 100% CPU
  // and starves the mbed scheduler of the slot `read_exact` needs for the next
  // packet.
  int c = Serial.read();
  if (c < 0) {
    yield();
    return;
  }

  switch (c) {
  case '?':
    link_identify();
    break;
  case 'R':
    link_serve();
    break;
  default:
    break;
  }
}
