#include "link.h"

// No `while (!Serial)`: CDC enumerates asynchronously, so waiting for a host
// hangs a battery-powered board. The handshake is the host's, in
// `Board.__init__`.
void setup() { link_begin(); }

void loop() {
  // The only place the sketch is idle. Without the yield the poll starves the
  // mbed scheduler of the slot `read_exact` needs for the next packet.
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
