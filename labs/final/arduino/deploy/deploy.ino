#include "link.h"

// No `while (!Serial)`: CDC enumerates asynchronously, so waiting for a host
// hangs a battery-powered board. The handshake is the host's, in
// `Board.__init__`.
void setup() { link_begin(); }

void loop() {
  // The only place the sketch is idle. Without the yield the poll starves the
  // mbed scheduler of the slot `read_exact` needs for the next packet.
  uint8_t c;
  if (!link_poll(&c)) {
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
