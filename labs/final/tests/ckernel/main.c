// Host harness for arduino/deploy/tinyml.h.
//
// Compiles the exact kernel the Nano runs against native gcc.
//
// Default mode: reads whitespace-separated float32 observations from stdin
// and prints one line of actions per frame. `tests/test_quantized_model.py`
// diffs that against the NumPy emulator in `deploy/quantize.py`, which is the
// only way to prove the two implementations of the quantized model agree
// without a board plugged in.
//
// `tanh` mode: reads one float per line and prints `tinyml_tanh` of it. The
// inference path requantizes every activation to int8, so a wrong table read
// hides behind the +-127 rail; this exposes the activation itself, which is
// what makes the table's edge -- exactly |x| == 8.0f -- observable.

#include <stdio.h>
#include <string.h>

#include "tinyml.h"

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "tanh") == 0) {
    float v;
    while (scanf("%f", &v) == 1) printf("%.9g\n", (double)tinyml_tanh(v));
    return 0;
  }

  float in[MODEL_N_IN];
  float out[MODEL_N_OUT];

  for (;;) {
    for (int i = 0; i < MODEL_N_IN; ++i) {
      // A frame boundary is the only place the stream may end. Anywhere else a
      // failed conversion means a malformed or truncated token, and returning
      // 0 there dropped the frame silently: the Python side then diffed a
      // short array and reported a shape mismatch instead of a parse error.
      if (scanf("%f", &in[i]) != 1) {
        if (i == 0 && feof(stdin)) return 0;
        fprintf(stderr, "ckernel: partial frame, got %d of %d values\n", i,
                MODEL_N_IN);
        return 1;
      }
    }
    tinyml_infer(in, out);
    for (int j = 0; j < MODEL_N_OUT; ++j) {
      printf("%s%.9g", j ? " " : "", (double)out[j]);
    }
    printf("\n");
  }
}
