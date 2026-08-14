// Inference kernel for the quantized racing actor.
//
// Per layer:
//     q[i]   = clamp(lrintf(h[i] * inv_sx), -127, 127)      int8
//     acc[j] = sum_i w[j*n_in + i] * q[i]                   int32, exact
//     h[j]   = b[j] + m[j] * (float)acc[j]                  float32
//     h[j]   = act(h[j])                                    except the last
//     layer

#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

// Numbered, not named: the preprocessor reads an unknown identifier as 0, so a
// `model.h` from another schema matches nothing and hits the `#error`.
#define TINYML_TANH 1

#include "model.h"

#if MODEL_ACTIVATION != TINYML_TANH
#error "MODEL_ACTIVATION must be TINYML_TANH: this kernel implements tanh only"
#endif

// Reported by IDENTITY, compared against `QuantModel.activation`.
#define TINYML_ACT_NAME "tanh"

// tanh from a 257-knot table, 1/32 apart over [0, 8], linear between knots.
// Interpolation error peaks at 9.4e-5, 84x below the 1/127 LSB this is requantized
// to one layer later (docs/findings/kernel-speed.md). Called directly: the trunk is
// `nn.Tanh` and `export.extract_actor` refuses anything else.
#define TINYML_TANH_N 256
#define TINYML_TANH_SCALE 32.0f // N / 8.0f: a power of two, so knots are exact

static const float tinyml_tanh_lut[TINYML_TANH_N + 1] = {
    0.0f,         0.0312398318f, 0.0624187477f, 0.0934763029f, 0.124352999f,
    0.154990733f, 0.185333207f,  0.215326339f,  0.244918659f,  0.27406159f,
    0.302709728f, 0.330821127f,  0.3583574f,    0.385283977f,  0.411570042f,
    0.437188774f, 0.462117195f,  0.486336052f,  0.509829998f,  0.53258729f,
    0.554599702f, 0.575862408f,  0.596373558f,  0.616134405f,  0.635149002f,
    0.653423607f, 0.670967102f,  0.687790215f,  0.703905582f,  0.71932745f,
    0.734071493f, 0.748154461f,  0.761594176f,  0.774409175f,  0.786618829f,
    0.798242748f, 0.809301078f,  0.819814026f,  0.829801917f,  0.839285076f,
    0.848283648f, 0.856817603f,  0.864906669f,  0.872570038f,  0.879826725f,
    0.886695147f, 0.893193364f,  0.899338782f,  0.905148208f,  0.910638213f,
    0.915824533f, 0.920722306f,  0.925346196f,  0.929710269f,  0.933827996f,
    0.937712312f, 0.941375494f,  0.944829404f,  0.948085248f,  0.951153815f,
    0.954045236f, 0.956769288f,  0.959335268f,  0.961751878f,  0.964027584f,
    0.966170132f, 0.968187213f,  0.9700858f,    0.971872747f,  0.973554313f,
    0.975136697f, 0.976625443f,  0.978026092f,  0.979343653f,  0.980583012f,
    0.9817487f,   0.982845008f,  0.98387599f,   0.984845519f,  0.985757113f,
    0.986614287f, 0.987420201f,  0.988177896f,  0.988890171f,  0.98955977f,
    0.990189195f, 0.99078089f,   0.991337001f,  0.991859734f,  0.992351055f,
    0.992812812f, 0.993246794f,  0.993654668f,  0.994037926f,  0.994398177f,
    0.994736671f, 0.995054781f,  0.995353699f,  0.995634615f,  0.995898545f,
    0.99614656f,  0.996379614f,  0.996598601f,  0.996804357f,  0.996997654f,
    0.997179329f, 0.997349977f,  0.997510314f,  0.997660995f,  0.997802556f,
    0.997935534f, 0.998060524f,  0.998177886f,  0.998288214f,  0.998391867f,
    0.998489201f, 0.998580694f,  0.998666584f,  0.998747349f,  0.998823166f,
    0.998894453f, 0.998961389f,  0.999024272f,  0.9990834f,    0.999138892f,
    0.999191046f, 0.999240041f,  0.999286056f,  0.999329329f,  0.999369919f,
    0.999408126f, 0.999443948f,  0.999477625f,  0.999509275f,  0.999539018f,
    0.999566913f, 0.999593198f,  0.999617815f,  0.999640942f,  0.999662697f,
    0.999683142f, 0.999702334f,  0.999720395f,  0.999737322f,  0.999753237f,
    0.999768198f, 0.999782205f,  0.999795437f,  0.999807835f,  0.999819458f,
    0.999830425f, 0.999840677f,  0.999850333f,  0.999859393f,  0.999867916f,
    0.999875903f, 0.999883413f,  0.999890506f,  0.999897122f,  0.999903381f,
    0.999909222f, 0.999914706f,  0.999919891f,  0.999924779f,  0.999929309f,
    0.9999336f,   0.999937594f,  0.999941409f,  0.999944925f,  0.999948263f,
    0.999951422f, 0.999954343f,  0.999957144f,  0.999959707f,  0.999962151f,
    0.999964476f, 0.999966621f,  0.999968648f,  0.999970555f,  0.999972343f,
    0.999974012f, 0.999975562f,  0.999977052f,  0.999978483f,  0.999979734f,
    0.999980986f, 0.999982119f,  0.999983251f,  0.999984264f,  0.999985218f,
    0.999986112f, 0.999986947f,  0.999987721f,  0.999988437f,  0.999989152f,
    0.999989808f, 0.999990404f,  0.999991f,     0.999991536f,  0.999992073f,
    0.999992549f, 0.999992967f,  0.999993384f,  0.999993801f,  0.999994159f,
    0.999994516f, 0.999994874f,  0.999995172f,  0.99999547f,   0.999995768f,
    0.999996006f, 0.999996245f,  0.999996483f,  0.999996662f,  0.999996901f,
    0.999997079f, 0.999997258f,  0.999997437f,  0.999997556f,  0.999997735f,
    0.999997854f, 0.999997973f,  0.999998093f,  0.999998212f,  0.999998331f,
    0.99999845f,  0.999998569f,  0.999998629f,  0.999998748f,  0.999998808f,
    0.999998868f, 0.999998927f,  0.999998987f,  0.999999046f,  0.999999106f,
    0.999999166f, 0.999999225f,  0.999999285f,  0.999999344f,  0.999999344f,
    0.999999404f, 0.999999464f,  0.999999464f,  0.999999523f,  0.999999523f,
    0.999999583f, 0.999999583f,  0.999999642f,  0.999999642f,  0.999999642f,
    0.999999702f, 0.999999702f,  0.999999702f,  0.999999762f,  0.999999762f,
    0.999999762f, 0.999999762f,
};

// Mirrors `quantize.tanh_lut` operation for operation, which is what the two-step
// clamp is for: `t` to N, then the *index* to N-1. At |v| == 8.0f (exact, the scale
// is a power of two) that gives i = 255, f = 1.0, interpolating to lut[256].
// Clamping only `t` leaves i == 256 and loads lut[257] off the end; the value still
// comes out right, since the stray knot meets a zero fraction, so only a sanitizer
// sees it (`test_c_tanh_reads_nothing_past_the_end_of_the_table`).
//
// NaN loses `t > N` and is mapped to 0 first, matching the Python side, where
// `(int)t` would be undefined here and INT32_MIN there. +-Inf needs no guard: the
// magnitude clamp saturates it to the last knot.
static inline float tinyml_tanh(float v) {
  float t = fabsf(v) * TINYML_TANH_SCALE;
  if (isnan(t))
    t = 0.0f;
  if (t > (float)TINYML_TANH_N)
    t = (float)TINYML_TANH_N;

  int i = (int)t;
  if (i > TINYML_TANH_N - 1)
    i = TINYML_TANH_N - 1;

  const float lo = tinyml_tanh_lut[i];
  const float hi = tinyml_tanh_lut[i + 1];
  float slope_approximation = lo + (t - (float)i) * (hi - lo);

  return copysignf(slope_approximation, v);
}

// Rounds half to even like NumPy's `rint`; `roundf` rounds half away from zero and
// would disagree on exactly the ties a random test never generates. -128 is unused:
// symmetric scales keep the grid centred on zero.
//
// Two implementations, one result. `lrintf` is a library call on this target, one
// per input element per layer, and no math flag reduces it. VCVTR is the
// instruction it calls out to: it rounds by FPSCR (round-half-even), and ARMv7-M
// conversion already gives NaN as 0 and saturates out of range, so the rails become
// integer compares afterwards instead of float compares before, each of which cost
// a VMRS. Same value either way, so the paths stay bit-identical: `tests/ckernel`
// diffs the portable one, `tinyml-board` diffs this one.
#ifndef TINYML_QUANT_VFP
#if defined(__ARM_FP) && (__ARM_FP & 4)
#define TINYML_QUANT_VFP 1
#else
#define TINYML_QUANT_VFP 0
#endif
#endif

#if TINYML_QUANT_VFP
static inline int8_t tinyml_quantize(float v, float inv_scale) {
  float s = v * inv_scale;
  int32_t r;
  // VCVTR writes the integer into the FPU register, hence the VMOV.
  __asm("vcvtr.s32.f32 %[s], %[s]\n\tvmov %[r], %[s]" : [r] "=r"(r), [s] "+t"(s));
  if (r > 127)
    r = 127;
  if (r < -127)
    r = -127;
  return (int8_t)r;
}
#else
static inline int8_t tinyml_quantize(float v, float inv_scale) {
  const float s = v * inv_scale;
  // NaN loses both comparisons, so it is mapped here as `quantize._quantize` does.
  // The rails run while the value is float: an out-of-range float-to-int conversion
  // is undefined in C, and `lrintf`'s `long` is 64-bit on the host.
  if (isnan(s))
    return 0;
  if (s > 127.0f)
    return 127;
  if (s < -127.0f)
    return -127;
  return (int8_t)lrintf(s);
}
#endif

// The only loop that matters: layer 0 dominates the MAC count because it is widest.
//
// SMLAD does two 16x16 MACs into one int32 accumulator, so four int8 products cost
// two SMLADs plus four SXTB16s to widen the bytes. Written as ACLE intrinsics, which
// is why the build requires a current arm-none-eabi (`build.py:ACLE_PROBE`; gcc 7's
// `arm_acle.h` is unusable). `memcpy` of four bytes is the portable spelling of the
// unaligned word load ARMv7-M does in one LDR, and the rows are contiguous.
//
// `TINYML_DOT_DSP` pins what the target picked. 0 forces the scalar path, and x86
// leaves `__ARM_FEATURE_DSP` undefined, so `tests/ckernel` diffs the fallback and a
// board verifies the SIMD path.
#ifndef TINYML_DOT_DSP
#if defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1)
#define TINYML_DOT_DSP 1
#else
#define TINYML_DOT_DSP 0
#endif
#endif

#if TINYML_DOT_DSP
#include <arm_acle.h>
static inline int32_t tinyml_dot(const int8_t *w, const int8_t *q, int n) {
  int32_t acc = 0;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    uint32_t wv, qv;
    memcpy(&wv, w + i, 4);
    memcpy(&qv, q + i, 4);
    // Bytes 0,2 in one pair, 1,3 in the other. The rotate is SXTB16's own operand
    // shift, so it is free, and both operands share the layout, so products pair up.
    acc = __smlad(__sxtb16(wv), __sxtb16(qv), acc);
    acc = __smlad(__sxtb16(__ror(wv, 8)), __sxtb16(__ror(qv, 8)), acc);
  }
  for (; i < n; ++i)
    acc += (int32_t)w[i] * (int32_t)q[i];
  return acc;
}
#else
static inline int32_t tinyml_dot(const int8_t *w, const int8_t *q, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; ++i)
    acc += (int32_t)w[i] * (int32_t)q[i];
  return acc;
}
#endif

// `in` is MODEL_N_IN raw observations, `out` receives MODEL_N_OUT clipped actions,
// both caller-owned. The layer buffers are static, so one caller at a time.
static void tinyml_infer(const float *in, float *out) {
  static float buf_a[MODEL_MAX_WIDTH];
  static float buf_b[MODEL_MAX_WIDTH];
  static int8_t q[MODEL_MAX_WIDTH];

  const float *h = in;
  float *dst = buf_a;

  for (int l = 0; l < MODEL_N_LAYERS; ++l) {
    const int n_in = model_dims[l];
    const int n_out = model_dims[l + 1];
    const int8_t *w = model_w[l];
    const float *m = model_m[l];
    const float *b = model_b[l];
    const float inv_sx = model_inv_sx[l];
    const int last = (l == MODEL_N_LAYERS - 1);

    for (int i = 0; i < n_in; ++i)
      q[i] = tinyml_quantize(h[i], inv_sx);

    for (int j = 0; j < n_out; ++j) {
      const int32_t acc = tinyml_dot(w + (int32_t)j * n_in, q, n_in);
      const float v = b[j] + m[j] * (float)acc;
      dst[j] = last ? v : tinyml_tanh(v);
    }

    h = dst;
    dst = (dst == buf_a) ? buf_b : buf_a;
  }

  for (int j = 0; j < MODEL_N_OUT; ++j) {
    float v = h[j];
    if (v > MODEL_CLIP)
      v = MODEL_CLIP;
    if (v < -MODEL_CLIP)
      v = -MODEL_CLIP;
    out[j] = v;
  }
}
