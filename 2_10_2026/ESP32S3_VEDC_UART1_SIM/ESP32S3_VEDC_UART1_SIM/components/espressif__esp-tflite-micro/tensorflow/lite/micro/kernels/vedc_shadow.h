/*
 * VEDC shadow check (debug only, CONFIG_VEDC_ESPNN_SHADOW_CHECK).
 *
 * After an ESP-NN kernel has produced its output, the same layer is computed
 * again with the TFLM reference kernel into a scratch buffer and both are
 * compared. One line per layer is logged, e.g.
 *
 *   [SHADOW conv#7] in=1x10x10x8 f=48x1x1x8 out=1x10x10x48 f_mod16=3 b_mod16=3 \
 *       mismatch=4800/4800 maxdiff=97 esp_const=YES  -> BAD
 *
 * The reference result is then copied over the output, so the next layer
 * receives a correct input and every layer is judged independently. The first
 * layer marked BAD is a real kernel/alignment problem, not a propagated error.
 * Only the first VEDC_SHADOW_MAX_LOG calls per op type are logged.
 */
#ifndef VEDC_SHADOW_H_
#define VEDC_SHADOW_H_

#include "sdkconfig.h"

#ifdef CONFIG_VEDC_ESPNN_SHADOW_CHECK

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_log.h"

#ifndef VEDC_SHADOW_MAX_LOG
#define VEDC_SHADOW_MAX_LOG 1000
#endif

namespace vedc_shadow {

inline int8_t *scratch(size_t n) {
  static int8_t *p = nullptr;
  static size_t cap = 0;
  if (n > cap) {
    if (p) heap_caps_free(p);
    p = (int8_t *)heap_caps_malloc(n, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!p) p = (int8_t *)heap_caps_malloc(n, MALLOC_CAP_8BIT);
    cap = p ? n : 0;
  }
  return p;
}

/* Compare `out` (ESP-NN) with `ref` (reference); log; then out := ref. */
inline void compare_and_fix(const char *op, int *counter, const char *desc,
                            unsigned f_mod16, unsigned b_mod16, int8_t *out,
                            const int8_t *ref, size_t n) {
  size_t bad = 0;
  int maxdiff = 0;
  int8_t omin = 127, omax = -128;
  for (size_t i = 0; i < n; i++) {
    int d = (int)out[i] - (int)ref[i];
    if (d < 0) d = -d;
    if (d) {
      bad++;
      if (d > maxdiff) maxdiff = d;
    }
    if (out[i] < omin) omin = out[i];
    if (out[i] > omax) omax = out[i];
  }
  if (*counter < VEDC_SHADOW_MAX_LOG) {
    MicroPrintf("[SHADOW %s#%d] %s f_mod16=%u b_mod16=%u mismatch=%u/%u "
                "maxdiff=%d esp_const=%s -> %s",
                op, *counter, desc, f_mod16, b_mod16, (unsigned)bad,
                (unsigned)n, maxdiff, (omin == omax) ? "YES" : "no",
                bad ? "BAD" : "ok");
  }
  (*counter)++;
  memcpy(out, ref, n);
}

}  // namespace vedc_shadow

#endif  // CONFIG_VEDC_ESPNN_SHADOW_CHECK
#endif  // VEDC_SHADOW_H_
