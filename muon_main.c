/* muon_test.c - verificacion de ad_muon (gram-schmidt ortho) */
#include "adaptive.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int ad_muon_test(void);

int main(void) {
    int r = ad_muon_test();
    printf("muon ortho test: %s\n", r ? "FALLO" : "OK");
    return r;
}
