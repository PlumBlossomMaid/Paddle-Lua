/* test_c_only.c -- 2.1.1 acceptance: paddle_c.h compiles as plain C.
 *
 * 01-c-abi.md sec.5: compile this file with a C compiler (NOT C++).
 * It may include nothing but paddle_c.h.
 */
#include "paddle_c.h"

int test_header_is_c_only(void)
{
    /* touch every declared symbol so mismatches surface at link time */
    pd_dtype d = PD_DTYPE_FLOAT32;
    int      kind = PD_PLACE_CPU;
    pd_status s = PD_OK;
    size_t   sz = pd_dtype_sizeof((int)d);
    (void)kind; (void)s; (void)sz;
    return 0;
}