// status.cc -- C++-exception boundary for the C ABI layer.
//
// Stage: P1 / M1 (WORKPLAN 2.1.2). Layout.md sec.3: this file comes second,
// right after paddle_c.h, because an exception escaping the C boundary is a
// process-killer and retrofitting try/catch onto already-written code is
// exactly how guards get missed.
//
// Design: docs/migration/plan/modules/01-c-abi.md sec.3.2
//
// The error slot is thread_local (NOT a global): Lanes threads (P13) will
// call pd_* concurrently, and a shared buffer would race. Paddle itself is
// able to throw any std::exception subtype; we record everything as PD_ERROR
// (PD_OOM / PD_INVALID_ARG, if ever needed, come from explicit checks in the
// callers before they touch fallible code paths).

#include <exception>
#include <string>

#include "capi_impl.h"

namespace {

thread_local pd_status t_last_status = PD_OK;
thread_local std::string t_last_error;

}  // namespace

extern "C" {

pd_status pd_last_status(void) { return t_last_status; }

const char* pd_last_error(void) { return t_last_error.c_str(); }

}  // extern "C"

void pd_set_error(pd_status st, const char* msg) {
  t_last_status = st;
  t_last_error = msg != nullptr ? msg : "unknown error";
}