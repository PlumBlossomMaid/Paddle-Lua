/* capi_impl.h -- internal shared header for csrc/capi (NOT part of the ABI).
 *
 * Declares the error-slot setter and the PD_GUARD exception-boundary macro
 * used by every .cc in this directory. This header is C++-only; it never
 * appears in paddle_c.h (which must stay pure C).
 *
 * Design: docs/migration/plan/modules/01-c-abi.md sec.3.2
 *   - every exported C function is wrapped in PD_GUARD;
 *   - an exception must never cross the C ABI boundary (UB, on Windows it
 *     usually takes the process down);
 *   - the error slot is thread_local so Lanes threads (P13) never corrupt
 *     each other's last-error.
 */

#ifndef PADDLE_LUA_CAPI_IMPL_H
#define PADDLE_LUA_CAPI_IMPL_H

#include <exception>

extern "C" {
#include "paddle_c.h"
}

/* Internal: store status+message into the thread-local slot.
 * Declared here, defined in status.cc. */
void pd_set_error(pd_status st, const char* msg);

/* Wrap the body of an exported function. On exception, records the error
 * in the thread-local slot and evaluates `on_err` (typically a return of
 * PD_ERROR or NULL). Usage:
 *
 *   pd_status pd_foo(...) {
 *     PD_GUARD_BEGIN;
 *     ...
 *     return PD_OK;
 *     PD_GUARD_END(PD_ERROR);
 *   }
 *
 * `on_err` MUST be an expression, e.g. `PD_ERROR` or `NULL`.
 */
#define PD_GUARD_BEGIN                                   \
  try {

#define PD_GUARD_END(on_err)                             \
    return (on_err);                                     \
  } catch (const std::exception& e) {                    \
    pd_set_error(PD_ERROR, e.what());                    \
    return (on_err);                                     \
  } catch (...) {                                        \
    pd_set_error(PD_ERROR, "unknown C++ exception");     \
    return (on_err);                                     \
  }

#endif  /* PADDLE_LUA_CAPI_IMPL_H */