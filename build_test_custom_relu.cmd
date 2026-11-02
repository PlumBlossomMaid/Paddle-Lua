@echo off
REM build_test_custom_relu.cmd -- compile the P16 smoke custom op into
REM build-test/custom_relu.dll (user-side C++ that includes paddle/extension.h).
REM
REM The FIRST tick hit "fatal error C1083: cannot open pybind11/pybind11.h"
REM because paddle/extension.h:46 pulls paddle/utils/pybind.h. The fix is
REM adding the pybind11 include dir to the compile line -- it ships inside
REM the no-Python build tree (third_party/pybind/src/extern_pybind/include),
REM exactly like paddle_capi's own compile already does.
REM
REM call this from a vcvars64 prompt:
REM   call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
REM   call build_test_custom_relu.cmd

setlocal
set ROOT=%~dp0
set PADDLE=E:\code\Paddle
set BUILD=%PADDLE%\build-nopy
set OUT=%ROOT%build-test

REM User op source (official example, UNCHANGED) + our CUDA link stub.
REM On CPU-only Windows, DLL links are strict: the official relu .cc
REM references relu_cuda_* that live in custom_relu_op.cu (GPU only).
REM The stub satisfies the linker; CPU dispatch never calls it.
set SRC=E:\code\Paddle\test\custom_op\custom_relu_op.cc
set STUB=%ROOT%tools\smoke\custom_relu_cuda_stub.cc
REM Link against the ALREADY-LOADED host Paddle only, never a full second
REM copy: paddle_core.dll IS the in-process Paddle (statically linked +
REM shared phi.dll). phi.lib supplies paddle::Tensor / OpMetaInfoBuilder /
REM CustomOpKernelContext / empty_like; paddle_core.lib supplies InitPhi
REM (we define+export it in custom_op.cc). Loading a second full Paddle
REM (paddle_inference.dll) re-registers common.dll's flag registry and
REM dies with "flag multiple definition ... io.cc" (flags_native.cc:309).
set LIBS=%BUILD%\paddle\phi\phi.lib %ROOT%build-test\paddle_core.lib
set INC=-IE:\code\Paddle -I%BUILD% -I%BUILD%\third_party\install\gflags\include -I%BUILD%\third_party\install\glog\include -I%E:\code\Paddle\third_party\eigen3 -I%E:\code\Paddle\third_party\threadpool -I%E:\code\Paddle\third_party\dlpack\include -I%BUILD%\third_party\install\xxhash\include -I%BUILD%\third_party\install\warpctc\include -I%BUILD%\third_party\install\warprnnt\include -I%BUILD%\third_party\install\utf8proc\include -I%BUILD%\third_party\install\mklml\include -I%BUILD%\third_party\install\onednn\include -I%BUILD%\third_party\install\protobuf\include -IE:\code\Paddle\third_party\nlohmann_json\include -I%BUILD%\third_party\install\yaml-cpp\include -IC:\Users\15895\AppData\Local\Programs\Python\Python311\include -IC:\Users\15895\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\_core\include -I%BUILD%\third_party\pybind\src\extern_pybind\include -I%BUILD%\third_party\install\libuv\include -I%BUILD%\third_party\install\cryptopp\include -I%BUILD%\third_party\pocketfft\src -I%BUILD%\third_party\dirent\src\extern_dirent\include

set DEFS=-DPADDLE_WITH_DNNL -DPADDLE_NO_PYTHON -DPADDLE_VERSION=3.5.0.dev20260803 -DPADDLE_WITH_AVX -DPADDLE_WITH_MKLML -DPADDLE_WITH_CRYPTO -DPADDLE_WITH_POCKETFFT -DPADDLE_WITH_SSE3 -DPHI_SHARED -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DEIGEN_STRONG_INLINE=inline -DGLOG_NO_ABBREVIATED_SEVERITIES -DGOOGLE_GLOG_DLL_DECL=""

REM Compile the user op and the CUDA stub to separate .obj, then link both
REM into the DLL. /Fo cannot be shared across multiple sources.
cl /nologo /c /O2 /EHsc /std:c++17 /MT /bigobj %DEFS% %INC% %SRC% /Fo:%OUT%\custom_relu.obj
if errorlevel 1 exit /b 1
cl /nologo /c /O2 /EHsc /std:c++17 /MT /bigobj %DEFS% %INC% %STUB% /Fo:%OUT%\custom_relu_cuda_stub.obj
if errorlevel 1 exit /b 1
cl /nologo /LD %OUT%\custom_relu.obj %OUT%\custom_relu_cuda_stub.obj /Fe:%OUT%\custom_relu.dll %LIBS%
endlocal