# Brilliance: the Windows contract tests, beside the tree's own tests/. Added by
# brl_windows_finalize (win/cmake/RadWindows.cmake) when tests are built, once radiance exists.
# They use the tree's harness (tests/rad_test.h) and run under ctest with the unit label.

# radiance as a Windows service, without the Service Control Manager (win/SERVICE.md): the command
# line, the control handler and stop pump, and radiance.exe --service run from a console, on the
# host backend with no card and no model.
add_executable(service_win_test service_win_test.cpp)
target_include_directories(service_win_test PRIVATE "${CMAKE_SOURCE_DIR}/tests")
target_link_libraries(service_win_test PRIVATE ws2_32)
target_compile_features(service_win_test PRIVATE cxx_std_20)
target_compile_definitions(service_win_test PRIVATE RAD_SERVICE_TEST_RADIANCE="$<TARGET_FILE:radiance>")
add_dependencies(service_win_test radiance)
# Out of bin/, which is the package's: brl_windows_finalize copies the runtime DLLs here.
set_target_properties(service_win_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tmp")
add_test(NAME service_win_test COMMAND service_win_test)
# HIP_VISIBLE_DEVICES=-1 keeps the radiance it starts off every card. The servers listen on loopback
# ports 18361-18363.
set_tests_properties(service_win_test PROPERTIES
  LABELS unit
  SKIP_RETURN_CODE 77
  TIMEOUT 300
  ENVIRONMENT "HIP_VISIBLE_DEVICES=-1;ROCR_VISIBLE_DEVICES=99;RAD_TEST_STRICT=1;RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp;OMP_WAIT_POLICY=passive")

# win\serve.ps1 -Install / -Uninstall without elevation: what -WhatIf prints, exactly, and what the
# service options refuse. Nothing is installed; the SCM is asked only about a service that is not
# there.
add_test(NAME serve_service_test
         COMMAND powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass
                 -File "${CMAKE_CURRENT_SOURCE_DIR}/serve_service_test.ps1" -Exe "$<TARGET_FILE:radiance>")
set_tests_properties(serve_service_test PROPERTIES LABELS unit TIMEOUT 120)
