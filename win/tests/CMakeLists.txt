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
set_tests_properties(serve_service_test PROPERTIES LABELS unit TIMEOUT 120
  ENVIRONMENT "BRILLIANCE_SERVE_TEST_DEVICES=0:33:1,1:6:0,2:3:0,3:9:0,4:32:0")

# win\serve.ps1 as a launcher: radiance runs on the profile itself (a reload rereads it); -Listen,
# -Port and -ApiKey as radiance's own reader applies them over any profile layout; the cards by PCI
# bus, the integrated GPU hidden by default; -Background -Log with a bare name. The HIP runtime is
# never asked (BRILLIANCE_SERVE_TEST_DEVICES is the device table); radiance runs as --check, and once
# with HIP_VISIBLE_DEVICES=-1 on a model that is not there (loopback port 18375).
add_test(NAME serve_launcher_test
         COMMAND powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass
                 -File "${CMAKE_CURRENT_SOURCE_DIR}/serve_launcher_test.ps1" -Exe "$<TARGET_FILE:radiance>")
set_tests_properties(serve_launcher_test PROPERTIES LABELS unit TIMEOUT 300
  ENVIRONMENT "BRILLIANCE_SERVE_TEST_DEVICES=0:33:1,1:6:0,2:3:0,3:9:0,4:32:0;HIP_VISIBLE_DEVICES=-1")

# Replay recorded engine rank rows without running radiance or the HIP runtime.
add_test(NAME rank_report_test
         COMMAND powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass
                 -File "${CMAKE_CURRENT_SOURCE_DIR}/rank_report_test.ps1")
set_tests_properties(rank_report_test PROPERTIES LABELS unit TIMEOUT 30)
# The published DeepSeek TP2 dry-plan command should explain the YAML form.
add_test(NAME plan_only_cli_test
         COMMAND ${CMAKE_COMMAND}
                 "-DEXE=$<TARGET_FILE:radiance>"
                 "-DPROFILE=${CMAKE_SOURCE_DIR}/win/profiles/deepseek41-2card.yaml"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/plan_only_cli_test.cmake")
set_tests_properties(plan_only_cli_test PROPERTIES LABELS unit TIMEOUT 30
  ENVIRONMENT "BRILLIANCE_DEVICE=host;HIP_VISIBLE_DEVICES=-1;RADIANCE_HOST_DEVICES=1")

# serve.ps1 on a profile of several models (profiles\sandbox.yaml): the entries needing more cards
# than radiance will see are dropped and named, and radiance reads a copy without them. The HIP
# runtime is never asked; radiance runs as --check only.
add_test(NAME serve_sandbox_test
         COMMAND powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass
                 -File "${CMAKE_CURRENT_SOURCE_DIR}/serve_sandbox_test.ps1" -Exe "$<TARGET_FILE:radiance>")
set_tests_properties(serve_sandbox_test PROPERTIES LABELS unit TIMEOUT 300
  ENVIRONMENT "BRILLIANCE_SERVE_TEST_DEVICES=0:33:1,1:6:0,2:3:0,3:9:0,4:32:0;HIP_VISIBLE_DEVICES=-1")

# win\libexec\hfget.ps1 and serve.ps1 -Fetch against a stand-in hub on loopback (hub_stub.py,
# Python's standard library; skipped, 77, without python.exe): the plan before a transfer, a fresh
# destination, a warm cache, a cut transfer resumed by the next run and, try by try, within a run, a
# repo of several containers, and the verified file served with no edit to the profile. Fixture
# bytes only; radiance runs as --check. About 60 s, most of it hfget's 5 s between tries on the cut
# transfers.
add_test(NAME fetch_launch_test
         COMMAND powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass
                 -File "${CMAKE_CURRENT_SOURCE_DIR}/fetch_launch_test.ps1" -Exe "$<TARGET_FILE:radiance>")
set_tests_properties(fetch_launch_test PROPERTIES LABELS unit TIMEOUT 300 SKIP_RETURN_CODE 77)

# win\tools\build-identity.ps1 on fixture headers: the "build identity:" line win\build.ps1 prints
# after a build, by which the release's G1.build gate binds a build log to its candidate. No build.
add_test(NAME build_identity_test
         COMMAND powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass
                 -File "${CMAKE_CURRENT_SOURCE_DIR}/build_identity_test.ps1")
set_tests_properties(build_identity_test PROPERTIES LABELS unit TIMEOUT 60
  ENVIRONMENT "RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp")

# profiles\sandbox.yaml is generated from the other profiles (win\tools\sandbox_profile.py): it
# fails while a profile changed and the sandbox was not regenerated.
find_program(RAD_PYTHON3 NAMES python3)
if(RAD_PYTHON3)
  add_test(NAME sandbox_profile_test
           COMMAND ${RAD_PYTHON3} -I ${CMAKE_SOURCE_DIR}/win/tools/sandbox_profile.py check)
  set_tests_properties(sandbox_profile_test PROPERTIES LABELS unit TIMEOUT 60)
  add_test(NAME docs_check_known_issues_test
           COMMAND ${RAD_PYTHON3} -I -B ${CMAKE_CURRENT_SOURCE_DIR}/docs_check_known_issues_test.py ${CMAKE_SOURCE_DIR})
  set_tests_properties(docs_check_known_issues_test PROPERTIES LABELS unit TIMEOUT 60)
  # REFERENCE.md lists every setting the code reads (win/tools/docs_check.py's settings rule): a
  # fixture of a missing, a stale, a misplaced and a moved setting, then the repository itself.
  add_test(NAME docs_check_reference_test
           COMMAND ${RAD_PYTHON3} -I ${CMAKE_CURRENT_SOURCE_DIR}/docs_check_reference_test.py ${CMAKE_SOURCE_DIR})
  set_tests_properties(docs_check_reference_test PROPERTIES LABELS unit TIMEOUT 120)
  # The stop journal's checker (win\tools\stop_log_check.py, core/util/stoplog.h) on synthetic
  # journals: bounded, over a bound, a split rank exit, a stream not drained, cut off mid-teardown;
  # and the retire (a timed-out drain) as expected, on saved journals in win/tools/fixtures/stoplog.
  add_test(NAME stop_log_check_test
           COMMAND ${RAD_PYTHON3} -I -B ${CMAKE_SOURCE_DIR}/win/tools/stop_log_check_test.py)
  set_tests_properties(stop_log_check_test PROPERTIES LABELS unit TIMEOUT 60
                       ENVIRONMENT "RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp")

  # win\tools\gate_runner.py, the release gate runner, on fixture candidates (a small ZIP with a
  # MANIFEST, logs, check plug-ins, criteria files): the pass, fail, skip and carryover verdicts, the
  # record tied to the ZIP's SHA-256 and the snapshot commit, the build lease. No GPU, no server.
  add_test(NAME gate_runner_test
           COMMAND ${RAD_PYTHON3} -I ${CMAKE_SOURCE_DIR}/win/tests/gate_runner_test.py
                   --tmp ${CMAKE_BINARY_DIR}/tmp)
  set_tests_properties(gate_runner_test PROPERTIES LABELS unit TIMEOUT 120
    ENVIRONMENT "RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp")
  add_test(NAME licence_checks_test
           COMMAND ${RAD_PYTHON3} -I -B ${CMAKE_CURRENT_SOURCE_DIR}/licence_checks_test.py)
  set_tests_properties(licence_checks_test PROPERTIES LABELS unit TIMEOUT 60
    ENVIRONMENT "RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp")
endif()

# bin\rad-export.cmd in an unpacked package runs the rad-convert beside it with bin\ off PATH, and
# returns rad-export's exit status. A stub rad-convert; no conversion. Skipped (77) with no Python.
add_test(NAME rad_export_cmd_test
         COMMAND powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass
                 -File "${CMAKE_CURRENT_SOURCE_DIR}/rad_export_cmd_test.ps1")
set_tests_properties(rad_export_cmd_test PROPERTIES LABELS unit TIMEOUT 120 SKIP_RETURN_CODE 77)

# win\tools\gate-lib.ps1, the gate windows' runner, with fixture gates under Windows PowerShell 5.1:
# a failed gate stops the window and is named, exit codes read right (a hidden process's from its
# object, a native command's from $LASTEXITCODE even when it writes to stderr). No GPU, no server.
add_test(NAME gate_lib_test
         COMMAND powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass
                 -File "${CMAKE_CURRENT_SOURCE_DIR}/gate_lib_test.ps1"
                 -Lib "${CMAKE_SOURCE_DIR}/win/tools/gate-lib.ps1")
set_tests_properties(gate_lib_test PROPERTIES LABELS unit TIMEOUT 120)

# The process's pinned-memory limit (win/device/pin_ledger.h): BRILLIANCE_PIN_LIMIT_GIB lowers it
# and is clamped to the shared-memory budget's figure, never raises it. Pure arithmetic on a budget
# the test hands in; no card, no HIP.
add_executable(pin_limit_test pin_limit_test.cpp)
target_include_directories(pin_limit_test PRIVATE "${CMAKE_SOURCE_DIR}/tests" "${CMAKE_SOURCE_DIR}/win/device")
target_compile_features(pin_limit_test PRIVATE cxx_std_20)
set_target_properties(pin_limit_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME pin_limit_test COMMAND pin_limit_test)
set_tests_properties(pin_limit_test PROPERTIES
  LABELS unit
  SKIP_RETURN_CODE 77
  TIMEOUT 60
  ENVIRONMENT "HIP_VISIBLE_DEVICES=-1;RAD_TEST_STRICT=1;RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp")

# The native HIP backend's stall watch (win/device/stall_watch.h): a long step whose progress
# beacons move is never a stall, a real one is still reported once, 5 s after they stop. Driven with
# the test's own clock; no card, no HIP.
add_executable(stall_watch_test stall_watch_test.cpp)
target_include_directories(stall_watch_test PRIVATE "${CMAKE_SOURCE_DIR}/tests" "${CMAKE_SOURCE_DIR}/win/device")
target_compile_features(stall_watch_test PRIVATE cxx_std_20)
set_target_properties(stall_watch_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME stall_watch_test COMMAND stall_watch_test)
set_tests_properties(stall_watch_test PROPERTIES
  LABELS unit
  SKIP_RETURN_CODE 77
  TIMEOUT 60
  ENVIRONMENT "HIP_VISIBLE_DEVICES=-1;RAD_TEST_STRICT=1;RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp")

# The same watch on a card, through the native HIP backend: a batch that keeps the card busy about
# 8 s behind one wait is not a stall, a stream held still by a host function (the card idle) is.
# About 20 s on one card; label gpu, one at a time, as the device tests on a card are.
if(RAD_HIP_FOUND)
  add_executable(stall_watch_card_test stall_watch_card_test.cpp)
  target_include_directories(stall_watch_card_test PRIVATE "${CMAKE_SOURCE_DIR}/tests" "${CMAKE_SOURCE_DIR}/core")
  target_link_libraries(stall_watch_card_test PRIVATE rad_core)
  target_compile_features(stall_watch_card_test PRIVATE cxx_std_20)
  set_target_properties(stall_watch_card_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  add_test(NAME stall_watch_card_test COMMAND stall_watch_card_test)
  set_tests_properties(stall_watch_card_test PROPERTIES
    LABELS gpu
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 120
    ENVIRONMENT "RAD_TEST_STRICT=1;RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp")
endif()

# How r4d_selftest finds libref beside the plugin under test when the plugin's path is written with
# '\' (win/compat/include/brl_path.h): the build tree's libref.dll, loaded through the POSIX layer's
# dlopen. No card, no HIP.
add_executable(selftest_ref_path_test selftest_ref_path_test.cpp)
target_include_directories(selftest_ref_path_test PRIVATE "${CMAKE_SOURCE_DIR}/tests")
target_link_libraries(selftest_ref_path_test PRIVATE radiance::abi)
target_compile_features(selftest_ref_path_test PRIVATE cxx_std_20)
target_compile_definitions(selftest_ref_path_test PRIVATE BRL_TEST_KERNELS="${RAD_PLUGIN_OUT}/kernels")
add_dependencies(selftest_ref_path_test libref)
set_target_properties(selftest_ref_path_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME selftest_ref_path_test COMMAND selftest_ref_path_test)
set_tests_properties(selftest_ref_path_test PROPERTIES
  LABELS unit
  SKIP_RETURN_CODE 77
  TIMEOUT 60
  ENVIRONMENT "HIP_VISIBLE_DEVICES=-1;RAD_TEST_STRICT=1;RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp")
