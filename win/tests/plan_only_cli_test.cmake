function(run_radiance expected_code expected_message)
  execute_process(COMMAND "${EXE}" ${ARGN}
                  RESULT_VARIABLE code ERROR_VARIABLE log)
  string(FIND "${log}" "${expected_message}" match)
  if(NOT code STREQUAL "${expected_code}" OR match EQUAL -1)
    message(FATAL_ERROR "radiance ${ARGN} exited ${code}; expected ${expected_code} and '${expected_message}'\n${log}")
  endif()
  message(STATUS "radiance ${ARGN} exited ${code}\n${log}")
endfunction()

set(hint "add 'plan-only: true' to that model's entry")
run_radiance(2 "${hint}" --config "${PROFILE}" --plan-only)
run_radiance(2 "${hint}" --plan-only --config "${PROFILE}")

# Exercise the suggested YAML on the host, without the model container.
set(copy "${CMAKE_CURRENT_BINARY_DIR}/deepseek41-plan-test.yaml")
file(READ "${PROFILE}" yaml)
file(WRITE "${copy}" "${yaml}\n    plan-only: true\n")
run_radiance(0 "model deepseek-v4.1-flash =" --config "${copy}" --check)
run_radiance(2 "tp 2 needs 2 cards; 1 are available" --config "${copy}")
