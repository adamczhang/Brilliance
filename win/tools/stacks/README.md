# brl-stacks: the stacks of a process that stopped making progress

When a server stops answering, `/health` reports `"status":"stuck"`. The engine's step watchdog
(`W step: nothing has been logged ...`) prints only the step thread's stack. `brl-stacks` prints
every thread's stack, from outside the process:

    bin\brl-stacks.exe <pid> > stacks.txt
    python -I libexec\brilliance\resolve-stacks.py stacks.txt bin\radiance.map > stacks-resolved.txt

- **What it does to the process.** Each thread is suspended only while its stack is read, then
  resumed. Nothing is changed and nothing is ended.
- **Why the map.** A frame prints as `module+offset`, and as the nearest symbol DbgHelp knows
  (exports, or a PDB). `radiance.exe` has no PDB, so `resolve-stacks.py` turns its offsets into
  function names from `radiance.map`.
  - The build writes `radiance.map` beside `radiance.exe` (`brl_win_host_exports` in
    `win/cmake/RadWindows.cmake`).
  - Only a map from the same link matches a running binary.
- **HIP frames.** `amdhip64_7.dll` frames show the nearest exported name, which for the runtime's
  internals is only a landmark (`hipEventSynchronize+0x29` is real; `hipProfiler...+0x3677a` is not).
- **Example.** On Flash-Next at TP2, in a package that hangs instead of re-sending a lost write
  (`SUPPORT.md`, known issue 1), it showed:
  - both rank threads in `hipEventSynchronize` under `Ctx::host_join_wait`;
  - the step thread at the drafter's rank barrier;
  - every GPU engine of the process idle.
- **Python.** `resolve-stacks.py` needs Python 3 (no packages); `brl-stacks.exe` does not.
- **Building it.** It is plain Win32 with no HIP, and is built in every Windows build. To build it
  alone:

      .toolchain\rocm-venv\Lib\site-packages\_rocm_sdk_devel\lib\llvm\bin\clang++.exe -O1 -std=c++17 win\tools\stacks\brl-stacks.cpp -o brl-stacks.exe

  (after `. win\env.ps1`).
