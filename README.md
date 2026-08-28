# amdhip64-hipLaunchKernel-crash-fix

Binary patch for `amdhip64_7.dll` that fixes a fatal `0xC0000005` access violation
crashing every HIP kernel dispatch on Windows with RDNA 4 GPUs (RX 9070 / 9070 XT).

## The Bug

On **Windows + ROCm 7.14 + gfx1201 (RDNA 4)**, any attempt to launch a HIP kernel
crashes the process immediately with no Python exception — the process just dies.

**Root cause** (confirmed via WinDbg live kernel debugging):

`hipLaunchKernel` calls into the ROCprofiler callback registry on every kernel
dispatch. The callback linked list contains a garbage pointer on first dispatch,
causing a null/invalid pointer dereference inside `hipProfilerRegisterChunkCallbackExt`.

```
hipLaunchKernel+0x85: call r10          ← dispatches profiler callbacks
  → hipRegisterTracerCallback (x5)
  → hipProfilerRegisterChunkCallbackExt
    → mov rax, [rcx+8]                  ← CRASH: rcx = 0x7b2c450fc892ac84 (garbage)
```

**Exception:** `0xC0000005 ACCESS_VIOLATION` in `amdhip64_7.dll`

## The Fix

NOP out the profiler callback dispatch call in `hipLaunchKernel`:

```
File offset 0x4549B5:
  Before: 41 FF D2   (call r10)
  After:  90 90 90   (NOP NOP NOP)
```

This disables the profiler callback dispatch entirely. All PyTorch training,
inference, and compute operations work correctly after this patch.

## Usage

```bash
python patch_amdhip64.py <path_to_amdhip64_7.dll>
```

```bash
# Example — default ROCm pip install location
python patch_amdhip64.py "rocm_env\Lib\site-packages\_rocm_sdk_core\bin\amdhip64_7.dll"
```

The script:
- Creates a `.bak` backup before patching
- Tries the known offset first (fast path)
- Falls back to dynamic export-table search for other ROCm versions
- Is idempotent — safe to run multiple times

## Environment

| Component | Version |
|-----------|---------|
| GPU | AMD Radeon RX 9070 XT (gfx1201, RDNA 4) |
| OS | Windows 11 |
| ROCm | 7.14.0 |
| PyTorch | 2.12.0+rocm7.14.0 |
| `torch.version.hip` | 7.14.60850 |

## Upstream Issues

- ROCm/TheRock: [https://github.com/ROCm/TheRock/issues/7732]
- ROCm/rocm-systems: [https://github.com/ROCm/rocm-systems/issues/10924]

## How It Was Found

The crash was analyzed using **WinDbg** live kernel debugging with child process
tracking. The access violation was caught at its origin (not at process exit),
revealing the full call chain and the garbage pointer value in `rcx`.

The patch offset was located by parsing the PE export table to find `hipLaunchKernel`,
then scanning the function body for the `call r10` instruction at `+0x85`.

## Notes

- Re-run the patcher after ROCm updates (the DLL will be overwritten)
- This patch disables ROCprofiler tracing — profiling tools like `rocprof` will
  not intercept HIP API calls on the patched binary
- For production/profiling use, wait for the upstream fix in a future ROCm release

## Related

Additional workaround needed for FlashAttention backward pass crash (separate bug):

```python
model = AutoModelForCausalLM.from_pretrained(
    ...,
    attn_implementation="eager"  # disables aotriton FlashAttention
)
```
