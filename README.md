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
hipLaunchKernel+0x85: call r10
  → hipRegisterTracerCallback (x5)
  → hipProfilerRegisterChunkCallbackExt
    → mov rax, [rcx+8]    ← CRASH: rcx = 0x7b2c450fc892ac84 (garbage pointer)
```

**Exception:** `0xC0000005 ACCESS_VIOLATION` in `amdhip64_7.dll`

## The Fix

Patch `hipProfilerRegisterChunkCallbackExt` to immediately return 0:

```
File offset 0x4BBD60 (hipProfilerRegisterChunkCallbackExt):
  Before: 48 89 5C ...   (function prologue)
  After:  33 C0 C3       (xor eax,eax; ret  →  always return 0 / not found)
```

This disables the profiler callback walk entirely. The function returns 0 (callback
not found) before dereferencing the corrupt pointer. All PyTorch training,
inference, and compute operations work correctly after this patch.

> **⚠️ Earlier versions of this patch NOPed `hipLaunchKernel+0x85` (`41 FF D2 → 90 90 90`).
> That was incorrect — it disabled the actual kernel dispatch call, causing all GPU
> tensors to be zero. The correct fix targets `hipProfilerRegisterChunkCallbackExt` directly.
> Re-run the latest patcher script to apply the correct patch.**

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
- Reverts the old wrong patch automatically if present
- Tries the known offset first (fast path)
- Falls back to dynamic export-table search for other ROCm versions
- Is idempotent — safe to run multiple times

Verify the patch worked:
```python
import torch
t = torch.tensor([0.265625, -0.1, 0.5], dtype=torch.bfloat16)
print(t.cuda())
# Expected: tensor([ 0.2656, -0.1001,  0.5000], device='cuda:0')
```

## Environment

| Component | Version |
|-----------|---------|
| GPU | AMD Radeon RX 9070 XT (gfx1201, RDNA 4) |
| OS | Windows 11 |
| ROCm | 7.14.0 |
| PyTorch | 2.12.0+rocm7.14.0 |
| `torch.version.hip` | 7.14.60850 |

## Upstream Issues

- ROCm/TheRock: https://github.com/ROCm/TheRock/issues/7732
- ROCm/rocm-systems: https://github.com/ROCm/rocm-systems/issues/10924

## How It Was Found

The crash was analyzed using **WinDbg** live kernel debugging with child process
tracking (`-o` flag). The access violation was caught at its origin (not at process
exit), revealing the full call chain and the garbage pointer value in `rcx`.

The crash location was confirmed by:
1. Setting `sxe av` (break on first-chance access violation)
2. Running `~* kn` to get all thread stacks
3. Identifying thread 9 with the `hipProfilerRegisterChunkCallbackExt` crash
4. Inspecting `rcx = 0x7b2c450fc892ac84` (garbage, not null)
5. Locating `hipProfilerRegisterChunkCallbackExt` via PE export table

The correct patch offset was found by exporting the function RVA from the PE
export directory and converting to file offset.

## Notes

- Re-run the patcher after ROCm updates (the DLL will be overwritten)
- This patch disables ROCprofiler tracing — profiling tools like `rocprof` will
  not intercept HIP API calls on the patched binary
- For production/profiling use, wait for the upstream fix in a future ROCm release

## Related

Additional workaround needed for FlashAttention backward pass crash (separate bug
in `aotriton_v2.dll` on RDNA 4 Windows):

```python
model = AutoModelForCausalLM.from_pretrained(
    ...,
    attn_implementation="eager"  # disables aotriton FlashAttention
)
```
