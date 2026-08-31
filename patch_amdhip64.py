"""
amdhip64_7.dll patcher
Fix: hipProfilerRegisterChunkCallbackExt → xor eax,eax; ret
     Prevents 0xC0000005 crash on every HIP kernel dispatch (Windows, gfx1201)

Root cause (confirmed via WinDbg):
  hipLaunchKernel calls into the profiler callback registry on every dispatch.
  On Windows/gfx1201 the callback linked list contains a garbage pointer,
  causing hipProfilerRegisterChunkCallbackExt to crash at [rcx+8].
  Patching the function to immediately return 0 (not found) is safe —
  it disables profiler intercept of HIP calls but leaves all compute intact.

Usage:
  python patch_amdhip64.py <path_to_amdhip64_7.dll>

Related issues:
  https://github.com/ROCm/TheRock/issues/7732
  https://github.com/ROCm/rocm-systems/issues/10924
"""

import struct, sys, shutil
from pathlib import Path

DLL_DEFAULT = r"rocm_env\Lib\site-packages\_rocm_sdk_core\bin\amdhip64_7.dll"

# Known file offset for hipProfilerRegisterChunkCallbackExt (ROCm 7.14 / gfx1201)
# Patch: first 3 bytes → xor eax,eax (33 C0) + ret (C3) → always return 0
KNOWN_OFFSET = 0x4BBD60
PATCH_BYTES  = bytes([0x33, 0xC0, 0xC3])   # xor eax,eax; ret

# Legacy wrong patch to revert if present (NOP'd the kernel dispatch call)
WRONG_OFF    = 0x4549B5
WRONG_BYTES  = bytes([0x90, 0x90, 0x90])   # NOP NOP NOP (was wrong)
WRONG_ORIG   = bytes([0x41, 0xFF, 0xD2])   # call r10   (correct)


def rva2file(data, rva):
    e = struct.unpack_from('<I', data, 0x3C)[0]
    n = struct.unpack_from('<H', data, e + 6)[0]
    opt_sz = struct.unpack_from('<H', data, e + 20)[0]
    sec = e + 24 + opt_sz
    for i in range(n):
        o  = sec + i * 40
        va = struct.unpack_from('<I', data, o + 12)[0]
        vs = struct.unpack_from('<I', data, o +  8)[0]
        ro = struct.unpack_from('<I', data, o + 20)[0]
        if va <= rva < va + vs:
            return ro + (rva - va)
    return None


def find_export(data, name):
    """Locate a named export's file offset via the PE export table."""
    try:
        e       = struct.unpack_from('<I', data, 0x3C)[0]
        exp_rva = struct.unpack_from('<I', data, e + 24 + 112)[0]
        exp_off = rva2file(data, exp_rva)
        n_names = struct.unpack_from('<I', data, exp_off + 24)[0]
        fn_off  = rva2file(data, struct.unpack_from('<I', data, exp_off + 28)[0])
        nm_off  = rva2file(data, struct.unpack_from('<I', data, exp_off + 32)[0])
        or_off  = rva2file(data, struct.unpack_from('<I', data, exp_off + 36)[0])
        for i in range(n_names):
            nrva = struct.unpack_from('<I', data, nm_off + i * 4)[0]
            noff = rva2file(data, nrva)
            end  = data.index(b'\x00', noff)
            n    = data[noff:end].decode('ascii', 'replace')
            if n == name:
                oidx = struct.unpack_from('<H', data, or_off + i * 2)[0]
                frva = struct.unpack_from('<I', data, fn_off + oidx * 4)[0]
                return rva2file(data, frva)
    except Exception as exc:
        print(f"  [!] Export parse error: {exc}")
    return None


def patch(dll_path: str) -> bool:
    path = Path(dll_path)
    if not path.exists():
        print(f"[!] File not found: {path}")
        return False

    bak = path.with_suffix('.dll.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[+] Backup: {bak}")
    else:
        print(f"[*] Backup already exists: {bak}")

    data    = bytearray(path.read_bytes())
    changed = False

    # ── Step 1: revert legacy wrong patch if present ─────────────────────
    if data[WRONG_OFF:WRONG_OFF + 3] == bytearray(WRONG_BYTES):
        data[WRONG_OFF:WRONG_OFF + 3] = WRONG_ORIG
        print(f"[+] Reverted legacy wrong patch at 0x{WRONG_OFF:X} (NOP → call r10)")
        changed = True
    elif data[WRONG_OFF:WRONG_OFF + 3] == bytearray(WRONG_ORIG):
        print(f"[*] hipLaunchKernel already clean (0x{WRONG_OFF:X})")

    # ── Step 2: patch hipProfilerRegisterChunkCallbackExt ────────────────
    # Try dynamic export-table lookup first; fall back to known offset.
    off = find_export(bytes(data), 'hipProfilerRegisterChunkCallbackExt')
    if off:
        print(f"[+] hipProfilerRegisterChunkCallbackExt at file offset 0x{off:X}")
    else:
        off = KNOWN_OFFSET
        print(f"[*] Dynamic lookup failed — using known offset 0x{off:X}")

    actual = bytes(data[off:off + 3])
    if actual == PATCH_BYTES:
        print(f"[*] Correct patch already applied (0x{off:X})")
    else:
        print(f"[+] Patching 0x{off:X}: {actual.hex()} → {PATCH_BYTES.hex()}"
              f"  (xor eax,eax; ret)")
        data[off:off + 3] = PATCH_BYTES
        changed = True

    if changed:
        path.write_bytes(data)
        print(f"[+] Written: {path}")

    return True


def main():
    print("=" * 60)
    print("  amdhip64_7.dll Patcher")
    print("  Fix: hipProfilerRegisterChunkCallbackExt → ret 0")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python patch_amdhip64.py <dll_path>")
        print()
        print("Example:")
        print(f'  python patch_amdhip64.py "{DLL_DEFAULT}"')
        sys.exit(1)

    dll = " ".join(sys.argv[1:])
    print(f"[*] Target: {dll}\n")

    ok = patch(dll)
    print()
    if ok:
        print("Done.")
        print("\nVerify:")
        print('  python -c "import torch; '
              't=torch.tensor([0.5]).cuda(); print(t)"')
        print("  Expected: tensor([0.5000], device='cuda:0')")
    else:
        print("Failed.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
