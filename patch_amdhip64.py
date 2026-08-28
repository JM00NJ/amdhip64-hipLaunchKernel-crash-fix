"""
amdhip64_7.dll patcher — hipLaunchKernel profiler callback crash fix
Usage:
  python patch_amdhip64.py <path_to_amdhip64_7.dll>
  python patch_amdhip64.py "C:\\path\\to\\amdhip64_7.dll"
"""

import struct
import sys
import shutil
from pathlib import Path


# Known patch targets: {(file_offset, original_bytes): description}
KNOWN_PATCHES = {
    (0x4549B5, bytes([0x41, 0xFF, 0xD2])): "hipLaunchKernel+0x85 — ROCm 7.14 / gfx1201",
}


def rva_to_file(data: bytes, rva: int):
    e_lfanew     = struct.unpack_from('<I', data, 0x3C)[0]
    num_sections = struct.unpack_from('<H', data, e_lfanew + 6)[0]
    opt_sz       = struct.unpack_from('<H', data, e_lfanew + 20)[0]
    sec_base     = e_lfanew + 24 + opt_sz
    for i in range(num_sections):
        o     = sec_base + i * 40
        vaddr = struct.unpack_from('<I', data, o + 12)[0]
        vsize = struct.unpack_from('<I', data, o +  8)[0]
        raw   = struct.unpack_from('<I', data, o + 20)[0]
        if vaddr <= rva < vaddr + vsize:
            return raw + (rva - vaddr)
    return None


def find_patch_dynamic(data: bytes):
    """
    Locate hipLaunchKernel via the export table, then search for
    the 'call r10' (41 FF D2) instruction near offset +0x85.
    """
    try:
        e_lfanew = struct.unpack_from('<I', data, 0x3C)[0]
        exp_rva  = struct.unpack_from('<I', data, e_lfanew + 24 + 112)[0]
        exp_off  = rva_to_file(data, exp_rva)

        n_names = struct.unpack_from('<I', data, exp_off + 24)[0]
        fn_rva  = struct.unpack_from('<I', data, exp_off + 28)[0]
        nm_rva  = struct.unpack_from('<I', data, exp_off + 32)[0]
        or_rva  = struct.unpack_from('<I', data, exp_off + 36)[0]
        fn_off  = rva_to_file(data, fn_rva)
        nm_off  = rva_to_file(data, nm_rva)
        or_off  = rva_to_file(data, or_rva)

        launch_file = None
        for i in range(n_names):
            nrva = struct.unpack_from('<I', data, nm_off + i * 4)[0]
            noff = rva_to_file(data, nrva)
            end  = data.index(b'\x00', noff)
            name = data[noff:end].decode('ascii', 'replace')
            if name == 'hipLaunchKernel':
                oidx = struct.unpack_from('<H', data, or_off + i * 2)[0]
                frva = struct.unpack_from('<I', data, fn_off + oidx * 4)[0]
                launch_file = rva_to_file(data, frva)
                break

        if launch_file is None:
            return None

        # Search for 'call r10' (41 FF D2) near hipLaunchKernel+0x85
        for delta in range(-48, 64):
            off = launch_file + 0x85 + delta
            if 0 <= off < len(data) - 2 and data[off:off+3] == b'\x41\xFF\xD2':
                return (off, bytes([0x41, 0xFF, 0xD2]),
                        f"hipLaunchKernel+{0x85+delta:#x} (dynamic)")

    except Exception as exc:
        print(f"  [!] Dynamic search error: {exc}")
    return None


def patch(dll_path: str) -> bool:
    path = Path(dll_path)

    if not path.exists():
        print(f"[!] File not found: {path}")
        return False

    # Create backup if it doesn't exist yet
    bak = path.with_suffix('.dll.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[+] Backup saved: {bak}")
    else:
        print(f"[*] Backup already exists: {bak}")

    data = bytearray(path.read_bytes())
    print(f"[+] DLL size: {len(data):,} bytes")

    # 1. Try known offsets first (fast path)
    for (off, orig), desc in KNOWN_PATCHES.items():
        if off >= len(data):
            continue
        actual = data[off:off+3]
        if actual == bytearray([0x90, 0x90, 0x90]):
            print(f"[*] Already patched ({desc})")
            return True
        if actual == bytearray(orig):
            print(f"[+] Known offset found: 0x{off:X}  ({desc})")
            data[off:off+3] = b'\x90\x90\x90'
            path.write_bytes(data)
            print(f"[+] Patch applied -> {path}")
            return True

    # 2. Fall back to dynamic export-table search
    print("[*] Known offset not found — running dynamic search...")
    result = find_patch_dynamic(bytes(data))
    if result:
        off, orig, desc = result
        if data[off:off+3] == bytearray([0x90, 0x90, 0x90]):
            print(f"[*] Already patched ({desc})")
            return True
        print(f"[+] Dynamic match: 0x{off:X}  ({desc})")
        data[off:off+3] = b'\x90\x90\x90'
        path.write_bytes(data)
        print(f"[+] Patch applied -> {path}")
        return True

    print("[!] Patch site not found. DLL version may differ from known offsets.")
    print("    Re-run WinDbg analysis to locate the correct offset.")
    return False


def main():
    print("=" * 60)
    print("  amdhip64_7.dll Patcher")
    print("  hipLaunchKernel profiler callback crash fix")
    print("  Patch: call r10 (41 FF D2) -> NOP NOP NOP (90 90 90)")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python patch_amdhip64.py <dll_path>")
        print()
        print("Examples:")
        print('  python patch_amdhip64.py '
              r'"rocm_env\Lib\site-packages\_rocm_sdk_core\bin\amdhip64_7.dll"')
        print('  python patch_amdhip64.py '
              r'"C:\Program Files\AMD\ROCm\bin\amdhip64_7.dll"')
        sys.exit(1)

    dll_path = " ".join(sys.argv[1:])  # handle paths with spaces
    print(f"[*] Target: {dll_path}\n")

    ok = patch(dll_path)
    print()
    if ok:
        print("Done.")
    else:
        print("Patch failed.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
