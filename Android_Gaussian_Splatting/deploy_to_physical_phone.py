#!/usr/bin/env python3
import os, sys, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APK_PATH = os.path.join(SCRIPT_DIR, "android", "app", "build", "outputs", "apk", "debug", "app-debug.apk")
ASSETS_MODELS_DIR = os.path.join(SCRIPT_DIR, "android", "app", "src", "main", "assets", "viewer")
TEMP_MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "temp_test_run")
MODELS = ["truck.splat", "train.splat", "room.splat"]

def run(cmd, timeout=60):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return res.returncode == 0, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def find_devices():
    ok, out, _ = run(["adb", "devices"])
    if not ok: return []
    devices = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            s = parts[0]
            _, b, _ = run(["adb", "-s", s, "shell", "getprop", "ro.product.brand"])
            _, m, _ = run(["adb", "-s", s, "shell", "getprop", "ro.product.model"])
            name = (b.capitalize() + " " + m).strip() if (b or m) else s
            devices.append({"serial": s, "name": name, "is_emulator": s.startswith("emulator-")})
    return devices

def main():
    print("=" * 60)
    print("  MOBILE 3DGS - 1-CLICK PHYSICAL PHONE DEPLOYER AND SYNC")
    print("=" * 60)
    devs = find_devices()
    if not devs:
        print("\n[!] No connected Android devices detected via USB.")
        print("    1. Connect phone via USB cable.")
        print("    2. Enable Developer Options & USB Debugging.")
        print("    3. Tap 'Always Allow' on phone prompt.")
        sys.exit(1)
    phys = [d for d in devs if not d["is_emulator"]]
    target = phys[0] if phys else devs[0]
    serial = target["serial"]
    print(f"\n[*] Target Device: {target['name']} ({serial})")
    if os.path.exists(APK_PATH):
        print(f"\n[1/3] Installing APK on {target['name']}...")
        ok, out, _ = run(["adb", "-s", serial, "install", "-r", "-d", APK_PATH], timeout=180)
        print("      " + ("[OK] Installed successfully!" if "Success" in out else f"Status: {out}"))
    print("\n[2/3] Transferring 3D Gaussian Models (.splat) to phone...")
    app_dir = "/sdcard/Android/data/com.splat.mobile3dgs/files"
    dl_dir = "/sdcard/Download/3DGS"
    run(["adb", "-s", serial, "shell", f"mkdir -p {app_dir} {dl_dir}"])
    for m in MODELS:
        src = os.path.join(ASSETS_MODELS_DIR, m)
        if not os.path.exists(src): src = os.path.join(TEMP_MODELS_DIR, m)
        if os.path.exists(src):
            size_mb = os.path.getsize(src) / (1024*1024)
            print(f"      -> Pushing {m} ({size_mb:.1f} MB)...")
            run(["adb", "-s", serial, "push", src, f"{app_dir}/{m}"])
            run(["adb", "-s", serial, "push", src, f"{dl_dir}/{m}"])
    print("\n[3/3] Launching Mobile3DGS app...")
    run(["adb", "-s", serial, "shell", "am", "start", "-S", "-n", "com.splat.mobile3dgs/.MainActivity"])
    print("\n[SUCCESS] 3D Models ready on phone screen!")

if __name__ == "__main__":
    main()
