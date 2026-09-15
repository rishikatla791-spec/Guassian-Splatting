#!/usr/bin/env python3
"""Install the debug APK on a connected phone and launch it.

Deliberately install-and-launch only. It does not push files onto the device,
create directories on its storage, or uninstall anything: a borrowed test phone
must come back unchanged. If the install fails because an existing copy was
signed with a different key, that is reported -- clearing it would delete that
app's data, which is the user's call, not this script's.
"""
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APK_PATH = os.path.join(SCRIPT_DIR, "android", "app", "build", "outputs",
                        "apk", "debug", "app-debug.apk")
PACKAGE = "com.splat.mobile3dgs"


def find_adb():
    """Locate adb: PATH first, then the usual SDK locations."""
    found = shutil.which("adb")
    if found:
        return found
    roots = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"),
             os.path.expanduser("~/AppData/Local/Android/Sdk"),
             os.path.expanduser("~/Android/Sdk"),
             os.path.expanduser("~/Library/Android/sdk")]
    exe = "adb.exe" if os.name == "nt" else "adb"
    for root in roots:
        if not root:
            continue
        candidate = os.path.join(root, "platform-tools", exe)
        if os.path.exists(candidate):
            return candidate
    return None


ADB = find_adb()


def run(args, timeout=60):
    try:
        res = subprocess.run([ADB] + args, capture_output=True, text=True,
                             timeout=timeout, stdin=subprocess.DEVNULL)
        return res.returncode == 0, res.stdout.strip(), res.stderr.strip()
    except Exception as exc:  # noqa: BLE001 - report, never crash the deploy
        return False, "", str(exc)


def find_devices():
    ok, out, _ = run(["devices"])
    if not ok:
        return []
    devices = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serial = parts[0]
            _, brand, _ = run(["-s", serial, "shell", "getprop", "ro.product.brand"])
            _, model, _ = run(["-s", serial, "shell", "getprop", "ro.product.model"])
            name = (brand.capitalize() + " " + model).strip() or serial
            devices.append({"serial": serial, "name": name,
                            "is_emulator": serial.startswith("emulator-")})
    return devices


def main():
    print("=" * 62)
    print("  Mobile 3DGS - install on a connected phone")
    print("=" * 62)

    if not ADB:
        print("\n[!] adb not found. Install Android platform-tools, or set "
              "ANDROID_HOME to your SDK directory.")
        return 1
    print(f"\n[*] adb: {ADB}")

    if not os.path.exists(APK_PATH):
        print(f"\n[!] No APK at {APK_PATH}")
        print("    Build it first:  cd android && gradlew assembleDebug")
        return 1

    devices = find_devices()
    if not devices:
        print("\n[!] No device detected over USB.")
        print("    1. Connect the phone by USB cable.")
        print("    2. Enable Developer Options and USB debugging.")
        print("    3. Accept the 'Allow USB debugging' prompt on the phone.")
        return 1

    physical = [d for d in devices if not d["is_emulator"]]
    target = physical[0] if physical else devices[0]
    serial = target["serial"]
    size_mb = os.path.getsize(APK_PATH) / (1024 * 1024)
    print(f"[*] Target: {target['name']} ({serial})")

    print(f"\n[1/2] Installing app-debug.apk ({size_mb:.1f} MB)...")
    ok, out, err = run(["-s", serial, "install", "-r", "-d", APK_PATH], timeout=300)
    combined = (out + "\n" + err).strip()
    if "Success" in combined:
        print("      Installed.")
    elif "INSTALL_FAILED_UPDATE_INCOMPATIBLE" in combined or "signatures do not match" in combined:
        print("      [!] A copy of this app is already installed with a different "
              "signing key.")
        print("          Clearing it would DELETE that app's data on this phone, so "
              "this script will not do it.")
        print(f"          To proceed manually:  adb -s {serial} uninstall {PACKAGE}")
        return 1
    else:
        print(f"      [!] Install failed: {combined or 'unknown error'}")
        return 1

    print("\n[2/2] Launching...")
    run(["-s", serial, "shell", "am", "start", "-S", "-n", f"{PACKAGE}/.MainActivity"])
    print("\nRunning. To watch the logs:")
    print(f'  adb -s {serial} logcat -s CaptureActivity BrushBridge TrainingService '
          'DepthPointExtractor FrameQualityFilter DatasetExporter')
    return 0


if __name__ == "__main__":
    sys.exit(main())
