"""
AudioMark v18.6 Patcher
Run this once to fix the Slice Audio error.
Usage: python APPLY_PATCH.py
"""
import zipfile, os, sys, shutil
from pathlib import Path

print("AudioMark v18.6 Patcher")
print("="*40)

# Find audiomark_annotator folder
script_dir = Path(__file__).parent
candidates = [
    script_dir / "audiomark_annotator",
    script_dir,
    script_dir.parent / "audiomark_annotator",
]

am_dir = None
for c in candidates:
    if (c / "static" / "index.html").exists() and (c / "server.py").exists():
        am_dir = c
        break

if not am_dir:
    print("ERROR: Cannot find audiomark_annotator folder")
    print("Put this file next to audiomark_annotator folder and run again")
    input("Press Enter to exit...")
    sys.exit(1)

print(f"Found: {am_dir}")

# Find patch zip
zip_path = script_dir / "audiomark_v18_6_patch.zip"
if not zip_path.exists():
    print(f"ERROR: audiomark_v18_6_patch.zip not found at {zip_path}")
    input("Press Enter to exit...")
    sys.exit(1)

# Backup and replace
print("Backing up...")
shutil.copy(str(am_dir / "static" / "index.html"), str(am_dir / "static" / "index.html.bak"))
shutil.copy(str(am_dir / "server.py"), str(am_dir / "server.py.bak"))

print("Applying patch...")
with zipfile.ZipFile(str(zip_path)) as zf:
    for entry in zf.infolist():
        if entry.filename.endswith('/'):
            continue
        # Strip audiomark_annotator/ prefix
        rel = entry.filename.replace('audiomark_annotator/', '', 1)
        dest = am_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(entry.filename))
        print(f"  Patched: {rel}")

# Verify
html = (am_dir / "static" / "index.html").read_text(encoding='utf-8')
if 'v18.6' in html and html.count('event.target') == 0:
    print()
    print("SUCCESS! Patch applied correctly.")
    print()
    print("Next steps:")
    print("1. Start server: double-click start_windows.bat")
    print("2. Open browser: localhost:8100")
    print("3. Press: Ctrl+Shift+R  (hard refresh)")
    print("4. Title should show: AudioMark v18.6")
else:
    print("WARNING: Patch may not have applied correctly")
    print(f"  v18.6 in title: {'v18.6' in html}")
    print(f"  event.target count: {html.count('event.target')}")

input("\nPress Enter to exit...")
