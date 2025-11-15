#!/usr/bin/env python3
"""
Verification script to check if web app integration is working correctly.
Run this before running the main arkit_scenes example.
"""

import sys

print("=" * 70)
print("ARKit Scenes + AI Web App Integration - Setup Verification")
print("=" * 70)
print()

# Check Python version
print(f"✓ Python version: {sys.version.split()[0]}")

# Check required packages
required_packages = {
    "rerun": "rerun-sdk",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "trimesh": "trimesh",
    "scipy": "scipy",
}

missing = []
for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is NOT installed")
        missing.append(package)

print()

if missing:
    print("ERROR: Missing required packages:")
    for pkg in missing:
        print(f"  - {pkg}")
    print()
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)

# Check if arkit_scenes module can be imported
try:
    import arkit_scenes
    print("✓ arkit_scenes module can be imported")
except ImportError as e:
    print(f"✗ Cannot import arkit_scenes: {e}")
    sys.exit(1)

# Check if dataset exists
from pathlib import Path

dataset_path = Path("dataset")
if dataset_path.exists():
    print(f"✓ Dataset directory exists: {dataset_path}")
    recordings = list(dataset_path.glob("*/"))
    if recordings:
        print(f"  Found {len(recordings)} recording(s)")
        for rec in recordings[:3]:  # Show first 3
            print(f"    - {rec.name}")
    else:
        print("  ⚠ No recordings found (will download on first run)")
else:
    print("⚠ Dataset directory not found (will be created on first run)")

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print()
print("To run with AI web app integration:")
print("  python -m arkit_scenes --include-highres --enable-webapp")
print()
print("Expected output in Rerun:")
print("  • 3D Scene (left side)")
print("  • 2x2 grid of images (right side):")
print("    - RGB Original (top left)")
print("    - Depth Original (top right)")
print("    - RGB Edited with AI (bottom left)")
print("    - Depth Edited with AI (bottom right)")
print()
