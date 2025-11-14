#!/usr/bin/env python3
"""
Minimal test to verify web app integration is working.
This creates a simple example with original and edited images.
"""

import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb

print("=" * 70)
print("MINIMAL WEB APP INTEGRATION TEST")
print("=" * 70)
print("Creating a simple test with synthetic images...")
print()

# Initialize Rerun
rr.init("webapp_test", spawn=True)

# Create synthetic test images
width, height = 640, 480

# Create original image (blue gradient)
original_bgr = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(height):
    original_bgr[i, :] = [255 * i // height, 0, 255 - 255 * i // height]

# Create "edited" image (red gradient)
edited_bgr = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(height):
    edited_bgr[i, :] = [0, 255 * i // height, 255 - 255 * i // height]

# Create depth images
original_depth = np.random.randint(0, 65535, (height, width), dtype=np.uint16)
edited_depth_colorized = cv2.applyColorMap(
    cv2.normalize(original_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
    cv2.COLORMAP_TURBO
)

# Set up blueprint with 2x2 grid
blueprint = rrb.Horizontal(
    rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial2DView(name="RGB Original", contents=["test/bgr/original"]),
            rrb.Spatial2DView(name="Depth Original", contents=["test/depth/original"]),
        ),
        rrb.Horizontal(
            rrb.Spatial2DView(name="RGB Edited (AI)", contents=["test/bgr/edited"]),
            rrb.Spatial2DView(name="Depth Edited (AI)", contents=["test/depth/edited"]),
        ),
        row_shares=[1, 1],
    ),
)

rr.send_blueprint(blueprint)

# Log the images
print("Logging images to Rerun...")
rr.log("test/bgr/original", rr.Image(original_bgr, color_model="BGR"))
rr.log("test/bgr/edited", rr.Image(edited_bgr, color_model="BGR"))
rr.log("test/depth/original", rr.DepthImage(original_depth, meter=1000))
rr.log("test/depth/edited", rr.Image(edited_depth_colorized, color_model="BGR"))

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print()
print("Check the Rerun viewer - you should see:")
print("  • 2x2 grid with 4 panels:")
print("    - Top Left: Blue gradient (RGB Original)")
print("    - Top Right: Random noise (Depth Original)")
print("    - Bottom Left: Red/Green gradient (RGB Edited)")
print("    - Bottom Right: Colorized depth (Depth Edited)")
print()
print("If you see this layout, then the blueprint works!")
print("If not, there may be an issue with your Rerun viewer version.")
print("=" * 70)
