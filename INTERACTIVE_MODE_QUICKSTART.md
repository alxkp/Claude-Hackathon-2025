# Interactive Frame Editing - Quick Start Guide

## What is This?

Interactive mode lets you **edit specific frames on-demand** using keyboard controls while viewing your ARKit Scenes data in Rerun. Perfect for:
- Quickly testing image editing pipelines
- Annotating or modifying specific frames
- Experimenting with different edit effects

## Installation

```bash
# Install interactive mode dependencies
pip install pynput

# Or install as optional dependency
pip install -e .[interactive]
```

## Quick Start

### Step 1: Launch Interactive Mode

```bash
python -m arkit_scenes --include-highres --interactive
```

### Step 2: Wait for Initial Loading

You'll see:
```
Loading frames: 100%|████████████| 1971/1971 [00:08<00:00]
Logging to Rerun…
Logging frames: 100%|████████████| 1971/1971 [00:06<00:00]

============================================================
STARTING INTERACTIVE MODE
============================================================
```

The Rerun viewer will open automatically.

### Step 3: Navigate and Edit

**Terminal shows current frame:**
```
Frame: 1/57 | Time: 12.345s | Status: Original
```

**Use keyboard controls:**

| Key | What It Does |
|-----|--------------|
| → or D | Next frame |
| ← or A | Previous frame |
| **E** | **Edit current frame** (inverts colors + adds watermark) |
| R | Reset edited frame back to original |
| H | Show help message |
| Q | Quit interactive mode |

### Step 4: See Your Edits in Rerun

When you press **E**:

1. Terminal shows: `✓ Frame 12.345 edited and logged to Rerun`
2. Status changes to: `Status: EDITED`
3. **Rerun viewer updates instantly** with the edited frame

In Rerun, you'll see the edited version at:
- `world/camera_highres/bgr/edited`
- `world/camera_highres/depth/edited`

## What the Edits Look Like

Each edited frame gets these **super obvious** changes:

✓ **Inverted colors** - All pixels flipped (255 - original)
✓ **Thick red border** - 10px frame around the image
✓ **"EDITED" text** - Large green watermark at top
✓ **Frame timestamp** - White text in bottom corner

**You can't miss it!**

## Example Session

```bash
# Start
$ python -m arkit_scenes --include-highres --interactive

[Loading and initial logging happens...]

============================================================
INTERACTIVE EDITING MODE - ACTIVE
============================================================

Rerun viewer is open. Navigate frames and press 'E' to edit.
Press 'H' for help, 'Q' to quit.
============================================================
Frame: 1/57 | Time: 12.345s | Status: Original

[Press → a few times to navigate forward]

Frame: 5/57 | Time: 13.120s | Status: Original

[Press E to edit this frame]

Editing frame 13.120...
✓ Frame 13.120 edited and logged to Rerun
Frame: 5/57 | Time: 13.120s | Status: EDITED

[Continue editing more frames or press Q to quit]
```

## Tips

### Finding Frames to Edit

1. **Scrub timeline in Rerun** - Move the playhead to find interesting moments
2. **Note the timestamp** shown in Rerun
3. **Navigate to that frame** in terminal using arrow keys
4. **Press E** to edit it

### Viewing Original vs Edited

In Rerun's left sidebar (entity tree):

```
world/
  └── camera_highres/
      ├── bgr/
      │   ├── original  ← Click to see original
      │   └── edited    ← Click to see your edits
      └── depth/
          ├── original
          └── edited
```

### Editing Multiple Frames

You can edit as many frames as you want! Just navigate and press E for each one.

### Undoing Edits

Press **R** on an edited frame to reset it to original (removes the edited version).

## Customizing the Edits

Want different edit effects? Edit this function in `arkit_scenes/interactive_editor.py`:

```python
def apply_obvious_edit(image: np.ndarray, frame_id: str) -> np.ndarray:
    """
    Apply your custom edits here!

    Ideas:
    - Run Stable Diffusion img2img
    - Apply SAM2 segmentation
    - Add custom overlays
    - Run your ML model
    """
    # Current: inverts colors and adds watermark
    edited = 255 - image  # Invert

    # Add your custom processing here!

    return edited
```

## Troubleshooting

### "pynput not installed" Error

```bash
pip install pynput
```

### Keyboard Not Working

Make sure the **terminal window has focus**, not the Rerun viewer.

### No High-Res Frames

Make sure you're using `--include-highres` flag:
```bash
python -m arkit_scenes --include-highres --interactive
```

### Can't See Edited Frames in Rerun

Check the left sidebar entity tree:
1. Expand `world` → `camera_highres`
2. Look for `bgr/edited` and `depth/edited`
3. Click on them to view

## Next Steps

- **Customize edits**: Modify `apply_obvious_edit()` in `interactive_editor.py`
- **Integrate AI models**: Add your ML model calls in the edit function
- **Automate**: Use the edit callback to process frames programmatically
- **Export**: Save edited frames to disk for further processing

## Command Reference

```bash
# Basic interactive mode
python -m arkit_scenes --include-highres --interactive

# With specific video
python -m arkit_scenes --video-id 48458663 --include-highres --interactive

# Get help
python -m arkit_scenes --help
```

---

**Happy editing! 🎨**
