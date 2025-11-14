<!--[metadata]
title = "ARKit scenes"
tags = ["2D", "3D", "Depth", "Mesh", "Object detection", "Pinhole camera", "Blueprint"]
thumbnail = "https://static.rerun.io/arkit-scenes/6d920eaa42fb86cfd264d47180ecbecbb6dd3e09/480w.png"
thumbnail_dimensions = [480, 480]
channel = "main"
-->

This example visualizes the [ARKitScenes dataset](https://github.com/apple/ARKitScenes/) using Rerun.
The dataset contains color images, depth images, the reconstructed mesh, and labeled bounding boxes around furniture.

<picture data-inline-viewer="examples/arkit_scenes">
  <source media="(max-width: 480px)" srcset="https://static.rerun.io/arkit_scenes/fb9ec9e8d965369d39d51b17fc7fc5bae6be10cc/480w.png">
  <source media="(max-width: 768px)" srcset="https://static.rerun.io/arkit_scenes/fb9ec9e8d965369d39d51b17fc7fc5bae6be10cc/768w.png">
  <source media="(max-width: 1024px)" srcset="https://static.rerun.io/arkit_scenes/fb9ec9e8d965369d39d51b17fc7fc5bae6be10cc/1024w.png">
  <source media="(max-width: 1200px)" srcset="https://static.rerun.io/arkit_scenes/fb9ec9e8d965369d39d51b17fc7fc5bae6be10cc/1200w.png">
  <img src="https://static.rerun.io/arkit_scenes/fb9ec9e8d965369d39d51b17fc7fc5bae6be10cc/full.png" alt="ARKit Scenes screenshot">
</picture>

## Used Rerun types
[`Image`](https://www.rerun.io/docs/reference/types/archetypes/image),
[`DepthImage`](https://www.rerun.io/docs/reference/types/archetypes/depth_image), [`Transform3D`](https://www.rerun.io/docs/reference/types/archetypes/transform3d),
[`Pinhole`](https://www.rerun.io/docs/reference/types/archetypes/pinhole), [`Mesh3D`](https://www.rerun.io/docs/reference/types/archetypes/mesh3d),
[`Boxes3D`](https://www.rerun.io/docs/reference/types/archetypes/boxes3d),
[`TextDocument`](https://www.rerun.io/docs/reference/types/archetypes/text_document)

## Background

The ARKitScenes dataset, captured using Apple's ARKit technology, encompasses a diverse array of indoor scenes, offering color and depth images, reconstructed 3D meshes, and labeled bounding boxes around objects like furniture. It's a valuable resource for researchers and developers in computer vision and augmented reality, enabling advancements in object recognition, depth estimation, and spatial understanding.

## Logging and visualizing with Rerun
This visualization through Rerun highlights the dataset's potential in developing immersive AR experiences and enhancing machine learning models for real-world applications while showcasing Reruns visualization capabilities.

### Logging a moving RGB-D camera
To log a moving RGB-D camera, we log four key components: the camera's intrinsics via a pinhole camera model, its pose or extrinsics, along with the color and depth images. The camera intrinsics, which define the camera's lens properties, and the pose, detailing its position and orientation, are logged to create a comprehensive 3D to 2D mapping. Both the RGB and depth images are then logged as child entities, capturing the visual and depth aspects of the scene, respectively. This approach ensures a detailed recording of the camera's viewpoint and the scene it captures, all stored compactly under the same entity path for simplicity.
```python
rr.log("world/camera_lowres", rr.Transform3D(transform=camera_from_world))
rr.log("world/camera_lowres", rr.Pinhole(image_from_camera=intrinsic, resolution=[w, h]))
rr.log(f"{entity_id}/rgb", rr.Image(rgb).compress(jpeg_quality=95))
rr.log(f"{entity_id}/depth", rr.DepthImage(depth, meter=1000))
```

### Ground-truth mesh
The mesh is logged as an [rr.Mesh3D archetype](https://www.rerun.io/docs/reference/types/archetypes/mesh3d).
In this case the mesh is composed of mesh vertices, indices (i.e., which vertices belong to the same face), and vertex
colors.
```python
rr.log(
    "world/mesh",
    rr.Mesh3D(
        vertex_positions=mesh.vertices,
        vertex_colors=mesh.visual.vertex_colors,
        triangle_indices=mesh.faces,
    ),
    static=True,
)
```
Here, the mesh is logged to the world/mesh entity and is marked as static, since it does not change in the context of this visualization.

### Logging 3D bounding boxes
Here we loop through the data and add bounding boxes to all the items found.
```python
for i, label_info in enumerate(annotation["data"]):
    rr.log(
        f"world/annotations/box-{uid}-{label}",
        rr.Boxes3D(
            half_sizes=half_size,
            centers=centroid,
            labels=label,
            colors=colors[i],
        ),
        rr.InstancePoses3D(mat3x3=mat3x3),
        static=True,
    )
```

### Setting up the default blueprint

This example benefits at lot from having a custom blueprint defined. This happens with the following code:

```python
primary_camera_entity = HIGHRES_ENTITY_PATH if args.include_highres else LOWRES_POSED_ENTITY_PATH

blueprint = rrb.Horizontal(
    rrb.Spatial3DView(name="3D"),
    rrb.Vertical(
        rrb.Tabs(
            rrb.Spatial2DView(
                name="RGB",
                origin=primary_camera_entity,
                contents=["$origin/rgb", "/world/annotations/**"],
            ),
            rrb.Spatial2DView(
                name="Depth",
                origin=primary_camera_entity,
                contents=["$origin/depth", "/world/annotations/**"],
            ),
            name="2D",
        ),
        rrb.TextDocumentView(name="Readme"),
    ),
)

rr.script_setup(args, "rerun_example_arkit_scenes", default_blueprint=blueprint)
```

In particular, we want to reproject 3D annotations onto the 2D camera views. To configure such a view, two things are necessary:
- The view origin must be set to the entity that contains the pinhole transforms. In this example, the entity path is stored in the `primary_camera_entity` variable.
- The view contents must explicitly include the annotations, which are not logged in the subtree defined by the origin. This is done using the `contents` argument, here set to `["$origin/depth", "/world/annotations/**"]`.


## AI Image Editing Web App Integration

This example has been enhanced with AI image editing web app integration! When enabled, images are sent to an external web app for processing, and both original and edited versions are visualized side-by-side in Rerun.

### Features
- **Batch async processing**: All high-resolution RGB and depth images are collected and processed in batches for efficiency
- **Base64 JSON format**: Images are encoded/decoded for API compatibility
- **Side-by-side comparison**: Custom Rerun blueprint shows original vs edited images in separate tabs
- **Mock implementation**: Currently uses local image processing (artistic filter for RGB, colormap for depth) to simulate web app behavior

### Usage
To enable web app integration:
```bash
python -m arkit_scenes --include-highres --enable-webapp
```

Available options:
- `--enable-webapp`: Enable AI image editing integration
- `--api-url URL`: Specify custom web app API endpoint (default: http://localhost:8000/edit)
- `--include-highres`: Required for webapp integration (processes high-res images)

### Implementing Your Own Web App

The mock functions can be replaced with real HTTP requests. Replace the mock implementation in `process_image_with_webapp()`:

```python
async def process_image_with_webapp(image: np.ndarray, is_depth: bool = False, api_url: str = "http://localhost:8000/edit") -> np.ndarray:
    # Encode image to base64
    base64_image = encode_image_to_base64(image)

    # Send to your web app
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json={"image": base64_image, "is_depth": is_depth}) as response:
            result = await response.json()

    # Decode edited image
    edited = decode_image_from_base64(result["edited_image"])
    return edited
```

Your web app should accept POST requests with JSON body:
```json
{
  "image": "base64_encoded_image_string",
  "is_depth": false
}
```

And return:
```json
{
  "edited_image": "base64_encoded_edited_image_string"
}
```

## Interactive Frame-by-Frame Editing

This example now includes an **interactive editing mode** where you can edit specific frames on-demand using keyboard controls!

### Features
- **Keyboard-driven editing**: Navigate frames with arrow keys, press 'E' to edit the current frame
- **Real-time visualization**: Edited frames appear instantly in Rerun viewer
- **Super obvious edits**: Color inversion + thick red border + "EDITED" watermark + frame timestamp
- **On-demand processing**: Only edit the frames you want, not all frames
- **Edit and re-edit**: Edit any frame multiple times, or reset to original

### Usage

```bash
# Install interactive dependencies
pip install -e .[interactive]

# For AI-powered editing with Gemini, also install:
pip install -e .[gemini]

# Run in interactive mode (mock edits)
python -m arkit_scenes --include-highres --interactive

# Run with Gemini AI editing (uses default API key)
python -m arkit_scenes --include-highres --interactive --gemini

# Or use your own API key
python -m arkit_scenes --include-highres --interactive --gemini-api-key YOUR_API_KEY
```

### Keyboard Controls

Once in interactive mode, use these controls:

| Key | Action |
|-----|--------|
| ← or A | Previous frame |
| → or D | Next frame |
| E | Edit current frame (apply color invert + watermark) |
| R | Reset current frame to original |
| H | Show help |
| Q or ESC | Quit interactive mode |

### AI-Powered Editing with Gemini

When you provide a Gemini API key, you get **real AI image editing**:

1. Press `e` to edit a frame
2. Terminal prompts: **"Enter editing instructions:"**
3. Type what you want (e.g., "make the jacket red", "add sunglasses", "change to sunset lighting")
4. Gemini API processes the image
5. **Automatic depth map generation** from the edited image using Depth Anything V2
6. Both edited RGB and depth appear in Rerun!

**Example prompts:**
- "make the jacket red"
- "add a plant in the corner"
- "change the lighting to sunset"
- "add sunglasses to the person"
- "make the walls blue"

### What You'll See

When you run in interactive mode:

1. **Initial logging**: All frames are logged as "original" first
2. **Rerun viewer opens**: You can see all the original frames
3. **Interactive prompt**: Terminal shows current frame status
4. **Navigate & Edit**: Use arrow keys to move between frames, press 'E' to edit
5. **AI or Mock Edit**: Either provide a Gemini prompt or use mock edits
6. **Instant updates**: Edited frames + depth maps appear in Rerun immediately

#### Terminal Example (with Gemini):
```
============================================================
INTERACTIVE EDITING MODE - ACTIVE
============================================================
Frame: 15/57 | Time: 12.450s | Status: Original

Command: e

======================================================================
EDITING FRAME 12.450
======================================================================

Enter editing instructions (or press Enter to use mock edit):
Prompt: make the jacket red

🤖 Calling Gemini API with prompt: 'make the jacket red'
✓ Image edited successfully! Size: 1920x1440
✓ Logged edited RGB image at timestamp 12.450
🗺️  Generating depth map from edited image...
✓ Depth map generated! Size: 1920x1440

======================================================================
✓ FRAME 12.450 EDITED AND LOGGED
======================================================================
  → RGB: world/camera_highres/bgr/edited
  → Depth: world/camera_highres/depth/edited
  → Timestamp: 12.450s
  → Scrub timeline in Rerun to see the edit!
======================================================================

Frame: 15/57 | Time: 12.450s | Status: EDITED
```

#### In Rerun Viewer:

The edited frames will appear at:
- `world/camera_highres/bgr/edited` - RGB image with **inverted colors + "EDITED" overlay**
- `world/camera_highres/depth/edited` - Depth with **jet colormap + "EDITED" overlay**

You can compare original vs edited by:
1. Opening the left sidebar entity tree
2. Clicking on `bgr/original` to see original
3. Clicking on `bgr/edited` to see your edits

### Edit Effects Applied

Each edited frame gets these obvious transformations:

1. **Color Inversion**: All pixel values inverted (255 - original)
2. **Red Border**: 10-pixel thick red frame around the image
3. **"EDITED" Watermark**: Large green text at the top center
4. **Frame Timestamp**: White text in bottom right showing frame ID

These effects make it **immediately obvious** which frames have been edited!

### Example Workflow

```bash
# 1. Start interactive mode
python -m arkit_scenes --include-highres --interactive

# 2. Wait for Rerun viewer to open

# 3. In terminal, navigate to interesting frames:
#    - Press → to go forward
#    - Press ← to go back

# 4. When you find a frame to edit:
#    - Press E to edit it
#    - Check Rerun viewer to see the edited result

# 5. Continue editing more frames or press Q to quit
```

### Customizing Edit Effects

To change what edits are applied, modify `apply_obvious_edit()` in `arkit_scenes/interactive_editor.py`:

```python
def apply_obvious_edit(image: np.ndarray, frame_id: str) -> np.ndarray:
    # Your custom edits here
    # Examples:
    # - Apply a different filter
    # - Add custom annotations
    # - Run your AI model
    # - etc.
    return edited_image
```

## Run the code

To run this example, make sure you have the Rerun repository checked out and the latest SDK installed:
```bash
pip install --upgrade rerun-sdk  # install the latest Rerun SDK
git clone git@github.com:rerun-io/rerun.git  # Clone the repository
cd rerun
git checkout latest  # Check out the commit matching the latest SDK release
```

Install the necessary libraries specified in the requirements file:
```bash
pip install -e examples/python/arkit_scenes
```

For web app integration or interactive mode, install optional dependencies:
```bash
# For web app integration
pip install -e examples/python/arkit_scenes[webapp]

# For interactive editing mode
pip install -e examples/python/arkit_scenes[interactive]

# For all optional features
pip install -e examples/python/arkit_scenes[all]
```

To run this example use:
```bash
# Basic usage (original functionality)
python -m arkit_scenes

# With high-resolution images
python -m arkit_scenes --include-highres

# With AI web app integration (batch process all frames)
python -m arkit_scenes --include-highres --enable-webapp

# With interactive editing mode (edit frames on-demand)
python -m arkit_scenes --include-highres --interactive
```

### Running the Example Web App

An example Flask server is provided in `example_webapp.py`:

```bash
# Terminal 1: Start the web app server
python example_webapp.py

# Terminal 2: Run ARKit with web app integration
python -m arkit_scenes --include-highres --enable-webapp
```

The example server applies artistic filters to RGB images and colormaps to depth images. You can modify `apply_ai_editing()` in `example_webapp.py` to integrate your own AI models (Stable Diffusion, SAM2, ControlNet, etc.).

