# AI Web App Integration Guide

This guide explains how to use the AI image editing web app integration with the ARKit Scenes Rerun example.

## Quick Start

### Option 1: Using Mock Implementation (No Web Server Required)

The simplest way to test the integration:

```bash
python -m arkit_scenes --include-highres --enable-webapp
```

Or use the test script:
```bash
./test_webapp_integration.sh
```

This will:
1. Load all frames from the ARKit dataset
2. Batch process high-res RGB and depth images with mock AI editing
3. Display both original and edited images in Rerun with 4 tabs

### Option 2: Using the Example Web Server

To test with a real web server (still using mock AI):

```bash
# Terminal 1: Start the Flask server
python example_webapp.py

# Terminal 2: Run with actual HTTP requests (requires aiohttp)
python -m arkit_scenes --include-highres --enable-webapp --api-url http://localhost:8000/edit
```

## What You'll See in Rerun

When `--enable-webapp` is enabled, you'll see **4 tabs** in the Rerun viewer:

1. **BGR Original** - Original RGB camera images with 3D bounding box annotations
2. **BGR Edited** - AI-edited RGB images (artistic bilateral filter in mock)
3. **Depth Original** - Original depth images with annotations
4. **Depth Edited** - AI-edited depth (turbo colormap in mock)

## Console Output

When webapp integration is active, you'll see:

```
============================================================
AI Web App Integration ENABLED
============================================================
Images will be processed with mock AI editing
Look for tabs: 'BGR Original', 'BGR Edited', 'Depth Original', 'Depth Edited'
============================================================

Processing frames…
Loading frames: 100%|████████████████████| 100/100

============================================================
BATCH PROCESSING WITH AI WEB APP
============================================================
Processing 100 frames with AI editing...
(Using mock implementation - artistic filter + depth colormap)
✓ Successfully processed 100 frames
============================================================

Logging to Rerun…
Logging frames: 100%|████████████████████| 100/100
```

## Architecture

### Two-Pass Processing

1. **First Pass - Data Loading**
   - Load all image frames
   - Load camera poses and intrinsics
   - Collect high-res images for processing

2. **Batch Processing** (if webapp enabled)
   - Encode images to base64 (for API compatibility)
   - Process all images asynchronously
   - Store edited results

3. **Second Pass - Logging**
   - Log original images to `world/camera_highres/bgr/original`
   - Log edited images to `world/camera_highres/bgr/edited`
   - Same for depth images

### Mock AI Functions

Currently implemented in `__main__.py`:

- `mock_ai_image_edit()` - Applies filters locally
  - **RGB**: Artistic bilateral filter
  - **Depth**: Turbo colormap

- `process_image_with_webapp()` - Simulates async API calls
- `batch_process_images()` - Processes all images concurrently

## Integrating Your Own AI Model

### Method 1: Replace Mock Functions

Edit `arkit_scenes/__main__.py` and replace `mock_ai_image_edit()`:

```python
def mock_ai_image_edit(image: np.ndarray, edit_type: str = "edge_detection") -> np.ndarray:
    # Replace with your AI model
    if edit_type == "artistic":
        # Call Stable Diffusion, SAM2, etc.
        result = your_ai_model.process(image)
        return result
    # ...
```

### Method 2: Use Real Web Server

Edit `example_webapp.py` and replace `apply_ai_editing()`:

```python
def apply_ai_editing(image: np.ndarray, is_depth: bool = False) -> np.ndarray:
    if not is_depth:
        # Example: Use Stable Diffusion
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("...")
        result = pipe(prompt="...", image=image).images[0]
        return np.array(result)
```

### Method 3: Connect to External API

If you have an existing web service, update `process_image_with_webapp()` to use real HTTP:

```python
async def process_image_with_webapp(image, is_depth, api_url):
    import aiohttp

    base64_image = encode_image_to_base64(image)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_url,
            json={"image": base64_image, "is_depth": is_depth}
        ) as response:
            result = await response.json()

    return decode_image_from_base64(result["edited_image"])
```

## API Specification

If building your own web service, implement this interface:

### Request

```json
POST /edit
Content-Type: application/json

{
  "image": "base64_encoded_png_string",
  "is_depth": false
}
```

### Response

```json
{
  "edited_image": "base64_encoded_png_string",
  "status": "success"
}
```

## Troubleshooting

### "No tabs showing up"

Make sure you're using BOTH flags:
```bash
python -m arkit_scenes --include-highres --enable-webapp
```

### "No high-resolution images found"

The dataset needs to have high-res images. Try a different video ID:
```bash
python -m arkit_scenes --video-id 48458663 --include-highres --enable-webapp
```

### "Images look the same"

The mock implementation applies subtle artistic filters. To see more dramatic changes, edit the `mock_ai_image_edit()` function.

## Performance Notes

- **Batch processing**: All images are processed asynchronously for efficiency
- **Mock delay**: Set to 0.01s per image to simulate network latency
- **Real API**: Remove the mock delay and implement actual HTTP requests
- **Memory**: All images are loaded into memory; for very large datasets, consider streaming

## Examples of AI Models to Integrate

- **Stable Diffusion** - Image-to-image generation
- **ControlNet** - Depth-guided image editing
- **SAM2** - Segment Anything for object segmentation
- **DepthAnything** - Depth estimation enhancement
- **ESRGAN** - Super-resolution
- **Style Transfer** - Neural style transfer
- **Inpainting** - Object removal/addition

## Command Line Options

```
--include-highres       Required for webapp integration
--enable-webapp         Enable AI image editing integration
--api-url URL          Specify custom API endpoint (default: http://localhost:8000/edit)
--video-id ID          Choose ARKit dataset video (default: 48458663)
```

## Next Steps

1. Test with mock implementation first
2. Verify all 4 tabs appear in Rerun
3. Integrate your AI model
4. Adjust processing parameters as needed
5. Deploy to production web service if needed
