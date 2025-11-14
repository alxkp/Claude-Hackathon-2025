#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
import trimesh
from dotenv import load_dotenv
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from .download_dataset import AVAILABLE_RECORDINGS, ensure_recording_available
from .interactive_editor import FrameEditor, apply_obvious_edit, start_interactive_mode
from .gemini_editor import get_editor

# Load environment variables from .env file
load_dotenv()

if TYPE_CHECKING:
    from pathlib import Path

DESCRIPTION = """
# ARKitScenes
This example visualizes the [ARKitScenes dataset](https://github.com/apple/ARKitScenes/) using Rerun. The dataset
contains color images, depth images, the reconstructed mesh, and labeled bounding boxes around furniture.

The full source code for this example is available
[on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/arkit_scenes).
""".strip()

Color = tuple[float, float, float, float]

# hack for now since dataset does not provide orientation information, only known after initial visual inspection
ORIENTATION = {
    "48458663": "landscape",
    "42444949": "portrait",
    "41069046": "portrait",
    "41125722": "portrait",
    "41125763": "portrait",
    "42446167": "portrait",
}
assert len(ORIENTATION) == len(AVAILABLE_RECORDINGS)
assert set(ORIENTATION.keys()) == set(AVAILABLE_RECORDINGS)

LOWRES_POSED_ENTITY_PATH = "world/camera_lowres"
HIGHRES_ENTITY_PATH = "world/camera_highres"


# ==============================================================================
# Mock AI Image Editing Web App Functions
# ==============================================================================


def mock_ai_image_edit(image: np.ndarray, edit_type: str = "edge_detection") -> np.ndarray:
    """
    Mock AI image editing function that simulates sending to a web app.
    In production, this would make an HTTP request to your web app.

    Args:
        image: Input image as numpy array (BGR or grayscale)
        edit_type: Type of editing to apply

    Returns:
        Edited image with same shape as input
    """
    if edit_type == "edge_detection":
        # Apply edge detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, 100, 200)
        # Convert back to BGR if input was BGR
        if len(image.shape) == 3:
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            result = edges
        return result

    elif edit_type == "color_pop":
        # Enhance saturation and contrast
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Increase saturation
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)  # Increase brightness
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        else:
            result = np.clip(image.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        return result

    elif edit_type == "artistic":
        # Apply artistic bilateral filter
        result = cv2.bilateralFilter(image, 9, 75, 75)
        return result

    elif edit_type == "depth_colormap":
        # For depth images, apply a colormap
        if len(image.shape) == 2 or image.shape[2] == 1:
            # Normalize depth to 0-255 range
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            result = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            return result
        return image

    else:
        # Default: return input unchanged
        return image


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode numpy image to base64 string (simulating API request format)."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def decode_image_from_base64(base64_str: str) -> np.ndarray:
    """Decode base64 string to numpy image (simulating API response)."""
    image_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)


async def process_image_with_webapp(
    image: np.ndarray,
    is_depth: bool = False,
    api_url: str = "http://localhost:8000/edit"
) -> np.ndarray:
    """
    Async function to process image with web app API.
    Currently mocked - replace with actual HTTP request for production.

    Args:
        image: Input image
        is_depth: Whether this is a depth image
        api_url: API endpoint URL (unused in mock)

    Returns:
        Edited image
    """
    # Simulate network delay
    await asyncio.sleep(0.01)

    # Mock: encode to base64 (as if sending to API)
    # base64_image = encode_image_to_base64(image)

    # Mock: apply image editing locally (simulating web app processing)
    if is_depth:
        edited = mock_ai_image_edit(image, edit_type="depth_colormap")
    else:
        edited = mock_ai_image_edit(image, edit_type="artistic")

    # Mock: decode from base64 (as if receiving from API)
    # In production, you would do:
    # response = await http_client.post(api_url, json={"image": base64_image})
    # edited = decode_image_from_base64(response.json()["edited_image"])

    return edited


def load_json(js_path: Path) -> dict[str, Any]:
    with open(js_path, encoding="utf8") as f:
        json_data: dict[str, Any] = json.load(f)
    return json_data


def log_annotated_bboxes(annotation: dict[str, Any]) -> None:
    """
    Logs annotated oriented bounding boxes to Rerun.

    annotation json file
    |  |-- label: object name of bounding box
    |  |-- axesLengths[x, y, z]: size of the origin bounding-box before transforming
    |  |-- centroid[]: the translation matrix (1,3) of bounding-box
    |  |-- normalizedAxes[]: the rotation matrix (3,3) of bounding-box
    """

    for label_info in annotation["data"]:
        uid = label_info["uid"]
        label = label_info["label"]

        half_size = 0.5 * np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)[0]
        centroid = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)[0]
        mat3x3 = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3).T

        rr.log(
            f"world/annotations/box-{uid}-{label}",
            rr.Boxes3D(
                half_sizes=half_size,
                labels=label,
            ),
            rr.InstancePoses3D(translations=centroid, mat3x3=mat3x3),
            static=True,
        )


def log_camera(
    intri_path: Path,
    frame_id: str,
    poses_from_traj: dict[str, rr.Transform3D],
    entity_id: str,
) -> None:
    """Logs camera transform and 3D bounding boxes in the image frame."""
    w, h, fx, fy, cx, cy = np.loadtxt(intri_path)
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera_from_world = poses_from_traj[frame_id]

    # clear previous centroid labels
    rr.log(f"{entity_id}/bbox-2D-segments", rr.Clear(recursive=True))

    # pathlib makes it easy to get the parent, but log methods requires a string
    rr.log(entity_id, camera_from_world)
    rr.log(entity_id, rr.Pinhole(image_from_camera=intrinsic, resolution=[w, h]))


def read_camera_from_world(traj_string: str) -> tuple[str, rr.Transform3D]:
    """
    Reads out camera_from_world transform from trajectory string.

    Args:
    ----
    traj_string:
        A space-delimited file where each line represents a camera position at a particular timestamp.
            The file has seven columns:
            * Column 1: timestamp
            * Columns 2-4: rotation (axis-angle representation in radians)
            * Columns 5-7: translation (usually in meters)

    Returns
    -------
    timestamp: float
        timestamp in seconds
    camera_from_world: tuple of two numpy arrays
        A tuple containing a translation vector and a quaternion that represent the camera_from_world transform

    Raises
    ------
    AssertionError:
        If the input string does not contain 7 tokens.

    """
    tokens = traj_string.split()  # Split the input string into tokens
    assert len(tokens) == 7, f"Input string must have 7 tokens, but found {len(tokens)}."
    ts: str = tokens[0]  # Extract timestamp from the first token

    # Extract rotation from the second to fourth tokens
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    rotation = R.from_rotvec(np.asarray(angle_axis))

    # Extract translation from the fifth to seventh tokens
    translation = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])

    # Create tuple in format log_transform3d expects
    camera_from_world = rr.Transform3D(
        translation=translation,
        rotation=rr.Quaternion(xyzw=rotation.as_quat()),
        relation=rr.TransformRelation.ChildFromParent,
    )

    return (ts, camera_from_world)


def find_closest_frame_id(target_id: str, frame_ids: dict[str, Any]) -> str:
    """Finds the closest frame id to the target id."""
    target_value = float(target_id)
    closest_id = min(frame_ids.keys(), key=lambda x: abs(float(x) - target_value))
    return closest_id


async def batch_process_images(images_to_process: list[dict]) -> list[dict]:
    """
    Process a batch of images asynchronously with the web app.

    Args:
        images_to_process: List of dicts containing image data and metadata

    Returns:
        List of dicts with edited images added
    """
    tasks = []
    for item in images_to_process:
        if "bgr" in item:
            tasks.append(process_image_with_webapp(item["bgr"], is_depth=False))
        if "depth" in item:
            tasks.append(process_image_with_webapp(item["depth"], is_depth=True))

    results = await asyncio.gather(*tasks)

    # Add edited images back to the items
    result_idx = 0
    for item in images_to_process:
        if "bgr" in item:
            item["bgr_edited"] = results[result_idx]
            result_idx += 1
        if "depth" in item:
            item["depth_edited"] = results[result_idx]
            result_idx += 1

    return images_to_process


def log_arkit(
    recording_path: Path,
    include_highres: bool,
    enable_webapp: bool = False,
    enable_interactive: bool = False
) -> dict[str, Any] | None:
    """
    Logs ARKit recording data using Rerun.

    Args:
    ----
    recording_path (Path):
        The path to the ARKit recording.

    include_highres (bool):
        Whether to include high resolution data.

    enable_webapp (bool):
        Whether to enable web app integration for AI image editing.

    enable_interactive (bool):
        Whether to enable interactive frame-by-frame editing mode.

    Returns
    -------
    dict | None:
        If interactive mode, returns frame metadata for editing.
        Otherwise returns None.

    """
    rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), static=True)

    video_id = recording_path.stem
    lowres_image_dir = recording_path / "lowres_wide"
    image_dir = recording_path / "wide"
    lowres_depth_dir = recording_path / "lowres_depth"
    depth_dir = recording_path / "highres_depth"
    lowres_intrinsics_dir = recording_path / "lowres_wide_intrinsics"
    intrinsics_dir = recording_path / "wide_intrinsics"
    traj_path = recording_path / "lowres_wide.traj"

    # frame_ids are indexed by timestamps, you can see more info here
    # https://github.com/apple/ARKitScenes/blob/main/threedod/README.md#data-organization-and-format-of-input-data
    depth_filenames = [x.name for x in sorted(lowres_depth_dir.iterdir())]
    lowres_frame_ids = [x.split(".png")[0].split("_")[1] for x in depth_filenames]
    lowres_frame_ids.sort()

    # dict of timestamp to pose which is a tuple of translation and quaternion
    camera_from_world_dict = {}
    with open(traj_path, encoding="utf-8") as f:
        trajectory = f.readlines()

    for line in trajectory:
        timestamp, camera_from_world = read_camera_from_world(line)
        # round timestamp to 3 decimal places as seen in the original repo here
        # https://github.com/apple/ARKitScenes/blob/e2e975128a0a9695ea56fa215fe76b4295241538/threedod/benchmark_scripts/utils/tenFpsDataLoader.py#L247
        timestamp = f"{round(float(timestamp), 3):.3f}"
        camera_from_world_dict[timestamp] = camera_from_world

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    ply_path = recording_path / f"{recording_path.stem}_3dod_mesh.ply"
    print(f"Loading {ply_path}…")
    assert os.path.isfile(ply_path), f"Failed to find {ply_path}"

    mesh = trimesh.load(str(ply_path))
    rr.log(
        "world/mesh",
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            vertex_colors=mesh.visual.vertex_colors,
            triangle_indices=mesh.faces,
        ),
        static=True,
    )

    # load the obb annotations and log them in the world frame
    bbox_annotations_path = recording_path / f"{recording_path.stem}_3dod_annotation.json"
    annotation = load_json(bbox_annotations_path)
    log_annotated_bboxes(annotation)

    print("Processing frames…")

    # First pass: collect all images for batch processing if webapp is enabled
    images_to_process = []
    frame_metadata = []

    for frame_timestamp in tqdm(lowres_frame_ids, desc="Loading frames"):
        # load the lowres image and depth
        bgr = cv2.imread(f"{lowres_image_dir}/{video_id}_{frame_timestamp}.png")
        depth = cv2.imread(f"{lowres_depth_dir}/{video_id}_{frame_timestamp}.png", cv2.IMREAD_ANYDEPTH)

        high_res_exists: bool = (image_dir / f"{video_id}_{frame_timestamp}.png").exists() and include_highres

        frame_data = {
            "timestamp": frame_timestamp,
            "has_pose": frame_timestamp in camera_from_world_dict,
            "high_res_exists": high_res_exists,
        }

        # Store lowres data
        frame_data["lowres_bgr"] = bgr
        frame_data["lowres_depth"] = depth

        # Load highres data if available
        if high_res_exists:
            highres_bgr = cv2.imread(f"{image_dir}/{video_id}_{frame_timestamp}.png")
            highres_depth = cv2.imread(f"{depth_dir}/{video_id}_{frame_timestamp}.png", cv2.IMREAD_ANYDEPTH)
            frame_data["highres_bgr"] = highres_bgr
            frame_data["highres_depth"] = highres_depth
            frame_data["closest_lowres_frame_id"] = find_closest_frame_id(frame_timestamp, camera_from_world_dict)

        frame_metadata.append(frame_data)

    # Batch process images with web app if enabled
    if enable_webapp:
        print("\n" + "=" * 60)
        print("BATCH PROCESSING WITH AI WEB APP")
        print("=" * 60)
        # Collect only highres images for processing (as per user request)
        for frame_data in frame_metadata:
            if frame_data["high_res_exists"]:
                images_to_process.append({
                    "bgr": frame_data["highres_bgr"],
                    "depth": frame_data["highres_depth"],
                    "timestamp": frame_data["timestamp"],
                })

        # Process in batches asynchronously
        if images_to_process:
            print(f"Processing {len(images_to_process)} frames with AI editing...")
            print("(Using mock implementation - artistic filter + depth colormap)")
            processed = asyncio.run(batch_process_images(images_to_process))

            # Map processed images back to frame_metadata
            processed_dict = {item["timestamp"]: item for item in processed}
            edited_count = 0
            for frame_data in frame_metadata:
                if frame_data["timestamp"] in processed_dict:
                    frame_data["highres_bgr_edited"] = processed_dict[frame_data["timestamp"]]["bgr_edited"]
                    frame_data["highres_depth_edited"] = processed_dict[frame_data["timestamp"]]["depth_edited"]
                    edited_count += 1

            print(f"✓ Successfully processed {edited_count} frames")
            print("=" * 60 + "\n")
        else:
            print("No high-resolution images found to process")
            print("=" * 60 + "\n")

    # Second pass: log all data to Rerun
    print("Logging to Rerun…")
    for frame_data in tqdm(frame_metadata, desc="Logging frames"):
        frame_timestamp = frame_data["timestamp"]

        # Set time for this frame
        rr.set_time("time", duration=float(frame_timestamp))

        # Log the camera transforms and lowres images
        if frame_data["has_pose"]:
            lowres_intri_path = lowres_intrinsics_dir / f"{video_id}_{frame_timestamp}.pincam"
            log_camera(
                lowres_intri_path,
                frame_timestamp,
                camera_from_world_dict,
                LOWRES_POSED_ENTITY_PATH,
            )

            rr.log(
                f"{LOWRES_POSED_ENTITY_PATH}/bgr",
                rr.Image(frame_data["lowres_bgr"], color_model="BGR").compress(jpeg_quality=95)
            )
            rr.log(f"{LOWRES_POSED_ENTITY_PATH}/depth", rr.DepthImage(frame_data["lowres_depth"], meter=1000))

        # Log the highres camera and images
        if frame_data["high_res_exists"]:
            rr.set_time("time high resolution", duration=float(frame_timestamp))
            highres_intri_path = intrinsics_dir / f"{video_id}_{frame_timestamp}.pincam"
            log_camera(
                highres_intri_path,
                frame_data["closest_lowres_frame_id"],
                camera_from_world_dict,
                HIGHRES_ENTITY_PATH,
            )

            if enable_webapp or enable_interactive:
                # Log original images with /original suffix for side-by-side comparison
                rr.log(
                    f"{HIGHRES_ENTITY_PATH}/bgr/original",
                    rr.Image(frame_data["highres_bgr"], color_model="BGR").compress(jpeg_quality=75)
                )
                rr.log(
                    f"{HIGHRES_ENTITY_PATH}/depth/original",
                    rr.DepthImage(frame_data["highres_depth"], meter=1000)
                )

                # Log edited images if they exist (only for webapp, not interactive)
                if enable_webapp and "highres_bgr_edited" in frame_data:
                    rr.log(
                        f"{HIGHRES_ENTITY_PATH}/bgr/edited",
                        rr.Image(frame_data["highres_bgr_edited"], color_model="BGR").compress(jpeg_quality=75)
                    )
                    rr.log(
                        f"{HIGHRES_ENTITY_PATH}/depth/edited",
                        rr.Image(frame_data["highres_depth_edited"], color_model="BGR").compress(jpeg_quality=75)
                    )
                elif enable_webapp and "highres_bgr_edited" not in frame_data:
                    # Debug: This shouldn't happen if processing worked
                    if frame_timestamp == lowres_frame_ids[0]:
                        print(f"WARNING: No edited images found for frame {frame_timestamp}")
            else:
                # Original behavior: log without /original suffix
                rr.log(
                    f"{HIGHRES_ENTITY_PATH}/bgr",
                    rr.Image(frame_data["highres_bgr"], color_model="BGR").compress(jpeg_quality=75)
                )
                rr.log(f"{HIGHRES_ENTITY_PATH}/depth", rr.DepthImage(frame_data["highres_depth"], meter=1000))

    # Return frame metadata for interactive mode
    if enable_interactive:
        return {
            "video_id": video_id,
            "frame_metadata": frame_metadata,
            "lowres_frame_ids": lowres_frame_ids,
            "camera_from_world_dict": camera_from_world_dict,
            "intrinsics_dir": intrinsics_dir,
        }

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the ARKitScenes dataset using the Rerun SDK.")
    parser.add_argument(
        "--video-id",
        type=str,
        choices=AVAILABLE_RECORDINGS,
        default=AVAILABLE_RECORDINGS[0],
        help="Video ID of the ARKitScenes Dataset",
    )
    parser.add_argument(
        "--include-highres",
        action="store_true",
        help="Include the high resolution camera and depth images",
    )
    parser.add_argument(
        "--enable-webapp",
        action="store_true",
        help="Enable AI image editing web app integration (shows original vs edited comparison)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/edit",
        help="Web app API endpoint URL (currently using mock implementation)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive frame-by-frame editing mode with keyboard controls",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Enable Gemini AI image editing (uses default API key)",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Google Gemini API key (optional, uses default if --gemini flag is set)",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    # Load API key from environment if --gemini flag is set and no custom key provided
    if args.gemini and not args.gemini_api_key:
        args.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not args.gemini_api_key:
            print("ERROR: GEMINI_API_KEY not found in environment variables.")
            print("Please create a .env file with: GEMINI_API_KEY=your_api_key")
            print("Or provide the key directly with --gemini-api-key")
            import sys
            sys.exit(1)

    # Validate arguments
    if args.enable_webapp and not args.include_highres:
        print("Warning: --enable-webapp requires --include-highres to be set. Disabling webapp integration.")
        args.enable_webapp = False

    if args.interactive and not args.include_highres:
        print("Warning: --interactive requires --include-highres to be set. Disabling interactive mode.")
        args.interactive = False

    if args.interactive and args.enable_webapp:
        print("Warning: --interactive and --enable-webapp cannot be used together. Disabling webapp.")
        args.enable_webapp = False

    if args.enable_webapp:
        print("\n" + "=" * 60)
        print("AI WEB APP INTEGRATION ENABLED")
        print("=" * 60)
        print("Images will be processed with mock AI editing")
        print("")
        print("In Rerun viewer, you will see:")
        print("  • Left: 3D Scene with mesh and camera trajectory")
        print("  • Right (2x2 grid):")
        print("    - Top Left:     RGB Original")
        print("    - Top Right:    Depth Original")
        print("    - Bottom Left:  RGB Edited (AI)")
        print("    - Bottom Right: Depth Edited (AI)")
        print("=" * 60 + "\n")

    primary_camera_entity = HIGHRES_ENTITY_PATH if args.include_highres else LOWRES_POSED_ENTITY_PATH

    # Create blueprint based on whether webapp or interactive is enabled
    if args.interactive and args.include_highres:
        # Interactive mode: show original + edited in timeline scrubbing view
        blueprint = rrb.Horizontal(
            rrb.Spatial3DView(name="3D Scene"),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="RGB Original",
                        origin=primary_camera_entity,
                        contents=["$origin/bgr/original", "/world/annotations/**"],
                    ),
                    rrb.Spatial2DView(
                        name="RGB Edited",
                        origin=primary_camera_entity,
                        contents=["$origin/bgr/edited"],  # Single time series
                    ),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="Depth Original",
                        origin=primary_camera_entity,
                        contents=["$origin/depth/original", "/world/annotations/**"],
                    ),
                    rrb.Spatial2DView(
                        name="Depth Edited",
                        origin=primary_camera_entity,
                        contents=["$origin/depth/edited"],  # Single time series
                    ),
                ),
                rrb.TextDocumentView(name="Info"),
                row_shares=[2, 2, 1],
            ),
            column_shares=[1, 2],
        )
    elif args.enable_webapp and args.include_highres:
        # Webapp mode: original vs edited comparison
        blueprint = rrb.Horizontal(
            rrb.Spatial3DView(name="3D Scene"),
            rrb.Vertical(
                # Top row: Original images (BGR and Depth)
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="RGB Original",
                        origin=primary_camera_entity,
                        contents=["$origin/bgr/original", "/world/annotations/**"],
                    ),
                    rrb.Spatial2DView(
                        name="Depth Original",
                        origin=primary_camera_entity,
                        contents=["$origin/depth/original", "/world/annotations/**"],
                    ),
                ),
                # Bottom row: Edited images (BGR and Depth)
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="RGB Edited (AI)",
                        origin=primary_camera_entity,
                        contents=["$origin/bgr/edited"],
                    ),
                    rrb.Spatial2DView(
                        name="Depth Edited (AI)",
                        origin=primary_camera_entity,
                        contents=["$origin/depth/edited"],
                    ),
                ),
                rrb.TextDocumentView(name="Info"),
                row_shares=[2, 2, 1],
            ),
            column_shares=[1, 2],
        )
    else:
        # Original blueprint
        blueprint = rrb.Horizontal(
            rrb.Spatial3DView(name="3D"),
            rrb.Vertical(
                rrb.Tabs(
                    # Note that we re-project the annotations into the 2D views:
                    # For this to work, the origin of the 2D views has to be a pinhole camera,
                    # this way the viewer knows how to project the 3D annotations into the 2D views.
                    rrb.Spatial2DView(
                        name="BGR",
                        origin=primary_camera_entity,
                        contents=["$origin/bgr", "/world/annotations/**"],
                    ),
                    rrb.Spatial2DView(
                        name="Depth",
                        origin=primary_camera_entity,
                        contents=["$origin/depth", "/world/annotations/**"],
                    ),
                    name="2D",
                ),
                rrb.TextDocumentView(name="Readme"),
                row_shares=[2, 1],
            ),
        )

    rr.script_setup(args, "rerun_example_arkit_scenes", default_blueprint=blueprint)

    if args.enable_webapp or args.interactive:
        print("Blueprint created with 4 views:")
        print("  - 3D Scene")
        print("  - RGB Original")
        print("  - Depth Original")
        print("  - RGB Edited (AI) - will appear when you edit frames")
        print("  - Depth Edited (AI) - will appear when you edit frames")
        print()

    recording_path = ensure_recording_available(args.video_id, args.include_highres)
    result = log_arkit(
        recording_path,
        args.include_highres,
        enable_webapp=args.enable_webapp,
        enable_interactive=args.interactive
    )

    if args.enable_webapp:
        print("\n" + "=" * 70)
        print("LOGGING COMPLETE - Check Rerun Viewer!")
        print("=" * 70)
        print("Data logged to these paths:")
        print(f"  • {HIGHRES_ENTITY_PATH}/bgr/original")
        print(f"  • {HIGHRES_ENTITY_PATH}/bgr/edited")
        print(f"  • {HIGHRES_ENTITY_PATH}/depth/original")
        print(f"  • {HIGHRES_ENTITY_PATH}/depth/edited")
        print()
        print("In the Rerun viewer:")
        print("  1. Look at the LEFT SIDEBAR (entity tree)")
        print("  2. Expand 'world' > 'camera_highres'")
        print("  3. You should see: bgr/ and depth/ folders")
        print("  4. Each contains: 'original' and 'edited'")
        print()
        print("  5. The main view should show the 2x2 grid automatically")
        print("  6. If not, try clicking 'Reset Blueprint' in the top menu")
        print("=" * 70 + "\n")

    # Interactive mode: frame-by-frame editing with keyboard controls
    if args.interactive and result is not None:
        print("\n" + "=" * 70)
        print("STARTING INTERACTIVE MODE")
        print("=" * 70)
        print("\nIn Rerun viewer, you should see a 2x2 grid:")
        print("  • Top-left: RGB Original")
        print("  • Top-right: Depth Original")
        print("  • Bottom-left: RGB Edited (empty until you press 'e')")
        print("  • Bottom-right: Depth Edited (empty until you press 'e')")
        print("\nWhen you edit a frame, the bottom panels will show your edits!")
        print("=" * 70)

        # Extract data from result
        frame_metadata = result["frame_metadata"]

        # Filter to only high-res frames
        highres_frames = [
            frame for frame in frame_metadata if frame["high_res_exists"]
        ]

        if not highres_frames:
            print("ERROR: No high-resolution frames available for interactive mode")
            print("=" * 70 + "\n")
        else:
            # Create timestamp list and frame data dict
            timestamps = [frame["timestamp"] for frame in highres_frames]
            frame_data_dict = {
                frame["timestamp"]: frame for frame in highres_frames
            }

            # Initialize Gemini editor if API key provided
            gemini_editor = None
            if args.gemini_api_key:
                print(f"\n{'='*70}")
                print("INITIALIZING GEMINI AI EDITOR")
                print(f"{'='*70}")
                gemini_editor = get_editor(args.gemini_api_key)
                if gemini_editor:
                    print("✓ Gemini API connected!")
                    print("✓ When you press 'e', you can enter custom editing prompts")
                    print("  Examples: 'make the jacket red', 'add sunglasses', etc.")
                    if gemini_editor.depth_pipe:
                        print("✓ Depth Anything V2 loaded - depth maps will be generated")
                    else:
                        print("⚠ Depth model not available - will use original depth")
                    print(f"{'='*70}\n")
                else:
                    print("✗ Failed to initialize Gemini editor")
                    print(f"{'='*70}\n")
            else:
                print(f"\n{'='*70}")
                print("MOCK EDITING MODE")
                print(f"{'='*70}")
                print("ℹ️  No Gemini API key provided")
                print("   When you press 'e', images will use mock edits:")
                print("   - Color inversion")
                print("   - Red border + 'EDITED' watermark")
                print("")
                print("   To use AI editing, run with: --gemini")
                print(f"{'='*70}\n")

            # Create frame editor
            editor = FrameEditor(timestamps, frame_data_dict)

            # Define edit callback that logs to Rerun
            def edit_frame_callback(timestamp: str, frame_data: dict) -> None:
                """Callback to edit and log frame when user presses 'E'."""
                print(f"\n\n{'='*70}")
                print(f"EDITING FRAME {timestamp}")
                print(f"{'='*70}")

                try:
                    # Get the original high-res BGR image
                    if "highres_bgr" not in frame_data:
                        print(f"ERROR: 'highres_bgr' not found in frame_data!")
                        return

                    original_bgr = frame_data["highres_bgr"]

                    # Get editing prompt from user
                    if gemini_editor:
                        print("\nEnter editing instructions (or press Enter to use mock edit):")
                        prompt = input("Prompt: ").strip()

                        if prompt:
                            print(f"\n🤖 Calling Gemini API with prompt: '{prompt}'")
                            edited_bgr, status = gemini_editor.edit_image(original_bgr, prompt)

                            if edited_bgr is None:
                                print(f"✗ Gemini editing failed: {status}")
                                print("Falling back to mock edit...")
                                edited_bgr = apply_obvious_edit(original_bgr, timestamp)
                            else:
                                print(f"✓ {status}")
                        else:
                            print("No prompt provided, using mock edit...")
                            edited_bgr = apply_obvious_edit(original_bgr, timestamp)
                    else:
                        print("Using mock edit (no Gemini API key provided)")
                        edited_bgr = apply_obvious_edit(original_bgr, timestamp)

                except Exception as e:
                    print(f"ERROR during editing: {e}")
                    import traceback
                    traceback.print_exc()
                    return

                # Generate depth map from edited image (if Gemini editor available)
                try:
                    # Use the same timeline as original frames for proper alignment
                    rr.set_time("time high resolution", duration=float(timestamp))

                    # Log edited BGR
                    rr.log(
                        f"{HIGHRES_ENTITY_PATH}/bgr/edited",
                        rr.Image(edited_bgr, color_model="BGR").compress(jpeg_quality=75)
                    )
                    print(f"✓ Logged edited RGB image at timestamp {timestamp}")

                    # Generate and log depth map
                    if gemini_editor and gemini_editor.depth_pipe:
                        print("🗺️  Generating depth map from edited image...")
                        depth_map, depth_status = gemini_editor.generate_depth_map(edited_bgr)

                        if depth_map is not None:
                            rr.log(
                                f"{HIGHRES_ENTITY_PATH}/depth/edited",
                                rr.Image(depth_map, color_model="BGR").compress(jpeg_quality=75)
                            )
                            print(f"✓ {depth_status}")
                        else:
                            print(f"✗ Depth generation failed: {depth_status}")
                    else:
                        # Fallback: use original depth with colormap if available
                        if "highres_depth" in frame_data:
                            print("Using original depth (colorized)...")
                            original_depth = frame_data["highres_depth"]
                            depth_normalized = cv2.normalize(
                                original_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                            )
                            depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                            rr.log(
                                f"{HIGHRES_ENTITY_PATH}/depth/edited",
                                rr.Image(depth_color, color_model="BGR").compress(jpeg_quality=75)
                            )
                            print(f"✓ Logged colorized depth map")

                except Exception as e:
                    print(f"ERROR logging to Rerun: {e}")
                    import traceback
                    traceback.print_exc()
                    return

                print(f"\n{'='*70}")
                print(f"✓ FRAME {timestamp} EDITED AND LOGGED")
                print(f"{'='*70}")
                print(f"  → RGB: {HIGHRES_ENTITY_PATH}/bgr/edited")
                print(f"  → Depth: {HIGHRES_ENTITY_PATH}/depth/edited")
                print(f"  → Timestamp: {timestamp}s")
                print(f"  → Scrub timeline in Rerun to see the edit!")
                print(f"{'='*70}\n")

            # Set the callback
            editor.edit_callback = edit_frame_callback

            # Start interactive mode
            start_interactive_mode(editor)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
