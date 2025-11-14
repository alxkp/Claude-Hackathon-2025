"""
Gemini Image Editor - Core functionality for ARKit integration
Extracted from gradio_image_augmentor.py for CLI use
"""

import io
from typing import Tuple

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Depth map generation disabled.")


class GeminiImageEditor:
    """Image editor using Gemini API and Depth Anything."""

    def __init__(self, api_key: str):
        """
        Initialize the editor with Gemini API key.

        Args:
            api_key: Google Gemini API key
        """
        self.client = genai.Client(api_key=api_key)
        self.depth_pipe = None

        # Initialize Depth Anything pipeline if available
        if HAS_TRANSFORMERS:
            try:
                print("Loading Depth Anything model...")
                self.depth_pipe = pipeline(
                    task="depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf"
                )
                print("✓ Depth model loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load depth model: {e}")

    def edit_image(
        self,
        image: np.ndarray,
        prompt: str
    ) -> Tuple[np.ndarray | None, str]:
        """
        Edit an image using Gemini's image generation model.

        Args:
            image: Input image as numpy array (BGR or RGB)
            prompt: Editing instructions

        Returns:
            Tuple of (edited_image_array, status_message)
        """
        try:
            if image is None:
                return None, "No image provided"

            if not prompt or prompt.strip() == "":
                return None, "No editing prompt provided"

            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR (from OpenCV)
                pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
            else:
                pil_image = Image.fromarray(image)

            # Convert PIL Image to bytes for Gemini
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Create properly formatted content for Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(
                            data=img_byte_arr,
                            mime_type="image/png"
                        ),
                    ],
                ),
            ]

            # Configure for image generation
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    image_size="1K",
                ),
            )

            # Generate edited image using Gemini with streaming
            edited_image = None
            response_text = ""

            for chunk in self.client.models.generate_content_stream(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue

                # Check for image data
                for part in chunk.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        inline_data = part.inline_data
                        data_buffer = inline_data.data

                        # Convert bytes to PIL Image
                        edited_image = Image.open(io.BytesIO(data_buffer))
                        # Ensure image is in RGB mode
                        if edited_image.mode != 'RGB':
                            edited_image = edited_image.convert('RGB')
                        break

                # Collect text response
                if hasattr(chunk, 'text') and chunk.text:
                    response_text += chunk.text

            if edited_image:
                # Convert back to numpy array (BGR for OpenCV)
                edited_array = np.array(edited_image)
                edited_array = edited_array[:, :, ::-1]  # RGB to BGR
                width, height = edited_image.size
                return edited_array, f"✓ Image edited successfully! Size: {width}x{height}"
            elif response_text:
                return None, f"No image generated. Response: {response_text}"
            else:
                return None, "No image was generated. Try a different prompt."

        except Exception as e:
            return None, f"Error: {str(e)}"

    def generate_depth_map(self, image: np.ndarray) -> Tuple[np.ndarray | None, str]:
        """
        Generate a depth map from an image using Depth Anything V2.

        Args:
            image: Input image as numpy array (BGR or RGB)

        Returns:
            Tuple of (depth_map_array, status_message)
        """
        try:
            if image is None:
                return None, "No image available"

            if self.depth_pipe is None:
                return None, "Depth model not available. Install transformers and depth-anything."

            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR (from OpenCV)
                pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
            else:
                pil_image = Image.fromarray(image)

            # Generate depth map
            depth_result = self.depth_pipe(pil_image)
            depth_map = depth_result["depth"]

            # Convert to RGB if needed
            if depth_map.mode != 'RGB':
                depth_map = depth_map.convert('RGB')

            # Convert back to numpy array (BGR for OpenCV)
            depth_array = np.array(depth_map)
            depth_array = depth_array[:, :, ::-1]  # RGB to BGR

            width, height = depth_map.size
            return depth_array, f"✓ Depth map generated! Size: {width}x{height}"

        except Exception as e:
            return None, f"Error generating depth map: {str(e)}"


# Global editor instance (initialized when API key is provided)
_editor_instance: GeminiImageEditor | None = None


def get_editor(api_key: str | None = None) -> GeminiImageEditor | None:
    """
    Get or create the global editor instance.

    Args:
        api_key: Gemini API key (required on first call)

    Returns:
        GeminiImageEditor instance or None if no API key
    """
    global _editor_instance

    if _editor_instance is None and api_key:
        _editor_instance = GeminiImageEditor(api_key)

    return _editor_instance
