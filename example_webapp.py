#!/usr/bin/env python3
"""
Example AI Image Editing Web App Server

This is a simple Flask server that demonstrates how to create a web app
that integrates with the ARKit Scenes Rerun example.

Usage:
    python example_webapp.py

Then run the ARKit example with:
    python -m arkit_scenes --include-highres --enable-webapp
"""

import base64
import io
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to numpy image."""
    image_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode numpy image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def apply_ai_editing(image: np.ndarray, is_depth: bool = False) -> np.ndarray:
    """
    Apply AI image editing.

    Replace this function with your actual AI image editing logic.
    This could call Stable Diffusion, ControlNet, SAM2, or any other model.
    """
    if is_depth:
        # For depth images: apply colormap
        if len(image.shape) == 2 or image.shape[2] == 1:
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            edited = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        else:
            edited = image
    else:
        # For RGB images: apply artistic filter
        # You could replace this with:
        # - Stable Diffusion img2img
        # - Style transfer
        # - Object segmentation with SAM2
        # - Depth-guided editing with ControlNet
        # - etc.
        edited = cv2.bilateralFilter(image, 9, 75, 75)

        # Optional: add some edge enhancement
        gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edited = cv2.addWeighted(edited, 0.8, edges_colored, 0.2, 0)

    return edited


@app.route('/edit', methods=['POST'])
def edit_image() -> Any:
    """
    API endpoint for image editing.

    Expected JSON body:
    {
        "image": "base64_encoded_image",
        "is_depth": false
    }

    Returns:
    {
        "edited_image": "base64_encoded_edited_image",
        "status": "success"
    }
    """
    try:
        data = request.json

        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the input image
        base64_image = data['image']
        is_depth = data.get('is_depth', False)

        image = decode_base64_image(base64_image)

        # Apply AI editing
        edited_image = apply_ai_editing(image, is_depth=is_depth)

        # Encode the result
        edited_base64 = encode_image_to_base64(edited_image)

        return jsonify({
            "edited_image": edited_base64,
            "status": "success",
            "is_depth": is_depth,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check() -> Any:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "AI Image Editing API"})


@app.route('/', methods=['GET'])
def home() -> str:
    """Home page with API documentation."""
    return """
    <h1>AI Image Editing Web App</h1>
    <p>This server provides an API for AI-powered image editing.</p>

    <h2>Endpoints:</h2>
    <ul>
        <li><code>POST /edit</code> - Edit an image</li>
        <li><code>GET /health</code> - Health check</li>
    </ul>

    <h2>Usage Example:</h2>
    <pre>
curl -X POST http://localhost:8000/edit \\
  -H "Content-Type: application/json" \\
  -d '{"image": "base64_encoded_image_string", "is_depth": false}'
    </pre>

    <h2>Integration with ARKit Scenes:</h2>
    <pre>
python -m arkit_scenes --include-highres --enable-webapp --api-url http://localhost:8000/edit
    </pre>
    """


if __name__ == '__main__':
    print("=" * 60)
    print("AI Image Editing Web App Server")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("\nTo use with ARKit Scenes, run:")
    print("  python -m arkit_scenes --include-highres --enable-webapp")
    print("=" * 60)

    app.run(host='0.0.0.0', port=8000, debug=True)
