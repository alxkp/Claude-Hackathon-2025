#!/bin/bash

# Test script for ARKit Scenes + AI Web App Integration
# This script demonstrates the web app integration feature

echo "=========================================="
echo "ARKit Scenes + AI Web App Integration Test"
echo "=========================================="
echo ""

# Check if running from the correct directory
if [ ! -d "arkit_scenes" ]; then
    echo "Error: Please run this script from the cbc_hack directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "This will run the ARKit Scenes example with AI web app integration enabled."
echo ""
echo "What you should see in Rerun:"
echo "  • 4 tabs: 'BGR Original', 'BGR Edited', 'Depth Original', 'Depth Edited'"
echo "  • Original images on the left tabs"
echo "  • AI-edited images (artistic filter) on the right tabs"
echo ""
echo "The mock AI editing applies:"
echo "  • Artistic bilateral filter to RGB images"
echo "  • Turbo colormap to depth images"
echo ""
echo "Press Enter to continue, or Ctrl+C to cancel..."
read

echo ""
echo "Running: python -m arkit_scenes --include-highres --enable-webapp"
echo ""

python -m arkit_scenes --include-highres --enable-webapp

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
