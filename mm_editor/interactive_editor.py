"""
Interactive frame editor for Rerun visualizations.

Allows frame-by-frame editing with keyboard controls while viewing in Rerun.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

try:
    from pynput import keyboard
except ImportError:
    print("Warning: pynput not installed. Interactive mode requires: pip install pynput")
    keyboard = None


class FrameEditor:
    """
    Interactive frame editor with keyboard controls.

    Allows users to navigate frames and trigger edits in real-time.
    """

    def __init__(self, frame_timestamps: list[str], frame_data: dict):
        """
        Initialize the frame editor.

        Args:
            frame_timestamps: List of frame timestamps in order
            frame_data: Dict mapping timestamp to frame data (images, metadata)
        """
        self.timestamps = frame_timestamps
        self.frame_data = frame_data
        self.current_index = 0
        self.edited_frames = set()  # Track which frames have been edited
        self.running = True
        self.edit_callback: Callable | None = None

    def get_current_timestamp(self) -> str:
        """Get the currently selected timestamp."""
        return self.timestamps[self.current_index]

    def get_current_frame_data(self) -> dict:
        """Get the data for the current frame."""
        return self.frame_data[self.get_current_timestamp()]

    def next_frame(self) -> None:
        """Navigate to next frame."""
        if self.current_index < len(self.timestamps) - 1:
            self.current_index += 1
            self.print_status()

    def prev_frame(self) -> None:
        """Navigate to previous frame."""
        if self.current_index > 0:
            self.current_index -= 1
            self.print_status()

    def jump_to_frame(self, index: int) -> None:
        """Jump to specific frame index."""
        if 0 <= index < len(self.timestamps):
            self.current_index = index
            self.print_status()

    def edit_current_frame(self) -> None:
        """Trigger edit for current frame."""
        timestamp = self.get_current_timestamp()
        if self.edit_callback:
            self.edited_frames.add(timestamp)
            self.edit_callback(timestamp, self.get_current_frame_data())
            self.print_status()

    def reset_current_frame(self) -> None:
        """Reset current frame to original (remove from edited set)."""
        timestamp = self.get_current_timestamp()
        if timestamp in self.edited_frames:
            self.edited_frames.remove(timestamp)
            print(f"\n✓ Reset frame {timestamp} to original")
            self.print_status()

    def print_status(self) -> None:
        """Print current frame status."""
        timestamp = self.get_current_timestamp()
        frame_num = self.current_index + 1
        total = len(self.timestamps)
        status = "EDITED" if timestamp in self.edited_frames else "Original"

        print(f"\rFrame: {frame_num}/{total} | Time: {timestamp}s | Status: {status} ", end="")
        sys.stdout.flush()

    def print_help(self) -> None:
        """Print help message."""
        print("\n" + "=" * 70)
        print("KEYBOARD CONTROLS")
        print("=" * 70)
        print("  ← / A     : Previous frame")
        print("  → / D     : Next frame")
        print("  E         : Edit current frame (invert + watermark)")
        print("  R         : Reset current frame to original")
        print("  H         : Show this help")
        print("  Q / ESC   : Quit interactive mode")
        print("=" * 70)
        self.print_status()

    def print_help_stdin(self) -> None:
        """Print help message for stdin mode."""
        print("\n" + "=" * 70)
        print("TERMINAL COMMANDS (type and press Enter)")
        print("=" * 70)
        print("  n / next / d / <Enter>  : Next frame")
        print("  p / prev / a            : Previous frame")
        print("  e / edit                : Edit current frame")
        print("                            (with --gemini: prompts for AI editing)")
        print("                            (without: uses mock edit)")
        print("  r / reset               : Reset current frame to original")
        print("  <number>                : Jump to frame number (e.g., '5')")
        print("  h / help / ?            : Show this help")
        print("  q / quit / exit         : Quit interactive mode")
        print("=" * 70)

    def quit(self) -> None:
        """Quit interactive mode."""
        self.running = False
        print(f"\n\nEdited {len(self.edited_frames)} frame(s)")
        print("Exiting interactive mode...")


def apply_obvious_edit(image: np.ndarray, frame_id: str) -> np.ndarray:
    """
    Apply super obvious edit effects to an image.

    Effects applied:
    1. Invert colors
    2. Add thick red border
    3. Add "EDITED" watermark
    4. Add frame timestamp overlay

    Args:
        image: Input BGR image
        frame_id: Frame identifier/timestamp

    Returns:
        Edited image with obvious visual changes
    """
    # Make a copy to avoid modifying original
    edited = image.copy()
    height, width = edited.shape[:2]

    # 1. INVERT COLORS - most obvious change
    edited = 255 - edited

    # 2. ADD THICK RED BORDER (10 pixels)
    border_thickness = 10
    cv2.rectangle(
        edited,
        (0, 0),
        (width - 1, height - 1),
        (0, 0, 255),  # Red in BGR
        border_thickness
    )

    # 3. ADD "EDITED" WATERMARK - large green text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(width, height) / 400.0  # Scale based on image size
    thickness = max(2, int(font_scale * 2))

    # Calculate text size for centering
    text = "EDITED"
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale * 2, thickness)

    # Position at top center
    text_x = (width - text_width) // 2
    text_y = 80 + text_height

    # Add black background for text visibility
    cv2.rectangle(
        edited,
        (text_x - 10, text_y - text_height - 10),
        (text_x + text_width + 10, text_y + baseline + 10),
        (0, 0, 0),
        -1
    )

    # Add text
    cv2.putText(
        edited,
        text,
        (text_x, text_y),
        font,
        font_scale * 2,
        (0, 255, 0),  # Green in BGR
        thickness,
        cv2.LINE_AA
    )

    # 4. ADD FRAME TIMESTAMP - bottom right
    timestamp_text = f"Frame: {frame_id}"
    timestamp_scale = font_scale * 0.8
    timestamp_thickness = max(1, int(timestamp_scale * 2))

    (ts_width, ts_height), ts_baseline = cv2.getTextSize(
        timestamp_text, font, timestamp_scale, timestamp_thickness
    )

    ts_x = width - ts_width - 20
    ts_y = height - 20

    # Add background
    cv2.rectangle(
        edited,
        (ts_x - 5, ts_y - ts_height - 5),
        (ts_x + ts_width + 5, ts_y + ts_baseline + 5),
        (0, 0, 0),
        -1
    )

    # Add text
    cv2.putText(
        edited,
        timestamp_text,
        (ts_x, ts_y),
        font,
        timestamp_scale,
        (255, 255, 255),  # White
        timestamp_thickness,
        cv2.LINE_AA
    )

    return edited


def start_interactive_mode_stdin(editor: FrameEditor) -> None:
    """
    Start interactive mode using stdin (no special permissions needed).

    Args:
        editor: FrameEditor instance to control
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE EDITING MODE - ACTIVE (Terminal Input)")
    print("=" * 70)
    print("\nRerun viewer is open. Type commands and press Enter.")
    print("=" * 70)

    editor.print_help_stdin()
    editor.print_status()

    while editor.running:
        try:
            print("\nCommand: ", end="", flush=True)
            command = input().strip().lower()

            if command in ['n', 'next', 'd', '']:
                editor.next_frame()
            elif command in ['p', 'prev', 'a']:
                editor.prev_frame()
            elif command in ['e', 'edit']:
                editor.edit_current_frame()
            elif command in ['r', 'reset']:
                editor.reset_current_frame()
            elif command in ['h', 'help', '?']:
                editor.print_help_stdin()
            elif command in ['q', 'quit', 'exit']:
                editor.quit()
            elif command.isdigit():
                # Jump to specific frame number
                frame_num = int(command) - 1  # Convert to 0-indexed
                editor.jump_to_frame(frame_num)
            else:
                print(f"Unknown command: '{command}'. Type 'h' for help.")

        except (KeyboardInterrupt, EOFError):
            editor.quit()

    print("\nInteractive mode ended.")


def start_interactive_mode(editor: FrameEditor) -> None:
    """
    Start interactive keyboard listener mode.

    Args:
        editor: FrameEditor instance to control
    """
    if keyboard is None:
        print("ERROR: pynput not installed. Install with: pip install pynput")
        print("Falling back to terminal input mode...")
        start_interactive_mode_stdin(editor)
        return

    print("\n" + "=" * 70)
    print("INTERACTIVE EDITING MODE - ACTIVE")
    print("=" * 70)
    print("\nRerun viewer is open. Navigate frames and press 'E' to edit.")
    print("Press 'H' for help, 'Q' to quit.")
    print("=" * 70)

    editor.print_status()

    def on_press(key):
        """Handle key press events."""
        try:
            if hasattr(key, 'char') and key.char:
                # Letter keys
                if key.char.lower() == 'a':
                    editor.prev_frame()
                elif key.char.lower() == 'd':
                    editor.next_frame()
                elif key.char.lower() == 'e':
                    editor.edit_current_frame()
                elif key.char.lower() == 'r':
                    editor.reset_current_frame()
                elif key.char.lower() == 'h':
                    editor.print_help()
                elif key.char.lower() == 'q':
                    editor.quit()
                    return False  # Stop listener
            else:
                # Special keys
                if key == keyboard.Key.left:
                    editor.prev_frame()
                elif key == keyboard.Key.right:
                    editor.next_frame()
                elif key == keyboard.Key.esc:
                    editor.quit()
                    return False  # Stop listener

        except AttributeError:
            pass

    # Try to start keyboard listener
    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        # Keep main thread alive until quit
        try:
            while editor.running:
                threading.Event().wait(0.1)
        except KeyboardInterrupt:
            editor.quit()

        listener.stop()
        print("\nInteractive mode ended.")

    except Exception as e:
        # If keyboard listener fails (permissions issue), fall back to stdin
        print("\n" + "=" * 70)
        print("KEYBOARD MONITORING NOT AVAILABLE")
        print("=" * 70)
        print(f"Error: {e}")
        print("\nOn macOS, you may need to grant Accessibility permissions:")
        print("  1. System Settings → Privacy & Security → Accessibility")
        print("  2. Add Terminal (or Python) to allowed apps")
        print("\nFalling back to terminal input mode...")
        print("=" * 70)

        start_interactive_mode_stdin(editor)
