# Penn x Anthropic Hackathon Project

Physical AI systems struggle with edge cases and rare events

Manually collecting these data are hard, so we use nano-banana to edit images and then add depthmaps via DepthAnything V2, and store the trajectories and modifications in Rerun.


## Install
`pip install -e '.[interactive, gemini]`


## Usage

Launch with `python -m mm_editor --include-highres --interactive --gemini`

Scrub through with the arrow keys till you hit the frames you want to enhance.

Press e, input prompt and wait.

Done