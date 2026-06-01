# 3D Pet Agent

`3d-pet-agent` is an experimental 3D computer pet system. The goal is to let a virtual 3D cat understand objects in a real desk or room scene, reason about their 3D positions, and respond to natural language commands such as `hide behind the keyboard` or `go to the right side of the red cup`.

The project combines open-vocabulary vision models, segmentation, depth estimation, 3D scene graphs, and a browser-based 3D pet runtime.

> Status: design and early scaffolding. The full implementation is planned in phases. See [spec.md](spec.md) for the complete software design document.

## Motivation

Most virtual pets live only inside a screen. This project explores a more grounded version: a pet that can observe the user's real environment through a camera, understand visible objects, and act in relation to those objects.

Example interactions:

```text
Go to the cup.
Hide behind the keyboard.
Do not get close to the water bottle.
Look at the object I just placed on the table.
Find a safe place between the mouse and the box.
Which object are you looking at?
```

The core idea is:

```text
camera input + language command
-> open-vocabulary object detection
-> object masks
-> depth estimation
-> 3D object states
-> scene graph
-> command grounding
-> pet behavior
```

## What This Project Is For

This project is intended for:

- AI and computer vision demos
- 3D interaction experiments
- language-grounded scene understanding
- desk-scale or room-scale virtual pet prototypes
- research reports comparing 2D-to-3D lifting with 3D open-vocabulary scene understanding

It is not intended to train a large foundation model from scratch. The project uses existing models and focuses on the engineering layers that connect perception to behavior.

## Planned Features

- Live webcam or video input
- Open-vocabulary object detection with GroundingDINO-compatible detectors
- Instance segmentation with SAM or SAM 2
- Monocular depth estimation with Depth Anything V2
- Optional RGB-D / point cloud support through Open3D
- Object tracking and temporal smoothing
- Object-centric 3D scene graph
- Natural language command parser and grounding resolver
- 3D cat runtime in a browser
- Vue 3 + Vite + TypeScript debug UI
- Python backend with FastAPI and WebSocket
- Optional OpenScene-style research backend for static 3D scenes
- Replay and evaluation modes for reproducible demos

## Technology Stack

### AI and Vision

- PyTorch
- CUDA on NVIDIA GPUs
- GroundingDINO-compatible open-vocabulary detection
- SAM / SAM 2 segmentation
- Depth Anything V2 monocular depth estimation
- OpenCV for image and video processing
- Open3D for point clouds and 3D geometry utilities
- supervision with a simple IoU tracker and optional ByteTrack-style tracking

### Backend

- Python
- FastAPI
- WebSocket / HTTP API
- Pydantic and pydantic-settings for typed schemas and configuration
- uv for Python environment and dependency management
- ruff for linting and formatting
- pytest for tests

### Frontend and 3D Runtime

- Vue 3
- Vite + Vue plugin toolchain
- TypeScript
- Three.js
- @tweenjs/tween.js for movement interpolation
- lil-gui for debug controls

## Target Hardware

Recommended development machine:

| Component | Recommendation |
|---|---|
| GPU | NVIDIA GPU with 12GB+ VRAM recommended |
| CPU | Modern 8-core desktop CPU recommended |
| RAM | 32GB recommended, 16GB minimum for lighter modes |
| Camera | Webcam, phone camera stream, or optional RGB-D camera |
| OS | Linux is the primary development target; Windows is secondary through WSL2 or native Python for lighter work |

The current design targets an NVIDIA RTX 4070-class GPU. Smaller GPUs may work with smaller models, lower image resolution, and slower perception update rates.

## Repository Layout

Planned structure:

```text
3d-pet-agent/
  README.md
  spec.md
  pyproject.toml
  requirements.txt
  main.py
  configs/
  src/
    camera_service/
    perception/
    spatial/
    tracking/
    language/
    planning/
    runtime/
    research/
    evaluation/
  frontend/
    package.json
    vite.config.ts
    tsconfig.json
    src/
  samples/
  eval/
  runs/
  tests/
```

At the moment, `spec.md` is the primary source of truth for the planned design.

## Installation

The exact dependency files will be added as the implementation progresses. The following is the intended setup flow. Linux is the primary setup path.

### Linux (Primary)

1. Install system prerequisites:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv nodejs npm
```

2. Install `uv` if it is not already available:

```bash
python3 -m pip install --user uv
```

3. Clone the repository:

```bash
git clone https://github.com/<your-user-or-org>/3D-pet-agent.git
cd 3D-pet-agent
```

4. Create the Python environment:

```bash
uv venv
source .venv/bin/activate
```

5. Install Python dependencies when `pyproject.toml` or `requirements.txt` is available:

```bash
uv sync
# or
uv pip install -r requirements.txt
```

6. Install frontend dependencies when the frontend is available:

```bash
cd frontend
npm install
cd ..
```

### Windows (Secondary)

Linux is the main development and validation environment for this project. Windows support is best treated as secondary. Windows can run the project in two ways:

- Native Windows Python + Node.js for lighter development and frontend work
- WSL2 Ubuntu for CUDA-oriented model inference, which is usually easier to keep close to Linux instructions

#### Option A: Native Windows

1. Install:

- Git for Windows
- Python 3.11 or newer
- Node.js 20 or newer
- NVIDIA driver if using CUDA inference

2. Clone the repository:

```powershell
git clone https://github.com/<your-user-or-org>/3D-pet-agent.git
cd 3D-pet-agent
```

3. Create the Python environment:

```powershell
py -m pip install uv
uv venv
.venv\Scripts\Activate.ps1
```

4. Install dependencies when available:

```powershell
uv sync
# or
uv pip install -r requirements.txt
```

5. Install frontend dependencies when available:

```powershell
cd frontend
npm install
cd ..
```

#### Option B: Windows WSL2

1. Install WSL2 with Ubuntu.
2. Install the NVIDIA driver on Windows with WSL CUDA support.
3. Open Ubuntu in WSL2.
4. Follow the Linux setup steps above.

WSL2 is the recommended Windows path for users who want to stay close to the Linux development environment and run PyTorch CUDA model inference with fewer Windows-specific dependency issues.

## Running the Project

The planned CLI modes are:

```bash
python main.py --mode sandbox
python main.py --mode snapshot --image samples/desk.jpg --command "go to the red cup"
python main.py --mode demo --camera 0 --prompts configs/desk_prompts.txt
python main.py --mode replay --video samples/desk_scene.mp4 --commands samples/commands.jsonl
python main.py --mode eval --dataset eval/desk_queries.jsonl
```

Frontend development server, once implemented:

```bash
cd frontend
npm run dev
```

Backend development server, once implemented:

```bash
uvicorn src.runtime.websocket_server:app --reload
```

## Development Workflow

Recommended checks:

```bash
ruff check .
ruff format .
pytest
```

Recommended development order:

1. 3D pet sandbox
2. Snapshot image detection and segmentation
3. Depth estimation and 3D object lifting
4. Object memory and tracking
5. Scene graph and spatial relation reasoning
6. Command parser and grounding resolver
7. Behavior planner
8. Live demo mode
9. Evaluation and report assets
10. Optional OpenScene research backend

## Configuration

The project is designed around YAML configuration files validated by Pydantic settings:

```text
configs/models.yaml
configs/thresholds.yaml
configs/runtime.yaml
```

Environment variables should use the `PET_AGENT_` prefix when applicable.

Example:

```bash
PET_AGENT_DEVICE=cuda
PET_AGENT_CAMERA_INDEX=0
```

## Limitations

- Monocular depth is approximate unless camera calibration or RGB-D input is available.
- Transparent and reflective objects can produce unstable masks or depth estimates.
- Open-vocabulary detection can miss unusual objects or ambiguous references.
- Live performance depends heavily on GPU VRAM, model size, camera resolution, and update rate.
- OpenScene-style 3D semantic querying is planned as an optional research backend, not the first live demo path.

## Roadmap

See [spec.md](spec.md) for the full phased roadmap, architecture, data contracts, testing plan, and risk analysis.

## License

License information has not been finalized yet.

## Acknowledgements

This project is designed around ideas and tools from the open-source computer vision and 3D graphics communities, including GroundingDINO, Segment Anything, SAM 2, Depth Anything V2, OpenScene, PyTorch, OpenCV, Open3D, Vue, Vite, and Three.js.
