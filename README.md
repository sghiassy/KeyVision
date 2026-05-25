# KeyVision

A Python server that identifies household keys from photos. Enroll keys by uploading images, then recognize unknown keys by pointing a camera at them.

Uses OpenCV for key segmentation, DINOv2 (ViT-B/14) for image embeddings, and cosine similarity for matching — no training data required.

**[iOS App →](docs/ios.md)** — native iPhone app with real-time on-device recognition (no network round-trip).

---

## Installing Python 3.9

Use [pyenv](https://github.com/pyenv/pyenv) to manage Python versions.

**Install pyenv (macOS):**

```bash
brew install pyenv
```

Add the following to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Restart your shell, then install Python 3.9:

```bash
pyenv install 3.9.19
```

---

## Dev Setup

**Requirements:** Python 3.9 (see above)

```bash
# Clone the repo
git clone <repo-url>
cd keyvision

# Create and activate a virtual environment
python3.9 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The first time the server handles an image request it downloads the DINOv2 model weights (~330MB) from HuggingFace. This happens once and is cached automatically.

---

## Running the Server

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000`. Interactive API docs are available at `http://localhost:8000/docs`.

---

## API Usage

### Create a key

```bash
curl -X POST http://localhost:8000/keys \
  -H "Content-Type: application/json" \
  -d '{"label": "Front Door", "notes": "deadbolt key"}'
```

```json
{"key_id": "a1b2c3d4-..."}
```

### Enroll an image

Upload a photo of the key against a plain background. Enroll 3–5 images from different angles for best results.

```bash
curl -X POST http://localhost:8000/keys/0edae8a0-f69a-4fb3-becf-07c8aa47fc42/images \
  -F "image=@front_door_key.jpg"
```

```json
{"image_id": "e5f6g7h8-...", "segmentation_ok": true}
```

### Recognize a key

```bash
curl -X POST http://localhost:8000/recognize \
  -F "image=@unknown_key.jpg"
```

```json
{
  "matches": [
    {"key_id": "a1b2c3d4-...", "label": "Front Door", "similarity": 0.94, "confidence": "high"},
    {"key_id": "z9y8x7w6-...", "label": "Mailbox",    "similarity": 0.61, "confidence": "no_match"}
  ],
  "segmentation_ok": true,
  "top_confidence": "high"
}
```

Confidence levels: `high` (≥ 0.85), `maybe` (0.65–0.85), `no_match` (< 0.65).

### List all keys

```bash
curl http://localhost:8000/keys
```

### Delete a key

```bash
curl -X DELETE http://localhost:8000/keys/<key_id>
```

---

## Tips for Good Recognition

- Photograph the key on a **plain white or light background**
- Keep the key in **sharp focus** — blurry images are rejected
- The key should fill a reasonable portion of the frame (5–80% of the image area)
- Enroll multiple images (3–5) from slightly different angles

---

## Running Tests

```bash
# Fast tests only (no model download)
pytest -m "not slow"

# Full suite including end-to-end tests (requires ~330MB model download)
pytest
```


---

## Deploying to Production

The app runs on a Hostinger VPS using Docker Swarm + Caddy at `https://keyvision.cloud`.

**Prerequisites:**
- Docker Desktop running locally
- Docker context `hostinger` configured: `docker context create hostinger --docker "host=ssh://root@72.61.65.125"`
- Logged in to Docker Hub: `docker login`

**Deploy:**

```bash
# 1. Build image for linux/amd64 (required — Mac is arm64)
npm run docker:build:image   # ~10–20 min first time

# 2. Push to Docker Hub
npm run docker:push

# 3. Deploy to VPS
npm run docker:deploy

# 4. Reload Caddy
npm run caddy:reload
```

**Check status:**

```bash
docker context use hostinger
docker stack services keyvision          # should show 1/1 replicas
docker service logs -f keyvision_web     # watch for "Application startup complete"
```

The first cold start after a fresh deploy takes 30–90 seconds while DINOv2 downloads into the persistent volume. Subsequent restarts are fast.

**All npm deploy scripts:**

| Script                       | Description                               |
| ---------------------------- | ----------------------------------------- |
| `npm run docker:build:image` | Build production image for linux/amd64    |
| `npm run docker:push`        | Push image to Docker Hub                  |
| `npm run docker:deploy`      | Deploy stack to VPS                       |
| `npm run caddy:reload`       | Reload Caddy after config changes         |
| `npm run docker:local:build` | Build image for local architecture        |
| `npm run start`              | Run Docker container locally on port 8000 |
| `npm run docker:local:stop`  | Stop local Docker container               |
| `npm run dev`                | Run uvicorn dev server with hot reload    |
