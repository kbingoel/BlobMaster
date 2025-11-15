# BlobMaster Backend

AI-powered Blob card game backend server built with Bun and TypeScript.

## Overview

The BlobMaster backend provides:
- **ONNX Model Inference**: AI player using trained neural network
- **Game Management**: REST API for creating and playing games
- **Real-time Updates**: WebSocket connections for live game state
- **Database**: SQLite for game history and statistics

## Prerequisites

- **Bun**: v1.0+ ([installation guide](https://bun.sh/docs/installation))
- **Node.js**: v18+ (for ONNX Runtime dependencies)
- **ONNX Model**: Trained model file at `../models/best_model.onnx` (see DVC setup in root README)

## Getting Started

### 1. Install Dependencies

```bash
cd backend
bun install
```

This installs:
- `onnxruntime-node` - ONNX model inference
- `ws` - WebSocket support
- `better-sqlite3` - SQLite database
- TypeScript and type definitions

### 2. Start Development Server

```bash
bun run dev
```

The server starts on http://localhost:3000 with hot reload enabled.

### 3. Verify Server is Running

Test the health check endpoint:

```bash
curl http://localhost:3000/health
```

Expected response:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2025-11-15T11:10:38.618Z"
}
```

## Development

### Project Structure

```
backend/
├── src/
│   ├── api/           # REST API routes (TODO: Phase 6)
│   ├── db/            # SQLite queries (TODO: Phase 6)
│   ├── game/          # Game engine TypeScript port (TODO: Phase 6)
│   ├── inference/     # ONNX model loading (TODO: Phase 6)
│   ├── ml/            # State encoder, MCTS (TODO: Phase 6)
│   └── server.ts      # Main entry point ✅
├── test/              # Unit tests (TODO: Phase 6)
├── package.json       # Dependencies ✅
└── tsconfig.json      # TypeScript config ✅
```

### Available Scripts

- `bun run dev` - Start development server with hot reload
- `bun run start` - Start production server
- `bun test` - Run test suite (once tests are implemented)
- `bun run lint` - Type check with TypeScript compiler

### Current Status: Infrastructure Ready ✅

**Completed:**
- ✅ Bun project initialized
- ✅ TypeScript configured
- ✅ Basic HTTP server running
- ✅ Health check endpoint working
- ✅ Directory structure created

**Next Steps (Phase 6):**
1. Port game engine from Python (`ml/game/blob.py` → `src/game/`)
2. Port state encoder (`ml/network/encode.py` → `src/ml/encoder.ts`)
3. Implement ONNX model loading (`src/inference/model.ts`)
4. Create REST API endpoints (`src/api/game.ts`)
5. Add WebSocket support for real-time updates
6. Set up SQLite database schema (`src/db/schema.sql`)

## API Endpoints (Planned)

### REST API

- `GET /health` - Health check ✅ (implemented)
- `POST /api/game/create` - Create new game (TODO)
- `POST /api/game/:id/join` - Join game (TODO)
- `POST /api/game/:id/bid` - Submit bid (TODO)
- `POST /api/game/:id/play` - Play card (TODO)
- `GET /api/game/:id/state` - Get game state (TODO)
- `GET /api/game/:id/history` - Get move history (TODO)

### WebSocket

- `WS /ws/game/:id` - Real-time game updates (TODO)

## Architecture Notes

### ONNX Inference

The backend loads the trained PyTorch model exported as ONNX (`models/best_model.onnx`) and uses ONNX Runtime for inference. This allows:
- Cross-platform deployment (Windows, Linux, macOS)
- GPU acceleration (Intel iGPU via OpenVINO, NVIDIA via CUDA)
- Faster inference than PyTorch (~2-5x speedup)

### Game Engine Port

The Python game engine (`ml/game/blob.py`) will be ported to TypeScript to:
- Validate moves server-side (prevent cheating)
- Run full game simulation for AI play
- Ensure consistency between training and production

### State Encoding

The state encoder (`ml/network/encode.py`) will be ported to generate the same 256-dim tensor input for the neural network. Critical for ONNX inference to work correctly.

## Troubleshooting

### "Module not found" errors

Make sure you're running commands from the `backend/` directory:
```bash
cd backend
bun run dev
```

### ONNX model not found

The trained model needs to be downloaded from DVC:
```bash
# From repository root
dvc pull models/best_model.onnx
```

### Port already in use

If port 3000 is already taken, set a different port:
```bash
PORT=3001 bun run dev
```

## Contributing

See main repository [CLAUDE.md](../CLAUDE.md) for development guidelines and architecture overview.

## License

See main repository LICENSE file.
