#!/usr/bin/env bash
# Run the Omega v0.1.0 closed loop: sidecar (Python, world + model)
# + bridge (Rust, kernel cognition). Usage: ./run_v010.sh [steps]
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$DIR/run"
STEPS="${1:-8}"
mkdir -p "$RUN"
rm -f "$RUN/action.json" "$RUN/outcome.json" "$RUN/observation.json"
python3 -u "$DIR/omega_kernel_bridge.py" --dir "$RUN" --steps "$STEPS" > "$RUN/sidecar.log" 2>&1 &
SIDECAR_PID=$!
trap 'kill $SIDECAR_PID 2>/dev/null || true' EXIT
"$DIR/target/release/omega-world-bridge" --dir "$RUN" --steps "$STEPS" --config "$DIR/omega.toml" 2>&1 | tee "$RUN/bridge.log"
wait $SIDECAR_PID 2>/dev/null || true
