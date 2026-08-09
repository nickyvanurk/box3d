#!/usr/bin/env bash
# Build the Box3D WASM binding.
#
# Emits three files into $1 (default: this directory's `dist/`):
#
#   box3d.wasm   the engine
#   box3d.mjs    emscripten's loader glue
#   build.json   which source produced them
#
# Consumers commit those artifacts rather than running this script, so that a
# checkout builds with nothing but npm. `build.json` is what makes that
# honest — it records the fork commit and compiler version behind the binary,
# so a shipped artifact can always be traced back to source.
#
# Requires emsdk at $EMSDK_DIR (default ~/dev/emsdk).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$HERE/dist}"
EMSDK_DIR="${EMSDK_DIR:-$HOME/dev/emsdk}"

# shellcheck disable=SC1091
source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1

mkdir -p "$OUT"

EXPORTS='[
	"_b3w_create_world","_b3w_destroy_world","_b3w_step",
	"_b3w_create_body","_b3w_add_box_shape","_b3w_body_count","_b3w_is_awake",
	"_b3w_transforms_ptr","_b3w_read_transforms",
	"_malloc","_free"
]'

# NDEBUG disables Box3D's assert-heavy validation. A build with BOX3D_VALIDATE
# active measures the validator, not the solver.
emcc \
	-O3 -flto \
	-msimd128 -msse2 \
	-DNDEBUG \
	-std=gnu17 \
	-I"$ROOT/include" \
	-I"$ROOT/src" \
	"$ROOT"/src/*.c \
	"$HERE/shim.c" \
	-o "$OUT/box3d.mjs" \
	-s MODULARIZE=1 \
	-s EXPORT_ES6=1 \
	-s ENVIRONMENT=web,worker,node \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s INITIAL_MEMORY=64MB \
	-s EXPORTED_FUNCTIONS="$(echo "$EXPORTS" | tr -d '\n\t ')" \
	-s EXPORTED_RUNTIME_METHODS='["HEAPF32","HEAPU8"]' \
	-s SINGLE_FILE=0 \
	-s FILESYSTEM=0

cat > "$OUT/build.json" <<EOF
{
	"source": "$(git -C "$ROOT" remote get-url origin 2>/dev/null || echo unknown)",
	"commit": "$(git -C "$ROOT" rev-parse HEAD)",
	"dirty": $(test -n "$(git -C "$ROOT" status --porcelain)" && echo true || echo false),
	"emcc": "$(emcc --version | head -1 | sed 's/.*replacement + linker emulating GNU ld) //;s/ .*//')"
}
EOF

echo "built into $OUT:"
ls -la "$OUT"
