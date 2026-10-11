#!/usr/bin/env bash
# Build the Box3D WASM binding.
#
# Emits four files into $1 (default: this directory's `dist/`):
#
#   box3d.wasm          the engine, wasm SIMD
#   box3d-scalar.wasm   the same engine without SIMD, for browsers that refuse
#                       the v128 opcodes; the consumer picks one at load
#   box3d.mjs           emscripten's loader glue, shared by both
#   build.json          which source produced them
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
	"_b3w_create_body","_b3w_add_box_shape","_b3w_body_defer_mass","_b3w_body_count","_b3w_live_body_count","_b3w_is_awake","_b3w_body_pair_impulse",
	"_b3w_transforms_ptr","_b3w_read_transforms",
	"_b3w_body_transform","_b3w_set_position","_b3w_set_rotation",
	"_b3w_linear_velocity","_b3w_angular_velocity","_b3w_body_center",
	"_b3w_set_linear_velocity","_b3w_set_angular_velocity","_b3w_set_target_transform","_b3w_set_damping","_b3w_vehicle_set_velocity","_b3w_vehicle_release_anchors","_b3w_set_rotation_lock","_b3w_set_shape_group","_b3w_set_body_exclusion","_b3w_set_body_placed","_b3w_body_set_bullet","_b3w_body_set_bullet","_b3w_body_contact_impulse","_b3w_body_allow_fast_rotation",
	"_b3w_apply_impulse","_b3w_apply_force","_b3w_apply_torque",
	"_b3w_mass","_b3w_principal_inertia",
	"_b3w_body_type","_b3w_set_body_type","_b3w_set_gravity_scale",
	"_b3w_wake","_b3w_keep_awake","_b3w_is_valid","_b3w_add_capsule_shape",
	"_b3w_add_height_field","_b3w_cast_ray_y","_b3w_alloc_floats",
	"_b3w_add_revolute_joint","_b3w_add_prismatic_joint","_b3w_add_weld_joint","_b3w_add_ball_joint","_b3w_joint_force","_b3w_add_motor_joint","_b3w_add_mesh_shape",
	"_b3w_add_distance_joint","_b3w_set_distance_limits","_b3w_set_distance_motor","_b3w_distance_length",
	"_b3w_set_revolute_motor","_b3w_set_revolute_limits","_b3w_set_joint_constraint_tuning","_b3w_wake_joint_bodies",
	"_b3w_set_prismatic_motor","_b3w_set_prismatic_limits","_b3w_cast_sphere",
	"_b3w_add_sphere_shape","_b3w_add_cylinder_shape","_b3w_add_hull_shape",
	"_b3w_destroy_body","_b3w_set_gravity","_b3w_set_contact_tuning","_b3w_set_maximum_linear_speed","_b3w_shape_count","_b3w_cast_ray_detailed",
	"_b3w_set_revolute_servo","_b3w_revolute_angle","_b3w_prismatic_translation",
	"_b3w_destroy_shape","_b3w_shape_is_valid","_b3w_destroy_joint","_b3w_set_shape_filter",
	"_b3w_joint_count","_b3w_live_joint_count",
	"_b3w_cast_ray_filtered","_b3w_cast_sphere_filtered","_b3w_set_body_mass","_b3w_set_sleep_threshold","_b3w_body_contacts","_b3w_body_max_approach_speed","_b3w_body_aabb","_b3w_body_inertia_tensor","_b3w_body_shape_states","_b3w_debug_draw",
	"_b3w_create_vehicle","_b3w_step_vehicle","_b3w_vehicle_destroy","_b3w_vehicle_add_wheel",
	"_b3w_vehicle_read_wheels","_b3w_vehicle_wheel_count","_b3w_vehicle_wheel_capacity","_b3w_vehicle_set_owner","_b3w_body_set_owner","_b3w_body_set_bullet","_b3w_set_detect_only",
	"_b3w_vehicle_set_up_axis","_b3w_vehicle_up_axis","_b3w_vehicle_wheel_direction",
	"_b3w_vehicle_set_engine_force","_b3w_vehicle_set_brake","_b3w_vehicle_set_steering",
	"_b3w_vehicle_set_side_friction","_b3w_vehicle_set_friction_slip",
	"_b3w_vehicle_set_suspension_stiffness","_b3w_vehicle_set_suspension_compression",
	"_b3w_vehicle_set_suspension_relaxation","_b3w_vehicle_set_suspension_rest_length",
	"_b3w_vehicle_set_max_suspension_travel","_b3w_vehicle_set_max_suspension_force",
	"_b3w_vehicle_set_wheel_inertia","_b3w_vehicle_set_wheel_mass",
	"_b3w_vehicle_debug_draw","_b3w_vehicle_awake_wheels","_b3w_vehicle_wheel_speeds","_b3w_vehicle_wheel_sleep","_b3w_body_sleep_state",
	"_b3w_vehicle_place_wheels",
	"_b3w_vehicle_set_anti_squat",
	"_malloc","_free"
]'

# NDEBUG disables Box3D's assert-heavy validation. A build with BOX3D_VALIDATE
# active measures the validator, not the solver.
#
# $1 is the SIMD selection, $2 the directory to emit into. Scalar is NOT just
# dropping -msimd128: without BOX3D_DISABLE_SIMD, core.h picks the SSE2 path on
# wasm unconditionally. The define keeps B3_SIMD_WIDTH at 4, so the data layout
# and every export are identical — which is why one glue file serves both.
build() {
emcc \
	-O3 -flto \
	$1 \
	-DNDEBUG \
	-std=gnu17 \
	-I"$ROOT/include" \
	-I"$ROOT/src" \
	"$ROOT"/src/*.c \
	"$HERE/shim.c" \
	"$HERE/vehicle.c" \
	-o "$2/box3d.mjs" \
	-s MODULARIZE=1 \
	-s EXPORT_ES6=1 \
	-s ENVIRONMENT=web,worker,node \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s INITIAL_MEMORY=64MB \
	-s EXPORTED_FUNCTIONS="$(echo "$EXPORTS" | tr -d '\n\t ')" \
	-s EXPORTED_RUNTIME_METHODS='["HEAPF32","HEAP32","HEAPU8"]' \
	-s SINGLE_FILE=0 \
	-s FILESYSTEM=0
}

build "-msimd128 -msse2" "$OUT"

SCALAR="$(mktemp -d)"
trap 'rm -rf "$SCALAR"' EXIT
build "-DBOX3D_DISABLE_SIMD" "$SCALAR"
# The consumer loads either binary through the one glue file, so the two builds
# must have emitted the same glue — if they ever differ, refuse rather than ship
# a scalar engine behind a loader written for the other one.
if ! cmp -s "$OUT/box3d.mjs" "$SCALAR/box3d.mjs"; then
	echo "build.sh: the scalar build emitted different glue from the SIMD build" >&2
	exit 1
fi
cp "$SCALAR/box3d.wasm" "$OUT/box3d-scalar.wasm"

# Provenance or nothing. Built outside a checkout — from a `git archive` tarball,
# say — `rev-parse` writes an empty commit and the `status` test sees no changes,
# so the file claims `"dirty": false` while recording no source at all. That is
# worse than refusing: it is the one field a reader trusts to trace a binary back.
# `BUILD_COMMIT` is the escape hatch, and it exists for one real job: verifying a
# shipped binary by building the same revision from a `git archive` tarball and
# comparing hashes. That tarball has no `.git`, so without this the check the
# guard protects would be the check the guard prevents.
COMMIT="${BUILD_COMMIT:-}"
if [ -z "$COMMIT" ] && ! COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"; then
	COMMIT=""
fi
if [ -z "$COMMIT" ]; then
	echo "build.sh: $ROOT is not a git checkout and BUILD_COMMIT is unset, so the" >&2
	echo "          artifact cannot record what produced it. Pass BUILD_COMMIT=<sha>." >&2
	exit 1
fi

cat > "$OUT/build.json" <<EOF
{
	"source": "$(git -C "$ROOT" remote get-url origin 2>/dev/null || echo unknown)",
	"commit": "$COMMIT",
	"dirty": $(test -n "${BUILD_COMMIT:-}" && echo false || { test -n "$(git -C "$ROOT" status --porcelain)" && echo true || echo false; }),
	"emcc": "$(emcc --version | head -1 | sed 's/.*replacement + linker emulating GNU ld) //;s/ .*//')"
}
EOF

echo "built into $OUT:"
ls -la "$OUT"
