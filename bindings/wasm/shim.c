// A flat C API over Box3D, for callers that reach it across a WASM boundary.
//
// Box3D's own API passes and returns small structs by value (b3WorldId,
// b3BodyId, b3Vec3). That is free in C and awkward from JavaScript, so this
// layer trades those for int32 handles and flat float arguments — the one shape
// every WASM host can call without a struct marshalling layer.
//
// Two rules shape everything here:
//
//  1. **One boundary crossing per step.** Per-body reads are the thing that
//     makes a WASM physics engine slow from JS. Transforms come back through a
//     single buffer the host reads directly out of the heap.
//  2. **No allocation while stepping.** Buffers grow when bodies are added,
//     never inside `b3w_step`.
//
// Game-specific behaviour that would otherwise be a JS callback belongs in this
// file, on this side of the boundary — a contact filter that compares two
// integers has no business making a host call per contact pair.

#include "box3d/box3d.h"

#include <stdlib.h>
#include <string.h>

// --- handle tables -------------------------------------------------------
//
// Box3D ids are {index, world, generation} structs. The host holds an int32
// index into these tables instead, so nothing about Box3D's id layout leaks
// across the boundary and a stale handle fails a bounds check rather than
// addressing a recycled body.

#define MAX_WORLDS 4
#define MAX_JOINTS 4096
#define MAX_HULLS 8192
#define MAX_SHAPES 65536

typedef struct World
{
	b3WorldId id;
	bool inUse;

	b3BodyId* bodies;
	int bodyCount;
	int bodyCapacity;

	b3JointId joints[MAX_JOINTS];
	int jointCount;

	// Heap-owned hulls, released with the world — see RetainHull.
	b3HullData* hulls[MAX_HULLS];
	int hullCount;

	// Shapes are handed back as handles so callers can destroy or re-filter one
	// without holding a Box3D id. Slots are never reused: a stale handle then
	// fails a validity check rather than addressing someone else's collider.
	b3ShapeId shapes[MAX_SHAPES];
	int shapeCount;

	// pos(3) + quat(4) per body, interleaved, in body-handle order.
	float* transforms;
} World;

static World g_worlds[MAX_WORLDS];

static World* GetWorld( int handle )
{
	if ( handle < 0 || handle >= MAX_WORLDS || g_worlds[handle].inUse == false )
	{
		return NULL;
	}

	return &g_worlds[handle];
}

static void ReserveBodies( World* world, int needed )
{
	if ( needed <= world->bodyCapacity )
	{
		return;
	}

	int capacity = world->bodyCapacity == 0 ? 256 : world->bodyCapacity;
	while ( capacity < needed )
	{
		capacity *= 2;
	}

	world->bodies = (b3BodyId*)realloc( world->bodies, sizeof( b3BodyId ) * capacity );
	world->transforms = (float*)realloc( world->transforms, sizeof( float ) * 7 * capacity );
	world->bodyCapacity = capacity;
}

// --- world ---------------------------------------------------------------

int b3w_create_world( float gx, float gy, float gz, int enableSleep, int enableContinuous )
{
	int handle = -1;
	for ( int i = 0; i < MAX_WORLDS; ++i )
	{
		if ( g_worlds[i].inUse == false )
		{
			handle = i;
			break;
		}
	}

	if ( handle == -1 )
	{
		return -1;
	}

	b3WorldDef def = b3DefaultWorldDef();
	def.gravity = ( b3Vec3 ){ gx, gy, gz };
	def.enableSleep = enableSleep != 0;
	def.enableContinuous = enableContinuous != 0;

	// Emscripten builds here are single-threaded. Box3D spawns no threads of its
	// own at workerCount 1, which is what we want inside a Web Worker that is
	// already the simulation's thread.
	def.workerCount = 1;

	World* world = &g_worlds[handle];
	world->id = b3CreateWorld( &def );
	world->inUse = true;
	world->bodyCount = 0;
	world->jointCount = 0;
	world->hullCount = 0;
	world->shapeCount = 0;

	ReserveBodies( world, 256 );

	return handle;
}

void b3w_destroy_world( int handle )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return;
	}

	b3DestroyWorld( world->id );

	for ( int i = 0; i < world->hullCount; ++i )
	{
		b3DestroyHull( world->hulls[i] );
	}

	free( world->bodies );
	free( world->transforms );
	memset( world, 0, sizeof( World ) );
}

void b3w_step( int handle, float timeStep, int subStepCount )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return;
	}

	b3World_Step( world->id, timeStep, subStepCount );
}

// --- bodies --------------------------------------------------------------

/**
 * Create a body.
 *
 * `lockRotations` uses Box3D's motion locks, which are exact — Rapier
 * approximates the same thing with a spring that has a compliance floor, so a
 * "locked" body there still rotates a little under load. A character capsule
 * that tips at all is one that faceplants, so this is a real improvement and
 * not just a different spelling.
 *
 * `softCcdPrediction` maps to continuous collision on the body. Box3D's CCD
 * works against height fields where Rapier's silently does not, which is the
 * root of terrain tunnelling — so this is the other place the swap is expected
 * to fix something rather than merely match.
 */
int b3w_create_body( int handle, int bodyType, float px, float py, float pz, float qx, float qy, float qz, float qw,
					 float linearDamping, float angularDamping, int lockRotations, int continuous )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return -1;
	}

	b3BodyDef def = b3DefaultBodyDef();
	def.type = (b3BodyType)bodyType;
	def.position = ( b3Pos ){ px, py, pz };
	def.rotation = ( b3Quat ){ { qx, qy, qz }, qw };
	def.linearDamping = linearDamping;
	def.angularDamping = angularDamping;
	def.isBullet = continuous != 0;

	if ( lockRotations != 0 )
	{
		def.motionLocks.angularX = true;
		def.motionLocks.angularY = true;
		def.motionLocks.angularZ = true;
	}

	ReserveBodies( world, world->bodyCount + 1 );

	int body = world->bodyCount++;
	world->bodies[body] = b3CreateBody( world->id, &def );

	// The handle rides ON the body, so a query result can name it without a
	// reverse scan of the table. Offset by one so that a body whose user data
	// was never set reads as "unknown" rather than as body zero.
	b3Body_SetUserData( world->bodies[body], (void*)(intptr_t)( body + 1 ) );

	return body;
}

/**
 * Attach a box shape.
 *
 * `group` is Box3D's collision group index: a shared NEGATIVE value means those
 * shapes never collide with each other, which is exactly the game's
 * same-structure rule (welded internals must not self-collide). Rapier needs a
 * JS callback invoked per contact pair to express that; here it is a field, so
 * the comparison never crosses the boundary at all. Zero disables the rule.
 */
static int RetainShape( World* world, b3ShapeId shape )
{
	if ( world->shapeCount >= MAX_SHAPES )
	{
		return -1;
	}

	int handle = world->shapeCount++;
	world->shapes[handle] = shape;
	return handle;
}

/**
 * Attach a box shape, optionally offset and rotated in body space.
 *
 * Box3D has no collider transform: a shape IS its geometry, so an offset box is
 * a transformed hull rather than a hull plus a pose. That is why the offset is
 * per shape here instead of a field on the shape def — the game's merged block
 * colliders sit off-centre, and placing them at the body origin would look like
 * a solver bug rather than a missing parameter.
 */
int b3w_add_box_shape( int handle, int body, float hx, float hy, float hz, float ox, float oy, float oz, float qx,
					   float qy, float qz, float qw, float density, float friction, float restitution, int group )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return -1;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;
	def.filter.groupIndex = group;

	// b3CreateHullShape copies the hull, so a stack local is enough — this is
	// how Box3D's own samples do it. Only hulls built with b3CreateHull are
	// heap-owned and have to outlive their shape.
	b3Transform pose = { .p = ( b3Vec3 ){ ox, oy, oz }, .q = ( b3Quat ){ { qx, qy, qz }, qw } };
	b3BoxHull hull = b3MakeTransformedBoxHull( hx, hy, hz, pose );
	return RetainShape( world, b3CreateHullShape( world->bodies[body], &def, &hull.base ) );
}

int b3w_body_count( int handle )
{
	World* world = GetWorld( handle );
	return world == NULL ? 0 : world->bodyCount;
}

int b3w_is_awake( int handle, int body )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return 0;
	}

	return b3Body_IsAwake( world->bodies[body] ) ? 1 : 0;
}

// --- bulk readback -------------------------------------------------------

/// Pointer to this world's transform buffer, as a byte offset into the WASM
/// heap. Stable until a body is added, which may reallocate it — hosts must
/// re-read it after any `b3w_create_body`, never cache it across one.
int b3w_transforms_ptr( int handle )
{
	World* world = GetWorld( handle );
	return world == NULL ? 0 : (int)(intptr_t)world->transforms;
}

/// Fill the transform buffer from the world. One call per step, then the host
/// reads 7 floats per body straight out of the heap.
void b3w_read_transforms( int handle )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return;
	}

	float* out = world->transforms;
	for ( int i = 0; i < world->bodyCount; ++i )
	{
		b3WorldTransform transform = b3Body_GetTransform( world->bodies[i] );

		out[0] = (float)transform.p.x;
		out[1] = (float)transform.p.y;
		out[2] = (float)transform.p.z;
		out[3] = transform.q.v.x;
		out[4] = transform.q.v.y;
		out[5] = transform.q.v.z;
		out[6] = transform.q.s;
		out += 7;
	}
}

// --- per-body operations -------------------------------------------------
//
// The set the game's `RigidBody` interface needs, and no more. Each takes the
// world and body handle rather than a Box3D id, so nothing about id layout
// crosses the boundary.

static b3BodyId BodyId( int handle, int body )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return b3_nullBodyId;
	}

	return world->bodies[body];
}

void b3w_body_transform( int handle, int body, float* out )
{
	b3WorldTransform t = b3Body_GetTransform( BodyId( handle, body ) );

	out[0] = (float)t.p.x;
	out[1] = (float)t.p.y;
	out[2] = (float)t.p.z;
	out[3] = t.q.v.x;
	out[4] = t.q.v.y;
	out[5] = t.q.v.z;
	out[6] = t.q.s;
}

void b3w_set_position( int handle, int body, float x, float y, float z )
{
	b3BodyId id = BodyId( handle, body );
	b3Body_SetTransform( id, ( b3Pos ){ x, y, z }, b3Body_GetRotation( id ) );
}

void b3w_set_rotation( int handle, int body, float qx, float qy, float qz, float qw )
{
	b3BodyId id = BodyId( handle, body );
	b3Body_SetTransform( id, b3Body_GetPosition( id ), ( b3Quat ){ { qx, qy, qz }, qw } );
}

void b3w_linear_velocity( int handle, int body, float* out )
{
	b3Vec3 v = b3Body_GetLinearVelocity( BodyId( handle, body ) );
	out[0] = v.x;
	out[1] = v.y;
	out[2] = v.z;
}

void b3w_angular_velocity( int handle, int body, float* out )
{
	b3Vec3 w = b3Body_GetAngularVelocity( BodyId( handle, body ) );
	out[0] = w.x;
	out[1] = w.y;
	out[2] = w.z;
}

/**
 * Set linear velocity, optionally without waking.
 *
 * Box3D wakes on any non-zero velocity and offers no flag to suppress it
 * (body.c: `if (b3LengthSquared(v) > 0) b3WakeBodyWithLock(...)`). The game
 * needs the suppressed form: the player's ground clamp writes a non-zero
 * velocity every frame while standing still, and waking there would keep the
 * player — and through contact everything they stand on — awake forever.
 *
 * Re-sleeping immediately is exact rather than approximate: no step runs
 * between the two calls, so there is no intermediate solver state, and for a
 * body that was already awake `wake = false` is a no-op either way.
 */
void b3w_set_linear_velocity( int handle, int body, float x, float y, float z, int wake )
{
	b3BodyId id = BodyId( handle, body );
	bool wasAsleep = b3Body_IsAwake( id ) == false;

	b3Body_SetLinearVelocity( id, ( b3Vec3 ){ x, y, z } );

	if ( wake == 0 && wasAsleep )
	{
		b3Body_SetAwake( id, false );
	}
}

void b3w_set_angular_velocity( int handle, int body, float x, float y, float z, int wake )
{
	b3BodyId id = BodyId( handle, body );
	bool wasAsleep = b3Body_IsAwake( id ) == false;

	b3Body_SetAngularVelocity( id, ( b3Vec3 ){ x, y, z } );

	if ( wake == 0 && wasAsleep )
	{
		b3Body_SetAwake( id, false );
	}
}

void b3w_apply_impulse( int handle, int body, float x, float y, float z )
{
	b3BodyId id = BodyId( handle, body );
	b3Body_ApplyLinearImpulse( id, ( b3Vec3 ){ x, y, z }, b3Body_GetWorldCenter( id ), true );
}

void b3w_apply_force( int handle, int body, float x, float y, float z, int wake )
{
	b3BodyId id = BodyId( handle, body );
	b3Body_ApplyForce( id, ( b3Vec3 ){ x, y, z }, b3Body_GetWorldCenter( id ), wake != 0 );
}

void b3w_apply_torque( int handle, int body, float x, float y, float z, int wake )
{
	b3Body_ApplyTorque( BodyId( handle, body ), ( b3Vec3 ){ x, y, z }, wake != 0 );
}

float b3w_mass( int handle, int body )
{
	return b3Body_GetMass( BodyId( handle, body ) );
}

/// Diagonal of the local rotational inertia, which is what the force pipeline
/// reads to judge whether a torque could actually turn the body.
void b3w_principal_inertia( int handle, int body, float* out )
{
	b3Matrix3 I = b3Body_GetLocalRotationalInertia( BodyId( handle, body ) );
	out[0] = I.cx.x;
	out[1] = I.cy.y;
	out[2] = I.cz.z;
}

int b3w_body_type( int handle, int body )
{
	return (int)b3Body_GetType( BodyId( handle, body ) );
}

void b3w_set_body_type( int handle, int body, int type )
{
	b3Body_SetType( BodyId( handle, body ), (b3BodyType)type );
}

void b3w_set_gravity_scale( int handle, int body, float scale )
{
	b3Body_SetGravityScale( BodyId( handle, body ), scale );
}

void b3w_wake( int handle, int body )
{
	b3Body_SetAwake( BodyId( handle, body ), true );
}

int b3w_is_valid( int handle, int body )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return 0;
	}

	return b3Body_IsValid( world->bodies[body] ) ? 1 : 0;
}

/// A capsule along local Y, the shape the player uses. `halfHeight` is the
/// distance from the centre to a hemisphere's centre, matching the game's
/// CapsuleColliderConfig rather than Box3D's two-point form.
int b3w_add_capsule_shape( int handle, int body, float halfHeight, float radius, float ox, float oy, float oz,
						   float density, float friction, float restitution )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return -1;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;

	b3Capsule capsule = {
		.center1 = ( b3Vec3 ){ ox, oy - halfHeight, oz },
		.center2 = ( b3Vec3 ){ ox, oy + halfHeight, oz },
		.radius = radius,
	};
	return RetainShape( world, b3CreateCapsuleShape( world->bodies[body], &def, &capsule ) );
}

// --- terrain and queries -------------------------------------------------

/// Allocate a float buffer in the wasm heap for the host to fill. Returned as a
/// byte offset; free with `free`.
int b3w_alloc_floats( int count )
{
	return (int)(intptr_t)malloc( sizeof( float ) * count );
}

/**
 * Attach a height field to a body.
 *
 * The host passes heights as a pointer into the wasm heap it filled itself, so
 * a terrain upload is one crossing rather than one per sample. Box3D keeps a
 * REFERENCE to the height field (see b3CreateHeightFieldShape's warning), so
 * the data must outlive the shape — hence the copy into memory we own.
 *
 * Index order is deliberately NOT converted here. Whether the host's array is
 * row-major over x or over z is the host's business, and hiding a transpose in
 * the binding is how "props fall through terrain in one quadrant" happens: the
 * bug becomes invisible from both sides. The host sends what Box3D wants.
 */
void b3w_add_height_field( int handle, int body, int heightsPtr, int countX, int countZ, float scaleX, float scaleY,
						   float scaleZ )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return;
	}

	int sampleCount = countX * countZ;
	float* heights = (float*)malloc( sizeof( float ) * sampleCount );
	memcpy( heights, (const void*)(intptr_t)heightsPtr, sizeof( float ) * sampleCount );

	float minHeight = heights[0];
	float maxHeight = heights[0];
	for ( int i = 1; i < sampleCount; ++i )
	{
		if ( heights[i] < minHeight ) minHeight = heights[i];
		if ( heights[i] > maxHeight ) maxHeight = heights[i];
	}

	b3HeightFieldDef def = { 0 };
	def.heights = heights;
	def.materialIndices = NULL;
	def.scale = ( b3Vec3 ){ scaleX, scaleY, scaleZ };
	def.countX = countX;
	def.countZ = countZ;
	def.globalMinimumHeight = minHeight;
	def.globalMaximumHeight = maxHeight;

	b3HeightFieldData* field = b3CreateHeightField( &def );
	free( heights );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3CreateHeightFieldShape( world->bodies[body], &shapeDef, field );
	// Height fields are never destroyed or re-filtered, so no handle is kept.
}

/// Cast a ray and return the hit's world Y, or `missValue` when nothing is hit.
float b3w_cast_ray_y( int handle, float ox, float oy, float oz, float dx, float dy, float dz, float missValue )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return missValue;
	}

	b3RayResult hit = b3World_CastRayClosest( world->id, ( b3Pos ){ ox, oy, oz }, ( b3Vec3 ){ dx, dy, dz },
											  b3DefaultQueryFilter() );
	if ( hit.hit == false )
	{
		return missValue;
	}

	return (float)hit.point.y;
}

// --- joints --------------------------------------------------------------
//
// Box3D expresses a joint as two local FRAMES and constrains a fixed axis of
// them: revolute rotates about the frame's z, prismatic slides along its x.
// Callers here pass an axis vector instead, the way most engines and this
// game's own data do, and the frame is built from it.
//
// This is the conversion the migration plan calls its worst trap, because a
// wrong frame is invisible in any test whose bodies happen to be axis-aligned
// the same way. It is deliberately written out here rather than borrowed.

/// Shortest-arc rotation taking `from` to `to`, both unit length.
static b3Quat QuatFromTo( b3Vec3 from, b3Vec3 to )
{
	float d = b3Dot( from, to );

	// Anti-parallel: no unique shortest arc, so pick any perpendicular axis.
	if ( d < -0.999999f )
	{
		b3Vec3 axis = b3Cross( ( b3Vec3 ){ 1.0f, 0.0f, 0.0f }, from );
		if ( b3LengthSquared( axis ) < 1e-6f )
		{
			axis = b3Cross( ( b3Vec3 ){ 0.0f, 1.0f, 0.0f }, from );
		}
		return b3MakeQuatFromAxisAngle( b3Normalize( axis ), 3.14159265358979323846f );
	}

	if ( d > 0.999999f )
	{
		return ( b3Quat ){ { 0.0f, 0.0f, 0.0f }, 1.0f };
	}

	b3Vec3 c = b3Cross( from, to );
	b3Quat q = { { c.x, c.y, c.z }, 1.0f + d };
	return b3NormalizeQuat( q );
}

/// A local frame at `anchor` whose `localAxis` points along `axis`.
static b3Transform FrameAlong( b3Vec3 anchor, b3Vec3 localAxis, b3Vec3 axis )
{
	b3Transform frame;
	frame.p = anchor;
	frame.q = QuatFromTo( localAxis, b3Normalize( axis ) );
	return frame;
}

static const b3Vec3 AXIS_X = { 1.0f, 0.0f, 0.0f };
static const b3Vec3 AXIS_Z = { 0.0f, 0.0f, 1.0f };

int b3w_add_revolute_joint( int handle, int bodyA, int bodyB, float ax, float ay, float az, float bx, float by,
							float bz, float axisX, float axisY, float axisZ )
{
	World* world = GetWorld( handle );
	if ( world == NULL || world->jointCount >= MAX_JOINTS )
	{
		return -1;
	}

	b3Vec3 axis = { axisX, axisY, axisZ };

	b3RevoluteJointDef def = b3DefaultRevoluteJointDef();
	def.base.bodyIdA = BodyId( handle, bodyA );
	def.base.bodyIdB = BodyId( handle, bodyB );
	def.base.localFrameA = FrameAlong( ( b3Vec3 ){ ax, ay, az }, AXIS_Z, axis );
	def.base.localFrameB = FrameAlong( ( b3Vec3 ){ bx, by, bz }, AXIS_Z, axis );

	int joint = world->jointCount++;
	world->joints[joint] = b3CreateRevoluteJoint( world->id, &def );
	return joint;
}

int b3w_add_prismatic_joint( int handle, int bodyA, int bodyB, float ax, float ay, float az, float bx, float by,
							 float bz, float axisX, float axisY, float axisZ )
{
	World* world = GetWorld( handle );
	if ( world == NULL || world->jointCount >= MAX_JOINTS )
	{
		return -1;
	}

	b3Vec3 axis = { axisX, axisY, axisZ };

	b3PrismaticJointDef def = b3DefaultPrismaticJointDef();
	def.base.bodyIdA = BodyId( handle, bodyA );
	def.base.bodyIdB = BodyId( handle, bodyB );
	def.base.localFrameA = FrameAlong( ( b3Vec3 ){ ax, ay, az }, AXIS_X, axis );
	def.base.localFrameB = FrameAlong( ( b3Vec3 ){ bx, by, bz }, AXIS_X, axis );

	int joint = world->jointCount++;
	world->joints[joint] = b3CreatePrismaticJoint( world->id, &def );
	return joint;
}

/**
 * Attach a static triangle mesh.
 *
 * No internal-edge flag is passed because Box3D has none to pass: ghost-contact
 * rejection is unconditional in `mesh_contact.c`, which classifies each
 * triangle manifold by feature and accepts an edge or vertex contact only when
 * no adjacent triangle already claimed it. Rapier needs
 * `TriMeshFlags.FIX_INTERNAL_EDGES` for the same effect, and without it a
 * capsule crossing the shared edge of two coplanar triangles is flicked
 * upward. `weldTolerance` covers the vertex merging that flag also implies —
 * adjacency cannot exist across duplicated vertices.
 *
 * Box3D keeps a reference to the mesh (see b3CreateMeshShape), so the data is
 * copied into memory we own and outlives the shape.
 */
int b3w_add_mesh_shape( int handle, int body, int verticesPtr, int vertexCount, int indicesPtr, int triangleCount,
						 float friction, float restitution, int group )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return -1;
	}

	b3Vec3* vertices = (b3Vec3*)malloc( sizeof( b3Vec3 ) * vertexCount );
	const float* src = (const float*)(intptr_t)verticesPtr;
	for ( int i = 0; i < vertexCount; ++i )
	{
		vertices[i] = ( b3Vec3 ){ src[3 * i], src[3 * i + 1], src[3 * i + 2] };
	}

	int32_t* indices = (int32_t*)malloc( sizeof( int32_t ) * 3 * triangleCount );
	memcpy( indices, (const void*)(intptr_t)indicesPtr, sizeof( int32_t ) * 3 * triangleCount );

	b3MeshDef def = { 0 };
	def.vertices = vertices;
	def.indices = indices;
	def.materialIndices = NULL;
	def.weldTolerance = 1e-4f;
	def.vertexCount = vertexCount;
	def.triangleCount = triangleCount;

	b3MeshData* mesh = b3CreateMesh( &def, NULL, 0 );
	free( vertices );
	free( indices );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = friction;
	shapeDef.baseMaterial.restitution = restitution;
	shapeDef.filter.groupIndex = group;

	return RetainShape( world, b3CreateMeshShape( world->bodies[body], &shapeDef, mesh, ( b3Vec3 ){ 1.0f, 1.0f, 1.0f } ) );
}

// --- joint motors and limits ---------------------------------------------
//
// The game drives every wheel, pivot and piston through a motor, so this is
// the rest of the joint surface. Revolute and prismatic take the same shape
// deliberately — one host-side call site rather than two — even though Box3D
// names the force cap differently for each (torque vs force).

static b3JointId JointId( int handle, int joint )
{
	World* world = GetWorld( handle );
	if ( world == NULL || joint < 0 || joint >= world->jointCount )
	{
		return b3_nullJointId;
	}

	return world->joints[joint];
}

/// Drive a revolute joint at `speed` rad/s, capped at `maxTorque`. A zero or
/// negative cap disables the motor rather than fighting with zero authority.
void b3w_set_revolute_motor( int handle, int joint, float speed, float maxTorque )
{
	b3JointId id = JointId( handle, joint );

	b3RevoluteJoint_EnableMotor( id, maxTorque > 0.0f );
	b3RevoluteJoint_SetMotorSpeed( id, speed );
	b3RevoluteJoint_SetMaxMotorTorque( id, maxTorque );
}

void b3w_set_revolute_limits( int handle, int joint, int enable, float lower, float upper )
{
	b3JointId id = JointId( handle, joint );

	b3RevoluteJoint_EnableLimit( id, enable != 0 );
	if ( enable != 0 )
	{
		b3RevoluteJoint_SetLimits( id, lower, upper );
	}
}

/// As above, for a prismatic joint: `speed` in m/s, capped at `maxForce`.
void b3w_set_prismatic_motor( int handle, int joint, float speed, float maxForce )
{
	b3JointId id = JointId( handle, joint );

	b3PrismaticJoint_EnableMotor( id, maxForce > 0.0f );
	b3PrismaticJoint_SetMotorSpeed( id, speed );
	b3PrismaticJoint_SetMaxMotorForce( id, maxForce );
}

void b3w_set_prismatic_limits( int handle, int joint, int enable, float lower, float upper )
{
	b3JointId id = JointId( handle, joint );

	b3PrismaticJoint_EnableLimit( id, enable != 0 );
	if ( enable != 0 )
	{
		b3PrismaticJoint_SetLimits( id, lower, upper );
	}
}

// --- shape casts ---------------------------------------------------------
//
// Box3D casts a POINT CLOUD with a radius rather than a shape object: a sphere
// is one point, a capsule is two. That covers what the game casts — the
// player's ground probe and the vehicle controller's suspension — without the
// binding needing a shape registry.
//
// The engine reports every hit through a callback and lets the callback clip
// the remaining range by what it returns. Closest-hit is therefore "return the
// fraction you just saw", which is cheaper than collecting and sorting.

typedef struct ClosestHit
{
	float fraction;
	b3Vec3 normal;
	bool hit;
} ClosestHit;

static float OnCastHit( b3ShapeId shapeId, b3Pos point, b3Vec3 normal, float fraction, uint64_t userMaterialId,
						int triangleIndex, int childIndex, void* context )
{
	(void)shapeId;
	(void)point;
	(void)userMaterialId;
	(void)triangleIndex;
	(void)childIndex;

	ClosestHit* closest = (ClosestHit*)context;
	closest->fraction = fraction;
	closest->normal = normal;
	closest->hit = true;

	// Clip the cast to here: anything beyond is no longer interesting.
	return fraction;
}

/**
 * Sweep a sphere and report the first surface it meets.
 *
 * `out` receives normal(3) + fraction(1). Returns 1 on a hit, 0 on a miss —
 * the fraction is only meaningful when it returns 1.
 */
int b3w_cast_sphere( int handle, float cx, float cy, float cz, float radius, float dx, float dy, float dz, float* out )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return 0;
	}

	b3Vec3 point = { 0.0f, 0.0f, 0.0f };
	b3ShapeProxy proxy = { .points = &point, .count = 1, .radius = radius };

	ClosestHit closest = { .fraction = 1.0f, .normal = { 0.0f, 1.0f, 0.0f }, .hit = false };
	b3World_CastShape( world->id, ( b3Pos ){ cx, cy, cz }, &proxy, ( b3Vec3 ){ dx, dy, dz }, b3DefaultQueryFilter(),
					   OnCastHit, &closest );

	if ( closest.hit == false )
	{
		return 0;
	}

	out[0] = closest.normal.x;
	out[1] = closest.normal.y;
	out[2] = closest.normal.z;
	out[3] = closest.fraction;
	return 1;
}

// --- the rest of the shape vocabulary ------------------------------------
//
// Hulls built by b3CreateCylinder and b3CreateHull are HEAP-owned and must
// outlive the shape that references them, unlike b3MakeBoxHull's stack value.
// They are kept on the world and released with it, because the game rebuilds
// structures constantly and a per-shape free would need a registry nobody
// would keep in sync.

static void RetainHull( World* world, b3HullData* hull )
{
	if ( world->hullCount >= MAX_HULLS )
	{
		return;
	}

	world->hulls[world->hullCount++] = hull;
}

int b3w_add_sphere_shape( int handle, int body, float radius, float ox, float oy, float oz, float density,
						  float friction, float restitution, int group )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return -1;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;
	def.filter.groupIndex = group;

	b3Sphere sphere = { .center = { ox, oy, oz }, .radius = radius };
	return RetainShape( world, b3CreateSphereShape( world->bodies[body], &def, &sphere ) );
}

/// A cylinder about local Y, matching how the game orients wheels.
int b3w_add_cylinder_shape( int handle, int body, float halfHeight, float radius, int sides, float density,
							 float friction, float restitution, int group )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return -1;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;
	def.filter.groupIndex = group;

	b3HullData* hull = b3CreateCylinder( 2.0f * halfHeight, radius, -halfHeight, sides );
	RetainHull( world, hull );
	return RetainShape( world, b3CreateHullShape( world->bodies[body], &def, hull ) );
}

/// A convex hull over a point cloud the host uploaded.
int b3w_add_hull_shape( int handle, int body, int pointsPtr, int pointCount, int maxVertices, float density,
						 float friction, float restitution, int group )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return -1;
	}

	b3Vec3* points = (b3Vec3*)malloc( sizeof( b3Vec3 ) * pointCount );
	const float* src = (const float*)(intptr_t)pointsPtr;
	for ( int i = 0; i < pointCount; ++i )
	{
		points[i] = ( b3Vec3 ){ src[3 * i], src[3 * i + 1], src[3 * i + 2] };
	}

	b3HullData* hull = b3CreateHull( points, pointCount, maxVertices );
	free( points );
	if ( hull == NULL )
	{
		return -1;
	}

	RetainHull( world, hull );

	b3ShapeDef def = b3DefaultShapeDef();
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;
	def.filter.groupIndex = group;

	return RetainShape( world, b3CreateHullShape( world->bodies[body], &def, hull ) );
}

// --- lifecycle and world state -------------------------------------------

/// Destroy a body and everything attached to it. The handle stays allocated so
/// that stale references fail a validity check rather than addressing a
/// recycled body — `b3w_is_valid` is the way to ask.
void b3w_destroy_body( int handle, int body )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return;
	}

	b3DestroyBody( world->bodies[body] );
}

void b3w_set_gravity( int handle, float x, float y, float z )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return;
	}

	b3World_SetGravity( world->id, ( b3Vec3 ){ x, y, z } );
}

int b3w_shape_count( int handle, int body )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return 0;
	}

	return b3Body_GetShapeCount( world->bodies[body] );
}

/**
 * Cast a ray and report the full hit: point, normal and which body was struck.
 *
 * `out` receives point(3) + normal(3) + fraction(1) + bodyHandle(1). Returns 1
 * on a hit, 0 on a miss. The body handle comes from the body's own user data
 * rather than a scan of the body table — a raycast per tool per frame cannot
 * afford a linear search, and the alternative registry would be one more thing
 * to keep in sync.
 *
 * A body handle of -1 means the shape belongs to a body this binding did not
 * create, which should not happen and is reported rather than guessed at.
 */
typedef struct ClosestRayHit
{
	float fraction;
	b3Vec3 normal;
	b3Pos point;
	int body;
	bool hit;
} ClosestRayHit;

static float OnRayHit( b3ShapeId shapeId, b3Pos point, b3Vec3 normal, float fraction, uint64_t userMaterialId,
					   int triangleIndex, int childIndex, void* context )
{
	(void)userMaterialId;
	(void)triangleIndex;
	(void)childIndex;

	ClosestRayHit* closest = (ClosestRayHit*)context;
	closest->fraction = fraction;
	closest->normal = normal;
	closest->point = point;
	closest->hit = true;

	intptr_t tagged = (intptr_t)b3Body_GetUserData( b3Shape_GetBody( shapeId ) );
	closest->body = tagged > 0 ? (int)( tagged - 1 ) : -1;

	return fraction;
}

/**
 * A closest-hit ray cast that can skip bodies.
 *
 * `b3World_CastRay`'s callback returns -1 to drop a hit and keep going, which
 * is the difference between "ignore this body" and "stop at this body and
 * report nothing" — the game's tool picks and sensors need to see what is
 * BEHIND the thing they exclude. `exclude` is a body index or -1; `skipLo`/
 * `skipHi` bound a contiguous run of body indices to ignore (the caller's own
 * structure), because passing a set across the wasm boundary per cast would
 * cost more than the cast.
 */
typedef struct
{
	ClosestRayHit closest;
	int exclude;
	int skipLo;
	int skipHi;
} FilteredRayHit;

static int BodyIndexOfShape( b3ShapeId shapeId )
{
	intptr_t tagged = (intptr_t)b3Body_GetUserData( b3Shape_GetBody( shapeId ) );
	return tagged > 0 ? (int)( tagged - 1 ) : -1;
}

static float OnFilteredRayHit( b3ShapeId shapeId, b3Pos point, b3Vec3 normal, float fraction, uint64_t userMaterialId,
							   int triangleIndex, int childIndex, void* context )
{
	(void)userMaterialId;
	(void)triangleIndex;
	(void)childIndex;

	FilteredRayHit* ctx = (FilteredRayHit*)context;
	int body = BodyIndexOfShape( shapeId );

	if ( body >= 0 && ( body == ctx->exclude || ( body >= ctx->skipLo && body <= ctx->skipHi ) ) )
	{
		return -1.0f;
	}

	ctx->closest.fraction = fraction;
	ctx->closest.normal = normal;
	ctx->closest.point = point;
	ctx->closest.hit = true;
	ctx->closest.body = body;

	return fraction;
}

int b3w_cast_ray_filtered( int handle, float ox, float oy, float oz, float dx, float dy, float dz, int categoryBits,
						   int maskBits, int exclude, int skipLo, int skipHi, float* out )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return 0;
	}

	b3QueryFilter filter = b3DefaultQueryFilter();
	if ( maskBits != 0 )
	{
		filter.categoryBits = (uint64_t)(uint32_t)categoryBits;
		filter.maskBits = (uint64_t)(uint32_t)maskBits;
	}

	FilteredRayHit ctx = { .closest = { .fraction = 1.0f, .body = -1, .hit = false },
						   .exclude = exclude,
						   .skipLo = skipLo,
						   .skipHi = skipHi };

	b3World_CastRay( world->id, ( b3Pos ){ ox, oy, oz }, ( b3Vec3 ){ dx, dy, dz }, filter, OnFilteredRayHit, &ctx );

	if ( ctx.closest.hit == false )
	{
		return 0;
	}

	out[0] = (float)ctx.closest.point.x;
	out[1] = (float)ctx.closest.point.y;
	out[2] = (float)ctx.closest.point.z;
	out[3] = ctx.closest.normal.x;
	out[4] = ctx.closest.normal.y;
	out[5] = ctx.closest.normal.z;
	out[6] = ctx.closest.fraction;
	out[7] = (float)ctx.closest.body;
	return 1;
}

int b3w_cast_ray_detailed( int handle, float ox, float oy, float oz, float dx, float dy, float dz, float* out )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return 0;
	}

	ClosestRayHit closest = { .fraction = 1.0f, .body = -1, .hit = false };
	b3World_CastRay( world->id, ( b3Pos ){ ox, oy, oz }, ( b3Vec3 ){ dx, dy, dz }, b3DefaultQueryFilter(), OnRayHit,
					 &closest );

	if ( closest.hit == false )
	{
		return 0;
	}

	out[0] = (float)closest.point.x;
	out[1] = (float)closest.point.y;
	out[2] = (float)closest.point.z;
	out[3] = closest.normal.x;
	out[4] = closest.normal.y;
	out[5] = closest.normal.z;
	out[6] = closest.fraction;
	out[7] = (float)closest.body;
	return 1;
}

/**
 * Servo a revolute joint toward a target angle.
 *
 * Rapier has a position motor; Box3D has a SPRING with a target angle, tuned in
 * hertz and damping ratio rather than stiffness and damping. That is a
 * different parameterisation of the same idea, not a different feature, but the
 * numbers do not carry across — hertz is a frequency, and the game's pivot
 * stiffness values are not.
 *
 * `hertz <= 0` disables the spring, which is how a servo is turned off without
 * leaving it pulling toward a stale angle.
 */
void b3w_set_revolute_servo( int handle, int joint, float targetRadians, float hertz, float dampingRatio )
{
	b3JointId id = JointId( handle, joint );

	b3RevoluteJoint_EnableSpring( id, hertz > 0.0f );
	b3RevoluteJoint_SetTargetAngle( id, targetRadians );
	b3RevoluteJoint_SetSpringHertz( id, hertz );
	b3RevoluteJoint_SetSpringDampingRatio( id, dampingRatio );
}

/// Current joint angle in radians, which the servo and the game's readouts need.
float b3w_revolute_angle( int handle, int joint )
{
	return b3RevoluteJoint_GetAngle( JointId( handle, joint ) );
}

// --- shape mutation ------------------------------------------------------

/**
 * Destroy one shape, leaving its body alone.
 *
 * `updateBodyMass` is true because the game rebuilds a body's colliders in
 * place — a structure that loses a block must lose its mass with it, or every
 * force applied afterwards is computed against a body heavier than it looks.
 */
void b3w_destroy_shape( int handle, int shape )
{
	World* world = GetWorld( handle );
	if ( world == NULL || shape < 0 || shape >= world->shapeCount )
	{
		return;
	}

	if ( b3Shape_IsValid( world->shapes[shape] ) )
	{
		b3DestroyShape( world->shapes[shape], true );
	}
}

int b3w_shape_is_valid( int handle, int shape )
{
	World* world = GetWorld( handle );
	if ( world == NULL || shape < 0 || shape >= world->shapeCount )
	{
		return 0;
	}

	return b3Shape_IsValid( world->shapes[shape] ) ? 1 : 0;
}

/// Destroy a joint. Joints otherwise live until their world does, which is
/// wrong for a game that rebuilds structures on every placement.
void b3w_destroy_joint( int handle, int joint )
{
	World* world = GetWorld( handle );
	if ( world == NULL || joint < 0 || joint >= world->jointCount )
	{
		return;
	}

	if ( b3Joint_IsValid( world->joints[joint] ) )
	{
		b3DestroyJoint( world->joints[joint], true );
	}
}

/**
 * Set a shape's category and mask bits.
 *
 * The OTHER filtering mechanism, distinct from `groupIndex`: a group index says
 * "these shapes are one assembly and must not self-collide", while these bits
 * say "this shape is terrain, and terrain collides with these categories".
 * Conflating them produces objects that collide with the wrong set, which reads
 * as a solver bug rather than a filter bug.
 *
 * Applied after creation because that is where callers already set it — the
 * equivalent Rapier call is made on the collider, not on its descriptor.
 */
void b3w_set_shape_filter( int handle, int shape, int category, int mask )
{
	World* world = GetWorld( handle );
	if ( world == NULL || shape < 0 || shape >= world->shapeCount )
	{
		return;
	}

	if ( b3Shape_IsValid( world->shapes[shape] ) == false )
	{
		return;
	}

	b3Filter filter = b3Shape_GetFilter( world->shapes[shape] );
	filter.categoryBits = (uint64_t)(uint32_t)category;
	filter.maskBits = (uint64_t)(uint32_t)mask;
	b3Shape_SetFilter( world->shapes[shape], filter, true );
}

/// Current extension of a prismatic joint. Box3D tracks this; Rapier's binding
/// exposes no equivalent, which is why its side re-derives it.
float b3w_prismatic_translation( int handle, int joint )
{
	return b3PrismaticJoint_GetTranslation( JointId( handle, joint ) );
}

// ---------------------------------------------------------------------------
// Vehicle controller
//
// The controller itself lives in vehicle.c. These wrappers exist only to turn
// JavaScript's world handle and body index into the ids it works in.
// ---------------------------------------------------------------------------

int b3w_vehicle_create( b3WorldId world, b3BodyId chassis );
void b3w_vehicle_update( void* vehicle, float dt );
void* b3w_vehicle_get( int handle );

int b3w_create_vehicle( int handle, int chassis )
{
	World* world = GetWorld( handle );
	if ( world == NULL || chassis < 0 || chassis >= world->bodyCount )
	{
		return -1;
	}
	return b3w_vehicle_create( world->id, world->bodies[chassis] );
}

void b3w_step_vehicle( int vehicle, float dt )
{
	void* v = b3w_vehicle_get( vehicle );
	if ( v != NULL )
	{
		b3w_vehicle_update( v, dt );
	}
}

/**
 * Force a body's total mass, keeping the inertia distribution Box3D computed.
 *
 * Box3D derives mass from shape density; the game sometimes states a mass
 * outright — the player capsule fixes its own so movement feel does not change
 * with height, and structural colliders carry per-cell masses. Scaling the
 * computed tensor rather than replacing it keeps the shape's inertia
 * PROPORTIONS, which is what makes the body rotate like its geometry.
 */
void b3w_set_body_mass( int handle, int body, float mass )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount || mass <= 0.0f )
	{
		return;
	}

	b3BodyId id = world->bodies[body];
	b3MassData md = b3Body_GetMassData( id );
	if ( md.mass <= 0.0f )
	{
		return;
	}

	float scale = mass / md.mass;
	md.mass = mass;
	md.inertia.cx = b3MulSV( scale, md.inertia.cx );
	md.inertia.cy = b3MulSV( scale, md.inertia.cy );
	md.inertia.cz = b3MulSV( scale, md.inertia.cz );
	b3Body_SetMassData( id, md );
}
