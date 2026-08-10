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

typedef struct World
{
	b3WorldId id;
	bool inUse;

	b3BodyId* bodies;
	int bodyCount;
	int bodyCapacity;

	b3JointId joints[MAX_JOINTS];
	int jointCount;

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

int b3w_create_body( int handle, int bodyType, float px, float py, float pz, float qx, float qy, float qz, float qw )
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

	ReserveBodies( world, world->bodyCount + 1 );

	int body = world->bodyCount++;
	world->bodies[body] = b3CreateBody( world->id, &def );

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
void b3w_add_box_shape( int handle, int body, float hx, float hy, float hz, float density, float friction,
						float restitution, int group )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;
	def.filter.groupIndex = group;

	// b3CreateHullShape copies the hull, so a stack local is enough — this is
	// how Box3D's own samples do it. Only hulls built with b3CreateHull are
	// heap-owned and have to outlive their shape.
	b3BoxHull hull = b3MakeBoxHull( hx, hy, hz );
	b3CreateHullShape( world->bodies[body], &def, &hull.base );
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
void b3w_add_capsule_shape( int handle, int body, float halfHeight, float radius, float density, float friction,
							float restitution )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;

	b3Capsule capsule = {
		.center1 = ( b3Vec3 ){ 0.0f, -halfHeight, 0.0f },
		.center2 = ( b3Vec3 ){ 0.0f, halfHeight, 0.0f },
		.radius = radius,
	};
	b3CreateCapsuleShape( world->bodies[body], &def, &capsule );
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
void b3w_add_mesh_shape( int handle, int body, int verticesPtr, int vertexCount, int indicesPtr, int triangleCount,
						 float friction, float restitution, int group )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return;
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

	b3CreateMeshShape( world->bodies[body], &shapeDef, mesh, ( b3Vec3 ){ 1.0f, 1.0f, 1.0f } );
}
