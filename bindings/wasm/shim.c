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
	/**
	 * Which structure owns each body, parallel to `bodies`.
	 *
	 * NOT in the body's userData: that slot already holds the body INDEX, which
	 * every raycast reads back to say what it hit. Storing the owner there
	 * silently rewrote every query result — four raycast suites went red at
	 * once. One slot, one meaning.
	 */
	uint32_t* bodyOwner;
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
	world->bodyOwner = (uint32_t*)realloc( world->bodyOwner, sizeof( uint32_t ) * capacity );
	memset( world->bodyOwner + world->bodyCapacity, 0, sizeof( uint32_t ) * (size_t)( capacity - world->bodyCapacity ) );
	world->bodyCapacity = capacity;
}

// Defined in vehicle.c; registered as the world substep callback below.
void b3w_vehicle_substep( float h, void* context );

// Defined in vehicle.c; registered as the world substep callback below.
void b3w_vehicle_substep( float h, void* context );
void b3w_vehicle_destroy_world( b3WorldId world );
void b3w_vehicle_disarm_world( b3WorldId world );

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

	// The vehicle force model runs per substep, not per tick — see
	// `b3w_vehicle_substep`. Registered once here so no caller has to know.
	// The context is WHICH world is stepping: the vehicle registry is global, so
	// the callback needs it to solve this world's vehicles and no others.
	b3World_SetSubStepCallback( world->id, b3w_vehicle_substep, (void*)(uintptr_t)b3StoreWorldId( world->id ) );
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

	// Before the world goes: its vehicles hold body ids that are about to be
	// freed, and ids are reused by the next world to take this slot.
	b3w_vehicle_destroy_world( world->id );
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

	// The casts this step ran on are spent — see `b3w_vehicle_disarm_world`.
	b3w_vehicle_disarm_world( world->id );
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
	world->bodyOwner[body] = 0;

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
					   float qy, float qz, float qw, float density, float friction, float restitution, int group,
					   int isSensor )
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
	// A sensor overlaps without solving contacts. Box3D binds this at CREATION
	// (it allocates a sensor slot in b3CreateShape), so there is no setter to
	// call afterwards and it has to arrive as a parameter.
	def.isSensor = isSensor != 0;
	def.enableSensorEvents = isSensor != 0;

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
						   float density, float friction, float restitution, int isSensor )
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
	def.isSensor = isSensor != 0;
	def.enableSensorEvents = isSensor != 0;

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
int b3w_add_height_field( int handle, int body, int heightsPtr, int countX, int countZ, float scaleX, float scaleY,
						  float scaleZ, float friction, float restitution )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return -1;
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

	// Terrain is a surface the game tunes like any other: it states a friction
	// and a collision group, and taking the library defaults instead left the
	// ground with neither. A retained handle is what lets the caller apply the
	// group through the same path every other shape uses.
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = friction;
	shapeDef.baseMaterial.restitution = restitution;

	return RetainShape( world, b3CreateHeightFieldShape( world->bodies[body], &shapeDef, field ) );
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
						 float friction, float restitution, int group, int isSensor )
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
	shapeDef.isSensor = isSensor != 0;
	shapeDef.enableSensorEvents = isSensor != 0;
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

/**
 * How hard a joint holds its bodies together.
 *
 * Box3D's joints are SOFT constraints and default to 60 Hz
 * (`src/joint.c:30`), which lets a linkage deviate visibly from its ideal
 * geometry — the strandbeest scenario runs its pivots at 240 for exactly this
 * reason. The engine clamps to a quarter of the substep rate internally
 * (`joint.c:1466`), so asking for more than is solvable is safe.
 */
void b3w_set_joint_constraint_tuning( int handle, int joint, float hertz, float dampingRatio )
{
	b3Joint_SetConstraintTuning( JointId( handle, joint ), hertz, dampingRatio );
}


/**
 * Wake the two bodies a joint connects.
 *
 * Box3D's joint setters only write their value — `b3RevoluteJoint_SetMotorSpeed`
 * and `SetTargetAngle` both just assign — so a machine that has settled and
 * gone to sleep never acts on a new command. Rapier's setters wake by default,
 * which is why the same build drives there and sits still here.
 *
 * Exposed separately rather than folded into the setters because waking on
 * EVERY command means nothing ever sleeps: the caller knows whether the command
 * actually asks for motion, and only it can tell.
 */
void b3w_wake_joint_bodies( int handle, int joint )
{
	b3Joint_WakeBodies( JointId( handle, joint ) );
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
						  float friction, float restitution, int group, int isSensor )
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
	def.isSensor = isSensor != 0;
	def.enableSensorEvents = isSensor != 0;

	b3Sphere sphere = { .center = { ox, oy, oz }, .radius = radius };
	return RetainShape( world, b3CreateSphereShape( world->bodies[body], &def, &sphere ) );
}

/// A cylinder about local Y, matching how the game orients wheels.
int b3w_add_cylinder_shape( int handle, int body, float halfHeight, float radius, int sides, float density,
							 float friction, float restitution, int group, int isSensor )
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
	def.isSensor = isSensor != 0;
	def.enableSensorEvents = isSensor != 0;

	b3HullData* hull = b3CreateCylinder( 2.0f * halfHeight, radius, -halfHeight, sides );
	RetainHull( world, hull );
	return RetainShape( world, b3CreateHullShape( world->bodies[body], &def, hull ) );
}

/// A convex hull over a point cloud the host uploaded.
int b3w_add_hull_shape( int handle, int body, int pointsPtr, int pointCount, int maxVertices, float density,
						 float friction, float restitution, int group, int isSensor )
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
	def.isSensor = isSensor != 0;
	def.enableSensorEvents = isSensor != 0;

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

/**
 * Contact stiffness, as a spring frequency in Hz.
 *
 * Box3D's default is 30 Hz, sized for metre-scale bodies. A block here is
 * 0.25 m, and a contact spring lets an impact sink roughly v/(2*pi*f) before it
 * pushes back — at 3 m/s that is 17 mm, which is 7% of a block and plainly
 * visible as one block dipping into another before popping out.
 *
 * Static contacts already get double this (`physics_world.c:1123`), which is
 * why a block landing on TERRAIN looks right and a block landing on another
 * BLOCK does not. Exposed so the game can state a stiffness that matches its
 * own scale instead of inheriting one that matches nobody's.
 *
 * The solver caps the effective value at an eighth of the substep rate, so
 * there is no way to ask for something it cannot resolve.
 */
void b3w_set_contact_tuning( int handle, float hertz, float dampingRatio, float pushSpeed )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return;
	}

	b3World_SetContactTuning( world->id, hertz, dampingRatio, pushSpeed );
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

/**
 * The ceiling on every awake body's linear speed.
 *
 * Not just a clamp on impulses: the solver reads it in IntegratePositions
 * (`src/solver.c:182`) and applies it to every awake body every substep, so the
 * default of 400 m/s is a hard limit on how fast anything in the world can
 * travel by any route. Games that fly need to say otherwise.
 */
void b3w_set_maximum_linear_speed( int handle, float speed )
{
	World* world = GetWorld( handle );
	if ( world == NULL || speed <= 0.0f )
	{
		return;
	}

	b3World_SetMaximumLinearSpeed( world->id, speed );
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

/**
 * Tag a body with the structure that owns it.
 *
 * A vehicle's wheels may not stand on bodies sharing their owner. This replaced
 * a per-tick upload of a capped list of body indices — the cap silently dropped
 * exclusions past 64 bodies, so a big enough structure started using its own
 * bodies as ground. Ownership is a property of the body, so it is stored on the
 * body and compared in O(1) during the wheel cast.
 *
 * 0 means unowned and always drivable (terrain). 0xFFFFFFFF means never ground
 * for anyone, which is what the player is.
 */
void b3w_body_set_owner( int handle, int body, unsigned int owner )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return;
	}

	world->bodyOwner[body] = (uint32_t)owner;
}

/** The owner of a body, by index. 0 when unowned. Read by the wheel cast. */
uint32_t b3w_owner_of_body( int handle, int body )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return 0;
	}

	return world->bodyOwner[body];
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

int b3w_vehicle_create( b3WorldId world, int worldHandle, b3BodyId chassis );
void b3w_vehicle_prepare( void* vehicle, float dt );
void b3w_vehicle_substep( float h, void* context );
void* b3w_vehicle_get( int handle );

int b3w_create_vehicle( int handle, int chassis )
{
	World* world = GetWorld( handle );
	if ( world == NULL || chassis < 0 || chassis >= world->bodyCount )
	{
		return -1;
	}
	return b3w_vehicle_create( world->id, handle, world->bodies[chassis] );
}

/**
 * The once-per-tick half: shape casts and wheel transforms. The force model
 * itself now runs per substep, driven by the world's substep callback — see
 * `b3w_vehicle_substep`. The JS API is unchanged: this is still called once
 * before `b3w_step`.
 */
void b3w_step_vehicle( int vehicle, float dt )
{
	void* v = b3w_vehicle_get( vehicle );
	if ( v != NULL )
	{
		b3w_vehicle_prepare( v, dt );
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
/**
 * The speed below which a body may fall asleep.
 *
 * Box3D defaults to 0.05 m/s and Rapier to 0.4 m/s — eight times stricter. A
 * parked vehicle's suspension leaves a residual jitter that sits between the
 * two, so the same car sleeps under Rapier and never qualifies under Box3D,
 * and a body that never sleeps integrates that residual into a creep: measured
 * at 14.8 m of drift over 30 s on a 1 degree slope, against Rapier's 0.39 m.
 */
void b3w_set_sleep_threshold( int handle, int body, float threshold )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount || threshold < 0.0f )
	{
		return;
	}

	b3Body_SetSleepThreshold( world->bodies[body], threshold );
}

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

/**
 * Flatten a body's contact manifolds into `out`, returning how many were
 * written (capped at `capacity`).
 *
 * 7 floats per manifold: normal xyz (pointing AWAY from the queried body),
 * deepest penetration depth, normal impulse, tangent impulse, other body index.
 *
 * The normal is flipped here when the queried body is shape B, because Box3D
 * reports A->B and the caller must not have to know which end it is on: a
 * chassis pushed along an inverted normal moves INTO what it hit.
 */
int b3w_body_contacts( int handle, int body, float* out, int capacity )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount || capacity <= 0 )
	{
		return 0;
	}

	b3BodyId id = world->bodies[body];
	int count = b3Body_GetContactCapacity( id );
	if ( count <= 0 )
	{
		return 0;
	}
	if ( count > 64 )
	{
		count = 64;
	}

	b3ContactData data[64];
	count = b3Body_GetContactData( id, data, count );

	int written = 0;
	for ( int i = 0; i < count && written < capacity; i++ )
	{
		b3BodyId bodyA = b3Shape_GetBody( data[i].shapeIdA );
		bool queriedIsA = B3_ID_EQUALS( bodyA, id );
		b3BodyId otherId = queriedIsA ? b3Shape_GetBody( data[i].shapeIdB ) : bodyA;
		float sign = queriedIsA ? 1.0f : -1.0f;

		intptr_t tagged = (intptr_t)b3Body_GetUserData( otherId );
		int otherIndex = tagged > 0 ? (int)( tagged - 1 ) : -1;

		for ( int m = 0; m < data[i].manifoldCount && written < capacity; m++ )
		{
			const b3Manifold* manifold = data[i].manifolds + m;
			if ( manifold->pointCount == 0 )
			{
				continue;
			}

			float depth = 0.0f;
			float normalImpulse = 0.0f;
			for ( int p = 0; p < manifold->pointCount; p++ )
			{
				float separation = manifold->points[p].separation;
				if ( separation < 0.0f && -separation > depth )
				{
					depth = -separation;
				}
				normalImpulse += manifold->points[p].normalImpulse;
			}

			float* o = out + written * 7;
			o[0] = manifold->normal.x * sign;
			o[1] = manifold->normal.y * sign;
			o[2] = manifold->normal.z * sign;
			o[3] = depth;
			o[4] = normalImpulse;
			o[5] = b3Length( manifold->frictionImpulse );
			o[6] = (float)otherIndex;
			written++;
		}
	}

	return written;
}

/** A body's world AABB, as min xyz then max xyz. */
void b3w_body_aabb( int handle, int body, float* out )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return;
	}

	b3AABB aabb = b3Body_ComputeAABB( world->bodies[body] );
	out[0] = (float)aabb.lowerBound.x;
	out[1] = (float)aabb.lowerBound.y;
	out[2] = (float)aabb.lowerBound.z;
	out[3] = (float)aabb.upperBound.x;
	out[4] = (float)aabb.upperBound.y;
	out[5] = (float)aabb.upperBound.z;
}

/** World-space angular inertia tensor, row-major 3x3 into `out`. */
void b3w_body_inertia_tensor( int handle, int body, float* out )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return;
	}

	b3BodyId id = world->bodies[body];
	b3MassData md = b3Body_GetMassData( id );
	b3Quat q = b3Body_GetRotation( id );

	// R * I_local * R^T, one column at a time.
	b3Vec3 basis[3] = { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };
	for ( int c = 0; c < 3; c++ )
	{
		b3Vec3 local = b3InvRotateVector( q, basis[c] );
		b3Vec3 mapped = b3RotateVector( q, b3MulMV( md.inertia, local ) );
		out[c] = mapped.x;
		out[3 + c] = mapped.y;
		out[6 + c] = mapped.z;
	}
}

/** Per-shape sensor flag and world position: 4 floats each (isSensor, xyz). */
int b3w_body_shape_states( int handle, int body, float* out, int capacity )
{
	World* world = GetWorld( handle );
	if ( world == NULL || body < 0 || body >= world->bodyCount )
	{
		return 0;
	}

	b3BodyId id = world->bodies[body];
	int count = b3Body_GetShapeCount( id );
	if ( count > capacity )
	{
		count = capacity;
	}
	if ( count <= 0 )
	{
		return 0;
	}

	b3ShapeId shapes[256];
	int max = count > 256 ? 256 : count;
	b3Body_GetShapes( id, shapes, max );

	for ( int i = 0; i < max; i++ )
	{
		// Box3D exposes no local offset accessor on a shape, so the position is
		// its world AABB centre. That is the shape's centre rather than the
		// authored origin — the two differ for an off-centre hull — and this is
		// used to locate a collider, not to reconstruct its transform.
		b3AABB aabb = b3Shape_GetAABB( shapes[i] );

		out[i * 4 + 0] = b3Shape_IsSensor( shapes[i] ) ? 1.0f : 0.0f;
		out[i * 4 + 1] = (float)( ( aabb.lowerBound.x + aabb.upperBound.x ) * 0.5 );
		out[i * 4 + 2] = (float)( ( aabb.lowerBound.y + aabb.upperBound.y ) * 0.5 );
		out[i * 4 + 3] = (float)( ( aabb.lowerBound.z + aabb.upperBound.z ) * 0.5 );
	}

	return max;
}

/**
 * A sphere cast that skips bodies, for the same reason the ray version does.
 *
 * The player's ground probe sweeps a sphere from inside its own capsule, so
 * without exclusion the first thing it finds is the player — and the movement
 * code that reads the result decides it is standing on itself.
 */
typedef struct
{
	ClosestHit closest;
	b3BodyId exclude;
} FilteredCastHit;

static float OnFilteredCastHit( b3ShapeId shapeId, b3Pos point, b3Vec3 normal, float fraction, uint64_t userMaterialId,
								int triangleIndex, int childIndex, void* context )
{
	(void)point;
	(void)userMaterialId;
	(void)triangleIndex;
	(void)childIndex;

	FilteredCastHit* ctx = (FilteredCastHit*)context;
	if ( B3_IS_NON_NULL( ctx->exclude ) && B3_ID_EQUALS( b3Shape_GetBody( shapeId ), ctx->exclude ) )
	{
		return -1.0f;
	}

	ctx->closest.fraction = fraction;
	ctx->closest.normal = normal;
	ctx->closest.hit = true;
	return fraction;
}

int b3w_cast_sphere_filtered( int handle, float cx, float cy, float cz, float radius, float dx, float dy, float dz,
							  int exclude, float* out )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return 0;
	}

	b3Vec3 point = { 0.0f, 0.0f, 0.0f };
	b3ShapeProxy proxy = { .points = &point, .count = 1, .radius = radius };

	FilteredCastHit ctx = { .closest = { .fraction = 1.0f, .normal = { 0.0f, 1.0f, 0.0f }, .hit = false },
							.exclude = ( exclude >= 0 && exclude < world->bodyCount ) ? world->bodies[exclude]
																					 : b3_nullBodyId };

	b3World_CastShape( world->id, ( b3Pos ){ cx, cy, cz }, &proxy, ( b3Vec3 ){ dx, dy, dz }, b3DefaultQueryFilter(),
					   OnFilteredCastHit, &ctx );

	if ( ctx.closest.hit == false )
	{
		return 0;
	}

	out[0] = ctx.closest.normal.x;
	out[1] = ctx.closest.normal.y;
	out[2] = ctx.closest.normal.z;
	out[3] = ctx.closest.fraction;
	return 1;
}

// --- debug wireframe -----------------------------------------------------
//
// The game's F3 view. Rapier shipped `world.debugRender()`; Box3D's own
// `b3World_Draw` hands shapes back through `DrawShapeFcn` as an opaque
// *application* object it expects the caller to have registered, so there is
// nothing to draw unless the app already built renderable shapes. This walks
// the world instead and emits line segments from the engine's OWN shape data —
// which is the point of a physics debug view: it must show what the solver
// believes, not what the renderer believes, or it cannot disagree with the
// picture and is worth nothing.
//
// Line list, two vertices per segment, matching what the host feeds a
// THREE.LineSegments: xyz per vertex, rgba per vertex.

typedef struct DebugSink
{
	float* verts;
	float* colors;
	int capacityVerts;
	int count;
	float r, g, b;
} DebugSink;

static void PushVertex( DebugSink* sink, b3Vec3 p )
{
	if ( sink->count >= sink->capacityVerts )
	{
		return;
	}

	sink->verts[sink->count * 3 + 0] = (float)p.x;
	sink->verts[sink->count * 3 + 1] = (float)p.y;
	sink->verts[sink->count * 3 + 2] = (float)p.z;
	sink->colors[sink->count * 4 + 0] = sink->r;
	sink->colors[sink->count * 4 + 1] = sink->g;
	sink->colors[sink->count * 4 + 2] = sink->b;
	sink->colors[sink->count * 4 + 3] = 1.0f;
	sink->count++;
}

static void PushSegment( DebugSink* sink, b3Vec3 a, b3Vec3 b )
{
	PushVertex( sink, a );
	PushVertex( sink, b );
}

/// World-space point of a body-local point.
static b3Vec3 ToWorld( b3WorldTransform t, b3Vec3 local )
{
	b3Vec3 r = b3RotateVector( t.q, local );
	return B3_LITERAL( b3Vec3 ){ (float)t.p.x + r.x, (float)t.p.y + r.y, (float)t.p.z + r.z };
}

/// A circle of `segments` lines around `centre`, spanning the two given axes.
static void PushCircle( DebugSink* sink, b3WorldTransform t, b3Vec3 centre, b3Vec3 axisA, b3Vec3 axisB, float radius,
						int segments )
{
	b3Vec3 previous;
	for ( int i = 0; i <= segments; i++ )
	{
		float angle = ( 2.0f * 3.14159265f * (float)i ) / (float)segments;
		float ca = cosf( angle ) * radius;
		float sa = sinf( angle ) * radius;
		b3Vec3 local = { centre.x + axisA.x * ca + axisB.x * sa, centre.y + axisA.y * ca + axisB.y * sa,
						 centre.z + axisA.z * ca + axisB.z * sa };
		b3Vec3 world = ToWorld( t, local );
		if ( i > 0 )
		{
			PushSegment( sink, previous, world );
		}
		previous = world;
	}
}

/**
 * Every shape in the world as a wireframe, in world space.
 *
 * `excludeBody` drops one body — the ground. Its collider is a mesh the size of
 * the island, and drawing it buries every machine in a haystack of triangles;
 * Rapier's version excluded it for the same reason. Mesh shapes are skipped
 * outright for that reason too.
 *
 * Colour carries body STATE, because "is the physics doing anything" is the
 * question this view exists to answer: grey static, blue kinematic, green awake
 * dynamic, red asleep. A machine that stops moving should go red, and one that
 * is asleep when it should not be is then visible at a glance.
 *
 * Returns the number of VERTICES written; the host reads 3 floats of position
 * and 4 of colour per vertex. Writing stops at capacity rather than growing:
 * the buffer is allocated once by the host and a debug view that reallocates
 * every frame would be its own performance bug.
 */
int b3w_debug_draw( int handle, float* verts, float* colors, int capacityVerts, int excludeBody )
{
	World* world = GetWorld( handle );
	if ( world == NULL || verts == NULL || colors == NULL )
	{
		return 0;
	}

	DebugSink sink = { verts, colors, capacityVerts, 0, 1.0f, 1.0f, 1.0f };

	for ( int i = 0; i < world->bodyCount; ++i )
	{
		if ( i == excludeBody )
		{
			continue;
		}

		b3BodyId bodyId = world->bodies[i];
		if ( b3Body_IsValid( bodyId ) == false )
		{
			continue;
		}

		b3BodyType type = b3Body_GetType( bodyId );
		bool awake = b3Body_IsAwake( bodyId );
		if ( type == b3_staticBody )
		{
			sink.r = 0.55f, sink.g = 0.55f, sink.b = 0.55f;
		}
		else if ( type == b3_kinematicBody )
		{
			sink.r = 0.35f, sink.g = 0.65f, sink.b = 1.0f;
		}
		else if ( awake )
		{
			sink.r = 0.35f, sink.g = 0.95f, sink.b = 0.45f;
		}
		else
		{
			sink.r = 0.95f, sink.g = 0.35f, sink.b = 0.35f;
		}

		b3WorldTransform transform = b3Body_GetTransform( bodyId );

		b3ShapeId shapes[256];
		int shapeCount = b3Body_GetShapeCount( bodyId );
		if ( shapeCount > 256 )
		{
			shapeCount = 256;
		}
		b3Body_GetShapes( bodyId, shapes, shapeCount );

		for ( int s = 0; s < shapeCount; ++s )
		{
			b3ShapeType shapeType = b3Shape_GetType( shapes[s] );

			if ( shapeType == b3_sphereShape )
			{
				b3Sphere sphere = b3Shape_GetSphere( shapes[s] );
				b3Vec3 x = { 1.0f, 0.0f, 0.0f }, y = { 0.0f, 1.0f, 0.0f }, z = { 0.0f, 0.0f, 1.0f };
				PushCircle( &sink, transform, sphere.center, x, y, sphere.radius, 16 );
				PushCircle( &sink, transform, sphere.center, y, z, sphere.radius, 16 );
				PushCircle( &sink, transform, sphere.center, x, z, sphere.radius, 16 );
			}
			else if ( shapeType == b3_capsuleShape )
			{
				b3Capsule capsule = b3Shape_GetCapsule( shapes[s] );
				b3Vec3 x = { 1.0f, 0.0f, 0.0f }, y = { 0.0f, 1.0f, 0.0f }, z = { 0.0f, 0.0f, 1.0f };
				PushCircle( &sink, transform, capsule.center1, x, z, capsule.radius, 12 );
				PushCircle( &sink, transform, capsule.center2, x, z, capsule.radius, 12 );
				b3Vec3 c1 = capsule.center1, c2 = capsule.center2;
				for ( int e = 0; e < 4; ++e )
				{
					b3Vec3 axis = ( e == 0 ) ? x : ( e == 1 ) ? z : ( e == 2 ) ? B3_LITERAL( b3Vec3 ){ -1, 0, 0 }
																			  : B3_LITERAL( b3Vec3 ){ 0, 0, -1 };
					b3Vec3 a = { c1.x + axis.x * capsule.radius, c1.y + axis.y * capsule.radius,
								 c1.z + axis.z * capsule.radius };
					b3Vec3 b = { c2.x + axis.x * capsule.radius, c2.y + axis.y * capsule.radius,
								 c2.z + axis.z * capsule.radius };
					PushSegment( &sink, ToWorld( transform, a ), ToWorld( transform, b ) );
				}
			}
			else if ( shapeType == b3_meshShape )
			{
				/*
				 * Terrain, and anything else built from triangles.
				 *
				 * Absent until now, so F3 drew every collider EXCEPT the ground —
				 * which is the one place a wrong collider is invisible from inside
				 * the game. A day was spent guessing whether terrain colliders
				 * existed while the overlay could have answered it in a second.
				 *
				 * One edge per triangle side, drawn from the mesh's own vertex and
				 * triangle arrays. Terrain is thousands of triangles per chunk, so
				 * the vertex budget is what stops this: the sink drops what will
				 * not fit rather than overrunning it.
				 */
				b3Mesh mesh = b3Shape_GetMesh( shapes[s] );
				const b3MeshData* data = mesh.data;
				if ( data != NULL )
				{
					const uint8_t* base = (const uint8_t*)data;
					const b3Vec3* verts = (const b3Vec3*)( base + data->vertexOffset );
					const b3MeshTriangle* tris = (const b3MeshTriangle*)( base + data->triangleOffset );

					for ( int t = 0; t < data->triangleCount; ++t )
					{
						b3Vec3 a = verts[tris[t].index1];
						b3Vec3 b = verts[tris[t].index2];
						b3Vec3 c = verts[tris[t].index3];

						a.x *= mesh.scale.x, a.y *= mesh.scale.y, a.z *= mesh.scale.z;
						b.x *= mesh.scale.x, b.y *= mesh.scale.y, b.z *= mesh.scale.z;
						c.x *= mesh.scale.x, c.y *= mesh.scale.y, c.z *= mesh.scale.z;

						PushSegment( &sink, ToWorld( transform, a ), ToWorld( transform, b ) );
						PushSegment( &sink, ToWorld( transform, b ), ToWorld( transform, c ) );
						PushSegment( &sink, ToWorld( transform, c ), ToWorld( transform, a ) );
					}
				}
			}
			else if ( shapeType == b3_hullShape )
			{
				// Boxes are hulls here, so this covers every block and every
				// authored collider. Half-edges come in twin pairs; emitting only
				// the one with the lower index draws each edge once.
				const b3HullData* hull = b3Shape_GetHull( shapes[s] );
				if ( hull == NULL )
				{
					continue;
				}

				const b3Vec3* points = b3GetHullPoints( hull );
				const b3HullHalfEdge* edges = b3GetHullEdges( hull );
				for ( int e = 0; e < hull->edgeCount; ++e )
				{
					if ( edges[e].twin < e )
					{
						continue;
					}

					b3Vec3 a = points[edges[e].origin];
					b3Vec3 b = points[edges[edges[e].twin].origin];
					PushSegment( &sink, ToWorld( transform, a ), ToWorld( transform, b ) );
				}
			}
			// Mesh shapes fall through: see the note above.
		}
	}

	return sink.count;
}
