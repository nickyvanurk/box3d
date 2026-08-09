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

typedef struct World
{
	b3WorldId id;
	bool inUse;

	b3BodyId* bodies;
	int bodyCount;
	int bodyCapacity;

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

void b3w_add_box_shape( int handle, int body, float hx, float hy, float hz, float density, float friction,
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
