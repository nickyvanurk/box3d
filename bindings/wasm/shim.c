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

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/em_js.h>
#endif

// --- the heap ------------------------------------------------------------
//
// The wasm heap has a ceiling (2 GB), and an allocation past it returns NULL.
// Nothing here used to look: a NULL from `malloc` was written through, so a
// full heap surfaced as "memory access out of bounds" inside whatever happened
// to allocate next, or as a refusal that named the wrong cause ("its shape
// registry is full"). Every allocation in this binding — the shim's, the
// vehicle code's, Box3D's own through `b3SetAllocator`, and the buffers the
// host asks for — goes through `CheckedAlloc` / `CheckedRealloc`, and running
// out is ONE error that says what happened.
//
// It is thrown, not returned. Box3D cannot report a failed allocation — it
// asserts and writes through it — so a return code would cover only the
// allocations this file makes. A JavaScript exception unwinds every wasm frame
// between here and the host call. The shim's own tables are left consistent
// (it allocates before it mutates, and claims a table slot before it creates
// the shape or joint that will fill it), and nothing it still held is lost
// (see "held allocations"), but a throw from inside Box3D can leave the
// world mid-operation, so a host that carries on after this error is carrying
// on with a world that may refuse or mis-answer from then on. Whether to carry
// on is the host's decision; this error is what makes it an informed one.

#ifdef __EMSCRIPTEN__
// clang-format off
EM_JS( void, b3w_throw_heap_exhausted, ( double requested ), {
	throw new Error( "box3d: wasm heap exhausted (" + Math.round( HEAPU8.length / 1048576 ) + " MB, asked for " +
					 requested + " more bytes)" );
} );
// clang-format on
#else
static void b3w_throw_heap_exhausted( double requested )
{
	fprintf( stderr, "box3d: heap exhausted (asked for %.0f more bytes)\n", requested );
	abort();
}
#endif

// --- held allocations ----------------------------------------------------
//
// The throw unwinds every C frame without running any of them, so whatever a
// call still owned when the heap ran out would be lost: a scratch array, or a
// hull not yet cloned into a shape. Such an allocation is HELD for exactly that
// window, and running out releases every held one before it throws.
//
// Only what nothing else can ever point at is held. Mesh and height-field
// geometry is not: the shape create takes its reference before it allocates.
// That rule is also what keeps a stale entry — left by a trap that unwound
// without passing here — safe to release on a later throw.

typedef struct Held
{
	void* p;
	void ( *release )( void* );
} Held;

// A call nests at most three windows (a body's shapes, a map grow, a mesh);
// one past the end is simply not held, so the worst case is the old leak.
#define HELD_MAX 8
static Held g_held[HELD_MAX];
static int g_heldCount = 0;

static void* Hold( void* p, void ( *release )( void* ) )
{
	if ( p != NULL && g_heldCount < HELD_MAX )
	{
		g_held[g_heldCount++] = ( Held ){ p, release };
	}
	return p;
}

/** Ownership has passed on (or the call is done with it): no longer released on a throw. */
static void Unhold( void* p )
{
	for ( int i = g_heldCount - 1; i >= 0; --i )
	{
		if ( g_held[i].p == p )
		{
			g_held[i] = g_held[--g_heldCount];
			return;
		}
	}
}

static void RunOutOfHeap( double requested )
{
	while ( g_heldCount > 0 )
	{
		Held held = g_held[--g_heldCount];
		held.release( held.p );
	}
	b3w_throw_heap_exhausted( requested );
}

void* CheckedAlloc( size_t size )
{
	void* p = malloc( size );
	if ( p == NULL && size > 0 )
	{
		RunOutOfHeap( (double)size );
	}
	return p;
}

void* CheckedRealloc( void* old, size_t size )
{
	void* p = realloc( old, size );
	if ( p == NULL && size > 0 )
	{
		RunOutOfHeap( (double)size );
	}
	return p;
}

/** Scratch for one call: freed by `FreeHeld`, or by a throw if the heap runs out first. */
static void* AllocHeld( size_t size )
{
	return Hold( CheckedAlloc( size ), free );
}

static void FreeHeld( void* p )
{
	Unhold( p );
	free( p );
}

// A hull not yet given to a shape. Safe to release even from inside the shape
// create: Box3D clones a hull into its database before it allocates anything
// else, so no shape ever points at this one.
static void ReleaseHull( void* p )
{
	b3DestroyHull( (b3HullData*)p );
}

/** Box3D's allocator: its default is `aligned_alloc` with nobody checking. */
static void* AllocForBox3D( int32_t size, int32_t alignment )
{
	void* p = aligned_alloc( (size_t)alignment, (size_t)size );
	if ( p == NULL && size > 0 )
	{
		RunOutOfHeap( (double)size );
	}
	return p;
}

static void FreeForBox3D( void* mem )
{
	free( mem );
}

// --- handle tables -------------------------------------------------------
//
// Box3D ids are {index, world, generation} structs. The host holds an int32
// index into these tables instead, so nothing about Box3D's id layout leaks
// across the boundary and a stale handle fails a bounds check rather than
// addressing a recycled body.

/*
 * "I cannot answer that", distinct from any real answer.
 *
 * A query about a handle this world does not own used to return 0, which is
 * also a real answer everywhere it was used: a body that does not exist read as
 * ASLEEP, a cast into a destroyed world read as a MISS, a body with no shapes
 * and a body that is not there were the same number. The caller then decided
 * something on a fact nobody established.
 *
 * Callers must treat this as a third state. Choosing to fall back to a
 * particular answer is fine — silently inheriting one is what this replaces.
 */
#define B3W_UNKNOWN ( -1 )

/*
 * Sixteen, not four.
 *
 * A synchronisation lab gives every simulated client its own world — that is
 * what makes it a model of N machines rather than one — so six players plus a
 * reference world wants seven. Four was silently exceeded: the fifth
 * `b3w_create_world` returns -1 and every call through that handle is then
 * ignored, so bodies appear to spawn and read back as zeros.
 */
#define MAX_WORLDS 16

typedef struct World
{
	b3WorldId id;
	bool inUse;

	/*
	 * BODIES: a handle is an identity, a slot is storage, and they are two
	 * different numbers.
	 *
	 * A HANDLE is issued once, in creation order, and never reused. The host
	 * depends on both halves of that: a handle held across a rebuild must find
	 * nothing rather than the body that replaced it, and a structure's bodies,
	 * created together, carry consecutive handles that a ray filter excludes as
	 * one range (`b3w_cast_ray_filtered`).
	 *
	 * A SLOT is where a live body's state is kept, and is reused through a free
	 * list. The table used to be indexed by handle, so it held a row for every
	 * body ever created — a terrain chunk re-mesh is a new body, so a drive
	 * across the map grew it without bound, and every walk over it (transform
	 * readback, the detect-only reset) paid for the dead. Now it is the size of
	 * the high-water mark of bodies alive at once.
	 *
	 * `handleSlot` maps one to the other (open addressing, linear probing,
	 * backward-shift deletion so no tombstones accumulate). The engine-side
	 * body's userData holds `slot + 1`, so the per-pair callbacks read their
	 * state directly and only the host boundary ever hashes.
	 *
	 * Every per-body array below is indexed by SLOT.
	 */
	b3BodyId* bodies;
	/** The handle living in each slot, or -1 for a free slot. */
	int32_t* slotHandle;
	/** Free slots, as a stack. */
	int32_t* freeSlots;
	int freeSlotCount;
	/** Slots ever used: the table's length. */
	int bodySlotCount;
	/** Slots allocated. */
	int bodyCapacity;
	/** The next handle to issue. */
	int nextBodyHandle;
	/** Live bodies. */
	int liveBodyCount;
	int32_t* mapKeys;
	int32_t* mapSlots;
	/** A power of two, at least twice the live count. */
	int mapCapacity;

	/**
	 * Which structure owns each body, parallel to `bodies`.
	 *
	 * NOT in the body's userData: that slot already holds the body INDEX, which
	 * every raycast reads back to say what it hit. Storing the owner there
	 * silently rewrote every query result — four raycast suites went red at
	 * once. One slot, one meaning.
	 */
	uint32_t* bodyOwner;
	/**
	 * Bodies whose contacts are DETECTED but never solved — the vehicle chassis.
	 *
	 * The game resolves these positionally after the step (`depenetrateChassis`)
	 * so a machine SLIDES along what it hits instead of being stopped by an
	 * impulse. Rapier did this through filterContactPair returning EMPTY; this is
	 * the same thing through Box3D's pre-solve callback, which returns false to
	 * drop a contact from the solver while still reporting it.
	 */
	uint8_t* detectOnly;
	/** The one body each body never collides with, as a HANDLE (-1: none): a
	 *  told rider and its parent. Checked by the custom pair filter, so two
	 *  riders of the same parent still collide with each other, which a shared
	 *  negative group index could not express. A handle rather than a slot so
	 *  that the parent's slot being reused can never exclude a stranger. */
	int32_t* excludedFrom;
	/** Bodies placed from a network stream. Two of them never solve a contact
	 *  between them: their owners resolve it, and the solver shoving one out
	 *  of the other here only fights the next placement. */
	uint8_t* placed;
	/**
	 * Bodies whose mass is NOT recomputed per shape, parallel to `bodies`.
	 *
	 * Box3D re-derives a body's mass from every shape on it each time one is
	 * added, which is O(shapes) per add and O(shapes²) to build a body — a
	 * grid of 3,600 cells cost a quarter of a second in that alone. A caller
	 * attaching a batch defers the update and applies it once at the end.
	 */
	uint8_t* massDeferred;
	/**
	 * Bodies whose mass, centre and inertia the caller stated outright
	 * (`b3w_set_body_mass_data`), parallel to `bodies`. Their shapes carry
	 * collision only: adding or destroying one never re-derives the mass from
	 * shapes, it re-applies the stated mass so the extents follow the shapes.
	 */
	uint8_t* massStated;
	b3MassData* statedMass;

	/*
	 * Joints are handed back as handles for the same reasons shapes are, and
	 * they got there the same way: this was a fixed array of 4096 with a
	 * bump allocator, `b3w_destroy_joint` freed nothing, and a machine rebuilds
	 * every pivot, piston and weld on every block the player places. Past 4096
	 * every add returned -1, the host stored it as a live joint, and
	 * `b3Array_Get( ..., -1 )` reads the word before the array — which traps in
	 * about one heap layout in forty and silently corrupts the other
	 * thirty-nine.
	 *
	 * A handle is `generation << 21 | slot`, so a handle held across the reuse
	 * of its slot resolves to null rather than to the joint that took its
	 * place. That is what lets slots be recycled without breaking the identity
	 * the host's motor cache keys on.
	 *
	 * Reclaim is a LAZY SWEEP for slots whose joint the engine no longer
	 * recognises, not a free list, because the dominant teardown path never
	 * tells us: the host drops its joint refs on a rebuild without calling
	 * destroy, and destroying a BODY takes its joints with it inside Box3D.
	 * Asking the engine what is still alive is the only answer that covers all
	 * three. `b3w_destroy_joint` still nulls its slot, which just makes that
	 * slot visible to the sweep immediately.
	 */
	b3JointId* joints;
	/** Per-slot generation, bumped whenever a slot is reused. */
	uint16_t* jointGen;
	/** Slots allocated. Grows on demand; never shrinks while the world lives. */
	int jointCapacity;
	int jointCount;
	/** Where the last reclaim sweep stopped, so a full table stays amortised. */
	int jointRover;


	/*
	 * Shapes are handed back as handles so callers can destroy or re-filter one
	 * without holding a Box3D id.
	 *
	 * A handle is `generation << 16 | slot`, NOT a bare index. Slots used to be
	 * append-only and never reused, so that a stale handle failed a validity
	 * check rather than addressing someone else's collider — a real hazard, and
	 * the generation is what keeps that guarantee now that slots ARE reused.
	 *
	 * They have to be reused, because append-only made this a LEAK with a wall
	 * at the end of it: nothing here is freed when a shape dies, and a machine
	 * rebuilds its colliders on every block the player places. A 76-wheel tank
	 * costs 646 slots per rebuild, so it exhausted all 65536 after about a
	 * hundred placements — and past that every collider came back -1, which the
	 * host stored as a live handle. The machine silently lost its collision.
	 *
	 * Reclaim is a LAZY SWEEP for slots whose shape is no longer valid, not a
	 * free list, because the dominant teardown path never tells us: destroying
	 * a BODY takes its shapes with it and `b3w_destroy_shape` is never called.
	 * Asking the engine what is still alive is the only answer that covers both.
	 */
	b3ShapeId* shapes;
	/** Per-slot generation, bumped whenever a slot is reused. Reset with the
	 *  world (`b3w_destroy_world` clears the whole struct) — a handle held from
	 *  a destroyed world is caught by the slot's null shape id instead. */
	uint16_t* shapeGen;
	/*
	 * GEOMETRY. Who owns a shape's data decides when it can be freed, and the
	 * answer differs by kind — see `OwnedGeometry`:
	 *
	 *  - A hull is CLONED by Box3D into its world's ref-counted hull database
	 *    (`b3AddHullToDatabase`), so the shim's own hull is freed the moment the
	 *    shape exists. It used to be kept per slot until the lazy sweep happened
	 *    to recycle that slot, which held a dead copy for nothing.
	 *  - A mesh or a height field is REFERENCED, not copied (box3d.h's warning
	 *    on `b3CreateMeshShape`). The shim made one per shape and never freed
	 *    any of them: every re-meshed terrain chunk, seam and scenery tile
	 *    leaked its triangles, ~47.5 bytes each, until the heap hit its 2 GB
	 *    ceiling and allocations started failing. The shim owns these, and
	 *    frees them when their shape dies — on every path a shape can die by.
	 */
	/** Slots allocated. Grows on demand; never shrinks while the world lives. */
	int shapeCapacity;
	int shapeCount;
	/** Where the last reclaim sweep stopped, so a full registry stays amortised. */
	int shapeRover;

	// pos(3) + quat(4) per body, interleaved, in SLOT order.
	float* transforms;
} World;

/**
 * Resolve a body slot to a live id, or null. Defined below with the rest of the
 * per-body operations; declared here because shape creation and the bulk
 * transform read both sit above it, and both must refuse a destroyed body.
 */
static b3BodyId BodyOf( World* world, int body );
static bool CustomFilterExclusion( b3ShapeId shapeIdA, b3ShapeId shapeIdB, void* context );
static bool PreSolveDetectOnly( b3ShapeId shapeIdA, b3ShapeId shapeIdB, b3Pos point, b3Vec3 normal, void* context );
static b3BodyId BodyId( int handle, int body );

static World g_worlds[MAX_WORLDS];

static World* GetWorld( int handle )
{
	if ( handle < 0 || handle >= MAX_WORLDS || g_worlds[handle].inUse == false )
	{
		return NULL;
	}

	return &g_worlds[handle];
}

/** Grow the per-slot arrays to hold `needed` slots, doubling. */
static void ReserveBodySlots( World* world, int needed )
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

	// Every field a new slot needs is written when a body takes it, so the
	// tails are left as they come.
	world->bodies = (b3BodyId*)CheckedRealloc( world->bodies, sizeof( b3BodyId ) * capacity );
	world->slotHandle = (int32_t*)CheckedRealloc( world->slotHandle, sizeof( int32_t ) * capacity );
	world->freeSlots = (int32_t*)CheckedRealloc( world->freeSlots, sizeof( int32_t ) * capacity );
	world->transforms = (float*)CheckedRealloc( world->transforms, sizeof( float ) * 7 * capacity );
	world->bodyOwner = (uint32_t*)CheckedRealloc( world->bodyOwner, sizeof( uint32_t ) * capacity );
	world->detectOnly = (uint8_t*)CheckedRealloc( world->detectOnly, sizeof( uint8_t ) * capacity );
	world->excludedFrom = (int32_t*)CheckedRealloc( world->excludedFrom, sizeof( int32_t ) * capacity );
	world->placed = (uint8_t*)CheckedRealloc( world->placed, sizeof( uint8_t ) * capacity );
	world->massDeferred = (uint8_t*)CheckedRealloc( world->massDeferred, sizeof( uint8_t ) * capacity );
	world->massStated = (uint8_t*)CheckedRealloc( world->massStated, sizeof( uint8_t ) * capacity );
	world->statedMass = (b3MassData*)CheckedRealloc( world->statedMass, sizeof( b3MassData ) * capacity );
	world->bodyCapacity = capacity;
}

/** Handles are dense and sequential, so a multiplicative hash spreads them. */
static int MapHome( World* world, int32_t handle )
{
	return (int)( ( (uint32_t)handle * 2654435761u ) & (uint32_t)( world->mapCapacity - 1 ) );
}

/** The slot a handle lives in, or -1: destroyed, never issued, or malformed. */
static int SlotOf( World* world, int handle )
{
	if ( world == NULL || handle < 0 || world->mapCapacity == 0 )
	{
		return -1;
	}

	int mask = world->mapCapacity - 1;
	for ( int i = MapHome( world, handle );; i = ( i + 1 ) & mask )
	{
		int32_t key = world->mapKeys[i];
		if ( key == handle )
		{
			return world->mapSlots[i];
		}
		if ( key < 0 )
		{
			return -1;
		}
	}
}

static void MapInsert( World* world, int32_t handle, int32_t slot );

/** Keep the map at most half full, so a probe is short and always ends. */
static void MapReserve( World* world, int live )
{
	if ( world->mapCapacity >= 2 * live && world->mapCapacity > 0 )
	{
		return;
	}

	int32_t* oldKeys = world->mapKeys;
	int32_t* oldSlots = world->mapSlots;
	int oldCapacity = world->mapCapacity;

	int capacity = oldCapacity == 0 ? 512 : oldCapacity;
	while ( capacity < 2 * live )
	{
		capacity *= 2;
	}

	// Both allocated before either is installed, so a full heap leaves the old map whole.
	int32_t* keys = (int32_t*)AllocHeld( sizeof( int32_t ) * (size_t)capacity );
	int32_t* slots = (int32_t*)CheckedAlloc( sizeof( int32_t ) * (size_t)capacity );
	Unhold( keys );
	memset( keys, 0xFF, sizeof( int32_t ) * (size_t)capacity );
	world->mapKeys = keys;
	world->mapSlots = slots;
	world->mapCapacity = capacity;

	for ( int i = 0; i < oldCapacity; ++i )
	{
		if ( oldKeys[i] >= 0 )
		{
			MapInsert( world, oldKeys[i], oldSlots[i] );
		}
	}
	free( oldKeys );
	free( oldSlots );
}

static void MapInsert( World* world, int32_t handle, int32_t slot )
{
	int mask = world->mapCapacity - 1;
	int i = MapHome( world, handle );
	while ( world->mapKeys[i] >= 0 )
	{
		i = ( i + 1 ) & mask;
	}
	world->mapKeys[i] = handle;
	world->mapSlots[i] = slot;
}

/**
 * Remove a handle, shifting later entries of its probe run back into the gap.
 * Leaves no tombstone, so a world that churns bodies for hours probes exactly
 * as far as one that never destroyed any.
 */
static void MapRemove( World* world, int32_t handle )
{
	int mask = world->mapCapacity - 1;
	int i = MapHome( world, handle );
	while ( world->mapKeys[i] != handle )
	{
		if ( world->mapKeys[i] < 0 )
		{
			return;
		}
		i = ( i + 1 ) & mask;
	}

	for ( int j = ( i + 1 ) & mask; world->mapKeys[j] >= 0; j = ( j + 1 ) & mask )
	{
		// The entry at j may move into the gap at i only if its home does not
		// lie in the cyclic range (i, j] — otherwise a lookup would stop at i.
		int home = MapHome( world, world->mapKeys[j] );
		bool homeInRange = i <= j ? ( i < home && home <= j ) : ( i < home || home <= j );
		if ( homeInRange )
		{
			continue;
		}
		world->mapKeys[i] = world->mapKeys[j];
		world->mapSlots[i] = world->mapSlots[j];
		i = j;
	}
	world->mapKeys[i] = -1;
}

/**
 * The slot the next body will take, with every allocation it needs already
 * made — and nothing about the table changed yet. Split from `CommitBodySlot`
 * so that Box3D's own create, which can run out of heap too, happens between
 * the two: a throw there leaves no slot or handle that nothing owns.
 */
static int ReserveNextBodySlot( World* world )
{
	ReserveBodySlots( world, world->bodySlotCount + 1 );
	MapReserve( world, world->liveBodyCount + 1 );
	return world->freeSlotCount > 0 ? world->freeSlots[world->freeSlotCount - 1] : world->bodySlotCount;
}

/** Take the slot `ReserveNextBodySlot` named and issue its handle. Allocates nothing. */
static int32_t CommitBodySlot( World* world, int slot )
{
	if ( world->freeSlotCount > 0 && world->freeSlots[world->freeSlotCount - 1] == slot )
	{
		world->freeSlotCount--;
	}
	else
	{
		world->bodySlotCount++;
	}

	int32_t handle = world->nextBodyHandle++;
	MapInsert( world, handle, slot );
	world->slotHandle[slot] = handle;
	world->liveBodyCount++;
	return handle;
}

static void ReleaseBodySlot( World* world, int slot )
{
	MapRemove( world, world->slotHandle[slot] );
	world->slotHandle[slot] = -1;
	world->bodies[slot] = b3_nullBodyId;
	world->freeSlots[world->freeSlotCount++] = slot;
	world->liveBodyCount--;
}

/** The handle a live body's slot holds, for anything reported to the host. */
static int HandleOfSlot( World* world, int slot )
{
	if ( slot < 0 || slot >= world->bodySlotCount )
	{
		return -1;
	}
	return world->slotHandle[slot];
}

// --- shape geometry ownership --------------------------------------------

/**
 * The data a shape REFERENCES and the shim therefore owns, or NULL.
 *
 * Only meshes and height fields: Box3D clones a hull into its own database and
 * copies spheres and capsules by value. Every mesh and height field shape in a
 * world is made by `b3w_add_mesh_shape` / `b3w_add_height_field`, each over
 * data of its own, so a shape's geometry is exactly one owner's.
 */
static void* OwnedGeometry( b3ShapeId shape, b3ShapeType* type )
{
	*type = b3Shape_GetType( shape );
	if ( *type == b3_meshShape )
	{
		return (void*)b3Shape_GetMesh( shape ).data;
	}
	if ( *type == b3_heightShape )
	{
		return (void*)b3Shape_GetHeightField( shape );
	}
	return NULL;
}

static void FreeGeometry( b3ShapeType type, void* data )
{
	if ( data == NULL )
	{
		return;
	}
	if ( type == b3_meshShape )
	{
		b3DestroyMesh( (b3MeshData*)data );
	}
	else if ( type == b3_heightShape )
	{
		b3DestroyHeightField( (b3HeightFieldData*)data );
	}
}

/** Hand geometry to the shape just made over it, or free it now if the engine
 *  made no shape: then nothing references it and nothing else will free it. */
static b3ShapeId AttachOwned( b3ShapeId shape, b3ShapeType type, void* data )
{
	if ( B3_IS_NULL( shape ) || !b3Shape_IsValid( shape ) )
	{
		FreeGeometry( type, data );
	}
	return shape;
}

/** Destroy one shape and free what it referenced — after, never before: the
 *  destroy still reads the geometry to take the shape out of the broadphase. */
static void DestroyShapeAndGeometry( b3ShapeId shape, bool updateBodyMass )
{
	if ( B3_IS_NULL( shape ) || !b3Shape_IsValid( shape ) )
	{
		return;
	}
	b3ShapeType type;
	void* data = OwnedGeometry( shape, &type );
	b3DestroyShape( shape, updateBodyMass );
	FreeGeometry( type, data );
}

typedef struct OwnedShapeData
{
	b3ShapeType type;
	void* data;
} OwnedShapeData;

/**
 * Destroy a body and free the geometry of every shape on it.
 *
 * Box3D takes a body's shapes with it and tells nobody, which is why the
 * geometry is gathered first. This is the dominant way a shape dies here — a
 * terrain chunk is re-meshed by destroying its body — so it is the path that
 * leaked.
 */
static void DestroyBodyAndGeometry( b3BodyId body )
{
	int count = b3Body_GetShapeCount( body );
	b3ShapeId stackShapes[16];
	b3ShapeId* shapes = count <= 16 ? stackShapes : (b3ShapeId*)AllocHeld( sizeof( b3ShapeId ) * (size_t)count );
	count = b3Body_GetShapes( body, shapes, count );

	// Only the owned entries, packed to the front: read before the body goes,
	// freed after it.
	OwnedShapeData stackOwned[16];
	OwnedShapeData* owned =
		count <= 16 ? stackOwned : (OwnedShapeData*)AllocHeld( sizeof( OwnedShapeData ) * (size_t)count );
	int ownedCount = 0;
	for ( int i = 0; i < count; ++i )
	{
		OwnedShapeData* o = &owned[ownedCount];
		o->data = OwnedGeometry( shapes[i], &o->type );
		ownedCount += o->data != NULL ? 1 : 0;
	}

	b3DestroyBody( body );

	for ( int i = 0; i < ownedCount; ++i )
	{
		FreeGeometry( owned[i].type, owned[i].data );
	}

	if ( shapes != stackShapes )
	{
		FreeHeld( shapes );
	}
	if ( owned != stackOwned )
	{
		FreeHeld( owned );
	}
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

	// Before Box3D's first allocation; idempotent, so every world may say it.
	b3SetAllocator( AllocForBox3D, FreeForBox3D );

	World* world = &g_worlds[handle];
	world->id = b3CreateWorld( &def );

	// The vehicle force model runs per substep, not per tick — see
	// `b3w_vehicle_substep`. Registered once here so no caller has to know.
	// The context is WHICH world is stepping: the vehicle registry is global, so
	// the callback needs it to solve this world's vehicles and no others.
	b3World_SetSubStepCallback( world->id, b3w_vehicle_substep, (void*)(uintptr_t)b3StoreWorldId( world->id ) );
	b3World_SetCustomFilterCallback( world->id, CustomFilterExclusion, world );
	// From the start: a tyre's contact rule lives here, and a world with no
	// detect-only or placed body would otherwise never have one installed.
	b3World_SetPreSolveCallback( world->id, PreSolveDetectOnly, world );
	world->inUse = true;
	world->jointCount = 0;
	world->jointCapacity = 0;
	world->jointRover = 0;
	world->shapeCount = 0;

	// The rest of the body table starts zeroed: a destroyed world clears its
	// whole struct, and the array of them is static.
	ReserveBodySlots( world, 256 );
	MapReserve( world, 1 );

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

	// The world takes every shape with it, so what they referenced is gathered
	// first. Every mesh and height field is in the shape table: the shim makes
	// them all, and one it could not register it destroyed on the spot.
	OwnedShapeData* owned =
		(OwnedShapeData*)CheckedAlloc( sizeof( OwnedShapeData ) * (size_t)( world->shapeCount > 0 ? world->shapeCount : 1 ) );
	int ownedCount = 0;
	for ( int i = 0; i < world->shapeCount; ++i )
	{
		if ( B3_IS_NULL( world->shapes[i] ) || !b3Shape_IsValid( world->shapes[i] ) )
		{
			continue;
		}
		OwnedShapeData* o = &owned[ownedCount];
		o->data = OwnedGeometry( world->shapes[i], &o->type );
		ownedCount += o->data != NULL ? 1 : 0;
	}

	b3DestroyWorld( world->id );

	for ( int i = 0; i < ownedCount; ++i )
	{
		FreeGeometry( owned[i].type, owned[i].data );
	}
	free( owned );

	// EVERY array `ReserveBodySlots` grows, not just some — `bodyOwner` and
	// `detectOnly` once leaked on every world teardown, found under ASan.
	free( world->joints );
	free( world->jointGen );
	free( world->shapes );
	free( world->shapeGen );
	free( world->bodies );
	free( world->slotHandle );
	free( world->freeSlots );
	free( world->mapKeys );
	free( world->mapSlots );
	free( world->transforms );
	free( world->bodyOwner );
	free( world->detectOnly );
	free( world->excludedFrom );
	free( world->placed );
	free( world->massDeferred );
	free( world->massStated );
	free( world->statedMass );
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

	int slot = ReserveNextBodySlot( world );
	world->bodies[slot] = b3CreateBody( world->id, &def );
	int32_t handleOut = CommitBodySlot( world, slot );
	world->bodyOwner[slot] = 0;
	world->detectOnly[slot] = 0;
	world->excludedFrom[slot] = -1;
	world->placed[slot] = 0;
	world->massDeferred[slot] = 0;
	world->massStated[slot] = 0;

	// The SLOT rides on the body, so a contact callback reaches the body's state
	// without a lookup, and a query result names it through `slotHandle`.
	// Offset by one so that a body whose user data was never set — a vehicle's
	// wheel — reads as "unknown" rather than as slot zero.
	b3Body_SetUserData( world->bodies[slot], (void*)(intptr_t)( slot + 1 ) );

	return handleOut;
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
/*
 * A handle packs `generation << SHAPE_SLOT_BITS | slot` into a positive int32
 * — 31 usable bits, split 21/10.
 *
 * The table grows now, so the SLOT FIELD is what would otherwise cap it: 16
 * bits held the world to 65,536 shapes however large the array got, which is
 * the same fixed ceiling one level down. 21 bits is 2,097,152 slots, about
 * 3,200 copies of the largest machine anyone has built, all alive at once.
 *
 * That leaves 10 bits of generation, so a slot must be reused 1024 times
 * before a stale handle could alias a live one. The rover walks the whole
 * table between revisits, so that is 1024 x capacity allocations — tens of
 * millions. The old 15 bits of generation bought headroom nobody could use
 * against a ceiling we were actually reaching.
 */
#define SHAPE_SLOT_BITS 21
#define SHAPE_SLOT_MASK 0x1FFFFF
/** 10 bits of generation above 21 bits of slot: a handle is at most 0x7FFFFFFF,
 *  so it is always a positive int32 and a negative value is never a real one.
 *  The trade the width buys: a slot recycled 1024 times re-issues a handle a
 *  stale reference could still hold, and that aliasing would be SILENT. At 21
 *  bits of slot that needs ~262k creations landing on one slot, which no
 *  session reaches — but it is a bound, not an impossibility. */
#define SHAPE_GEN_MASK 0x3FF

/**
 * Make room for one more shape slot, doubling. False only when the slot field
 * cannot address another one (a full heap throws — see the heap).
 *
 * Generations are PRESERVED across a grow: realloc keeps the existing ones and
 * only the new tail is zeroed. A generation that reset would hand a stale
 * handle exactly the number it was waiting for.
 */
static bool ReserveShapeSlot( World* world )
{
	if ( world->shapeCount < world->shapeCapacity )
	{
		return true;
	}

	int prev = world->shapeCapacity;
	int next = prev > 0 ? prev * 2 : 256;
	if ( next > SHAPE_SLOT_MASK + 1 )
	{
		next = SHAPE_SLOT_MASK + 1;
	}
	if ( next <= prev )
	{
		return false;
	}

	world->shapes = (b3ShapeId*)CheckedRealloc( world->shapes, (size_t)next * sizeof( b3ShapeId ) );
	world->shapeGen = (uint16_t*)CheckedRealloc( world->shapeGen, (size_t)next * sizeof( uint16_t ) );

	memset( world->shapes + prev, 0, (size_t)( next - prev ) * sizeof( b3ShapeId ) );
	memset( world->shapeGen + prev, 0, (size_t)( next - prev ) * sizeof( uint16_t ) );

	world->shapeCapacity = next;
	return true;
}

/** The body's mass is re-derived from its shapes on each add or destroy. */
static bool MassFromShapes( World* world, int body )
{
	if ( body < 0 || body >= world->bodySlotCount )
	{
		return true;
	}
	return world->massDeferred[body] == 0 && world->massStated[body] == 0;
}

/** Bring the body's mass up to date: the stated mass if there is one, else its shapes'. */
static void ApplyMass( World* world, int body )
{
	b3BodyId id = BodyOf( world, body );
	if ( B3_IS_NULL( id ) || !b3Body_IsValid( id ) )
	{
		return;
	}
	if ( world->massStated[body] != 0 && b3Body_GetType( id ) == b3_dynamicBody )
	{
		b3Body_SetMassData( id, world->statedMass[body] );
	}
	else
	{
		b3Body_ApplyMassFromShapes( id );
	}
}

static int BodyIndexOfShape( b3ShapeId shapeId );

/**
 * The slot the next shape will take, with any growth it needs already done, or
 * -1 when the slot field cannot address another shape (a full heap throws).
 *
 * Claimed BEFORE the shape is created and filled by `RetainShape` after, which
 * allocates nothing: a throw while the table grows then leaves no shape in the
 * world that the table does not know of — one nothing could ever destroy.
 * Nothing about the table is taken here; an unused claim costs nothing.
 *
 * Recycling is tried BEFORE growing, deliberately: it holds the table at the
 * high-water mark of what is actually alive rather than at everything ever
 * created, which is the whole difference between this and the append-only
 * version that leaked.
 */
static int ClaimShapeSlot( World* world )
{
	if ( world->shapeCount < world->shapeCapacity )
	{
		return world->shapeCount;
	}

	// Full. Sweep for a slot whose shape has since died, resuming where the
	// last sweep stopped so a steady state costs about one probe.
	for ( int n = 0; n < world->shapeCapacity; n++ )
	{
		int i = world->shapeRover;
		world->shapeRover = ( i + 1 ) % world->shapeCapacity;
		if ( !b3Shape_IsValid( world->shapes[i] ) )
		{
			return i;
		}
	}

	// Every slot holds a LIVE shape, so the table has to get bigger.
	return ReserveShapeSlot( world ) ? world->shapeCount : -1;
}

/** Record a shape in the slot `ClaimShapeSlot` named and return its handle. Allocates nothing. */
static int RetainShape( World* world, int slot, b3ShapeId shape )
{
	if ( !B3_IS_NULL( shape ) && b3Shape_IsValid( shape ) )
	{
		int body = BodyIndexOfShape( shape );
		if ( body >= 0 && body < world->bodySlotCount && world->massStated[body] != 0 && world->massDeferred[body] == 0 )
		{
			ApplyMass( world, body );
		}
	}

	if ( slot == world->shapeCount )
	{
		world->shapeCount++;
	}
	else
	{
		// Reusing the slot invalidates every handle that pointed at it.
		world->shapeGen[slot] = (uint16_t)( ( world->shapeGen[slot] + 1 ) & SHAPE_GEN_MASK );
	}

	world->shapes[slot] = shape;
	return (int)( ( (uint32_t)world->shapeGen[slot] << SHAPE_SLOT_BITS ) | (uint32_t)slot );
}

/**
 * Resolve a shape handle, or NULL if it is stale, out of range, or malformed.
 *
 * The generation check is the whole point: a handle held across the reuse of
 * its slot resolves to NULL here rather than to the collider that took its
 * place.
 */
static b3ShapeId* ResolveShape( World* world, int handle )
{
	if ( world == NULL || handle < 0 )
	{
		return NULL;
	}

	int slot = handle & SHAPE_SLOT_MASK;
	int generation = ( handle >> SHAPE_SLOT_BITS ) & SHAPE_GEN_MASK;

	if ( slot >= world->shapeCount || world->shapeGen[slot] != (uint16_t)generation )
	{
		return NULL;
	}

	return &world->shapes[slot];
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
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return -1;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.updateBodyMass = MassFromShapes( world, body );
	def.enableCustomFiltering = true;
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
	b3BodyId bodyId = BodyOf( world, body );
	if ( B3_IS_NULL( bodyId ) )
	{
		return -1;
	}

	int slot = ClaimShapeSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	return RetainShape( world, slot, b3CreateHullShape( bodyId, &def, &hull.base ) );
}

/** Handles ever issued: one past the newest. Never falls — a handle is never
 *  reused — so it bounds a handle, and counts nothing alive. */
int b3w_body_count( int handle )
{
	World* world = GetWorld( handle );
	return world == NULL ? 0 : world->nextBodyHandle;
}

/** Bodies that EXIST. A gauge that wants a population reads this one. */
int b3w_live_body_count( int handle )
{
	World* world = GetWorld( handle );
	return world == NULL ? 0 : world->liveBodyCount;
}

/** Rows in the body table: the most bodies alive at once. What the table
 *  costs, where `b3w_body_count` is what was ever made. */
int b3w_body_slot_count( int handle )
{
	World* world = GetWorld( handle );
	return world == NULL ? 0 : world->bodySlotCount;
}

/** The transform row a body's pose is written to by `b3w_read_transforms`, or
 *  -1 for a body that does not exist. */
int b3w_body_slot( int handle, int body )
{
	return SlotOf( GetWorld( handle ), body );
}

int b3w_is_awake( int handle, int body )
{
	b3BodyId id = BodyId( handle, body );
	if ( B3_IS_NULL( id ) )
	{
		// NOT 0. A body that is gone is not "asleep", and the host read it that
		// way at three call sites — one of which asks the inverse question.
		return B3W_UNKNOWN;
	}

	return b3Body_IsAwake( id ) ? 1 : 0;
}

/**
 * Why a body is not sleeping: `out[0]` the velocity the sleep test judged it by
 * last step, `out[1]` its threshold, `out[2]` how long it has been continuously
 * sleepy. A body whose sleep time never grows is being woken by something every
 * step, and the velocity says whether it is moving or being pushed.
 */
void b3w_body_sleep_state( int handle, int body, float* out )
{
	out[0] = 0.0f;
	out[1] = 0.0f;
	out[2] = 0.0f;

	b3BodyId id = BodyId( handle, body );
	if ( B3_IS_NULL( id ) )
	{
		return;
	}

	out[0] = b3Body_GetSleepVelocity( id );
	out[1] = b3Body_GetSleepThreshold( id );
	out[2] = b3Body_GetSleepTime( id );
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
/// reads 7 floats per body straight out of the heap, at the body's SLOT
/// (`b3w_body_slot`) — `b3w_body_slot_count` rows. Slots are handed out in
/// handle order until the first body is destroyed, so a world that never
/// destroys one reads row = handle.
void b3w_read_transforms( int handle )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return;
	}

	float* out = world->transforms;
	for ( int i = 0; i < world->bodySlotCount; ++i )
	{
		// A free slot is written as a finite identity rather than skipped, so
		// the buffer never holds garbage a stale reader could act on. It runs
		// every tick over every row, so it is the hottest place a dangling id
		// could be dereferenced — BodyOf refuses one.
		b3BodyId id = BodyOf( world, i );
		if ( B3_IS_NULL( id ) )
		{
			out[0] = out[1] = out[2] = 0.0f;
			out[3] = out[4] = out[5] = 0.0f;
			out[6] = 1.0f;
			out += 7;
			continue;
		}

		b3WorldTransform transform = b3Body_GetTransform( id );

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

/**
 * Resolve a body SLOT, or null if nothing lives there any more.
 *
 * Entry points take a handle and resolve it to a slot first (`SlotOf`, which
 * refuses a destroyed or never-issued handle); this is the second line, for a
 * slot. A freed slot is nulled on destroy, and a Box3D id the engine itself no
 * longer recognises is refused too. Handing a dangling id to Box3D reads freed
 * memory — and since the shipped wasm is built `-DNDEBUG`, there is no
 * complaint: the heap is quietly corrupted and the NEXT unrelated query traps.
 * That cost a flying machine its simulation ~1.5 s after take-off, and the trap
 * always named the innocent query rather than the stale handle.
 */
static b3BodyId BodyOf( World* world, int body )
{
	if ( world == NULL || body < 0 || body >= world->bodySlotCount )
	{
		return b3_nullBodyId;
	}

	// The ONE place that reads the slot directly. Everything else comes through
	// here, and this line calling BodyOf instead is not a compile error — it is
	// an infinite recursion that only a run reveals.
	b3BodyId id = world->bodies[body];
	if ( B3_IS_NULL( id ) || b3Body_IsValid( id ) == false )
	{
		return b3_nullBodyId;
	}

	return id;
}

/** A body by world handle and body HANDLE, the way the host names one. */
static b3BodyId BodyId( int handle, int body )
{
	World* world = GetWorld( handle );
	return BodyOf( world, SlotOf( world, body ) );
}

/**
 * Take the answer `BodyOf` already worked out.
 *
 * It decides validity once and returns null for a body that has been destroyed
 * or whose slot never held one — and then eighteen entry points passed that null
 * straight into Box3D anyway. What happens next is luck: `b3Body_GetTransform`
 * on a null id indexes an array from -1, which reads whatever is there and
 * usually returns zeros, until the day the address is outside the heap and the
 * whole tick dies with "memory access out of bounds". A player froze a machine
 * he was holding and that is what he got, from a read that had been wrong for
 * every rebuild before it and had simply landed somewhere harmless.
 *
 * A dead body reads as zero and ignores what it is told. The caller asked about
 * something that is not there; that is not a reason to kill the simulation.
 */
#define BODY_OR_RETURN( id, handle, body, ... )                                                                        \
	b3BodyId id = BodyId( handle, body );                                                                              \
	if ( B3_IS_NULL( id ) )                                                                                            \
	{                                                                                                                  \
		return __VA_ARGS__;                                                                                            \
	}

void b3w_body_transform( int handle, int body, float* out )
{
	out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = 0.0f;
	out[6] = 1.0f;
	BODY_OR_RETURN( id, handle, body )

	b3WorldTransform t = b3Body_GetTransform( id );

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
	BODY_OR_RETURN( id, handle, body )
	b3Body_SetTransform( id, ( b3Pos ){ x, y, z }, b3Body_GetRotation( id ) );
}

void b3w_set_rotation( int handle, int body, float qx, float qy, float qz, float qw )
{
	BODY_OR_RETURN( id, handle, body )
	b3Body_SetTransform( id, b3Body_GetPosition( id ), ( b3Quat ){ { qx, qy, qz }, qw } );
}

/** The body's centre of mass in the world: the point its linear velocity belongs to. */
void b3w_body_center( int handle, int body, float* out )
{
	out[0] = out[1] = out[2] = 0.0f;
	BODY_OR_RETURN( id, handle, body )
	b3Pos c = b3Body_GetWorldCenter( id );
	out[0] = (float)c.x;
	out[1] = (float)c.y;
	out[2] = (float)c.z;
}

void b3w_linear_velocity( int handle, int body, float* out )
{
	out[0] = out[1] = out[2] = 0.0f;
	BODY_OR_RETURN( id, handle, body )

	b3Vec3 v = b3Body_GetLinearVelocity( id );
	out[0] = v.x;
	out[1] = v.y;
	out[2] = v.z;
}

void b3w_angular_velocity( int handle, int body, float* out )
{
	out[0] = out[1] = out[2] = 0.0f;
	BODY_OR_RETURN( id, handle, body )

	b3Vec3 w = b3Body_GetAngularVelocity( id );
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
/**
 * The velocity change this body took from CONTACTS during the last step.
 *
 * Exposed because a replicated car needs to tell three things apart that all
 * arrive as one velocity: what its owner's driver did, what the world did to it,
 * and what hitting something did. Only the last is discovered locally and
 * contested between two machines, and it is the only one that must survive a
 * correction toward the owner's account.
 *
 * Measured, not modelled. The alternative in the client was to recompute the
 * exchange analytically after the fact — a sphere-radius model with one
 * restitution constant, against a solver that used boxes, wheels, friction and
 * the real manifold — so it restored a different quantity than the correction
 * removed. It is also the reason a client-side version could only ever be
 * linear: the angular half of a collision is not recoverable from a recomputed
 * normal impulse, and a car that is shoved is also spun.
 *
 * `totalNormalImpulse` is the accumulation across sub-steps rather than the
 * final sub-step's `normalImpulse`, which is what makes this the whole of what
 * the step applied.
 *
 * Writes six floats: linear velocity delta, then angular velocity delta.
 */
void b3w_body_contact_impulse( int handle, int body, float* out )
{
	out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = 0.0f;
	BODY_OR_RETURN( id, handle, body )

	int capacity = b3Body_GetContactCapacity( id );
	if ( capacity <= 0 )
	{
		return;
	}
	if ( capacity > 16 )
	{
		capacity = 16;
	}
	b3ContactData data[16];
	int count = b3Body_GetContactData( id, data, capacity );

	b3Vec3 linear = { 0.0f, 0.0f, 0.0f };
	b3Vec3 angular = { 0.0f, 0.0f, 0.0f };

	for ( int i = 0; i < count; i++ )
	{
		// THE NORMAL POINTS FROM A TO B, so which end of the contact this body
		// is decides the sign. Getting this wrong is silent: it reads as the
		// struck car being flung toward whatever hit it.
		b3ShapeId shapeA = data[i].shapeIdA;
		b3BodyId ownerA = b3Shape_GetBody( shapeA );
		bool isA = B3_ID_EQUALS( ownerA, id );
		float sign = isA ? -1.0f : 1.0f;

		// ONLY WHAT THE OWNER DOES NOT ALREADY KNOW ABOUT.
		//
		// A replica's correction tracks its owner's account, and that account
		// already contains everything the owner's own simulation did — including
		// resting on the ground. Re-adding this body's ground contact on top of
		// it double-counts a force the estimate has already paid for, which on a
		// slope is large and continuous rather than transient.
		//
		// What the owner cannot know is a collision with a car being simulated
		// HERE, a round trip before their snapshot can mention it. That is
		// dynamic against dynamic, so a contact with static or kinematic
		// geometry is not part of this term.
		b3BodyId other = isA ? b3Shape_GetBody( data[i].shapeIdB ) : ownerA;
		if ( b3Body_GetType( other ) != b3_dynamicBody )
		{
			continue;
		}

		for ( int m = 0; m < data[i].manifoldCount; m++ )
		{
			const b3Manifold* man = &data[i].manifolds[m];
			b3Vec3 n = man->normal;

			for ( int k = 0; k < man->pointCount; k++ )
			{
				const b3ManifoldPoint* pt = &man->points[k];
				float j = pt->totalNormalImpulse * sign;
				b3Vec3 impulse = { n.x * j, n.y * j, n.z * j };
				linear.x += impulse.x;
				linear.y += impulse.y;
				linear.z += impulse.z;

				// r x J, about this body's own centre of mass.
				b3Vec3 r = isA ? pt->anchorA : pt->anchorB;
				angular.x += r.y * impulse.z - r.z * impulse.y;
				angular.y += r.z * impulse.x - r.x * impulse.z;
				angular.z += r.x * impulse.y - r.y * impulse.x;
			}

			// Friction is central and already a linear impulse; twist is about
			// the normal. Both are part of what the contact did and leaving them
			// out would under-report a glancing hit, which is the case where the
			// tangential part is most of it.
			linear.x += man->frictionImpulse.x * sign;
			linear.y += man->frictionImpulse.y * sign;
			linear.z += man->frictionImpulse.z * sign;
			angular.x += ( man->rollingImpulse.x + n.x * man->twistImpulse ) * sign;
			angular.y += ( man->rollingImpulse.y + n.y * man->twistImpulse ) * sign;
			angular.z += ( man->rollingImpulse.z + n.z * man->twistImpulse ) * sign;
		}
	}

	// Reported as a VELOCITY change rather than an impulse, because that is what
	// the caller needs and the mass and the world inertia are both here.
	float mass = b3Body_GetMass( id );
	if ( mass > 0.0f )
	{
		out[0] = (float)( linear.x / mass );
		out[1] = (float)( linear.y / mass );
		out[2] = (float)( linear.z / mass );
	}
	b3Matrix3 invI = b3Body_GetWorldInverseRotationalInertia( id );
	b3Vec3 dw = b3MulMV( invI, angular );
	out[3] = (float)dw.x;
	out[4] = (float)dw.y;
	out[5] = (float)dw.z;
}

/**
 * Drive a KINEMATIC body to a pose over one step: the engine sets the velocity
 * that lands it there, so contacts see the motion instead of a teleport.
 */
void b3w_set_target_transform( int handle, int body, float x, float y, float z, float qx, float qy, float qz, float qw,
							   float dt )
{
	BODY_OR_RETURN( id, handle, body )
	b3WorldTransform target = { .p = ( b3Pos ){ x, y, z }, .q = ( b3Quat ){ qx, qy, qz, qw } };
	b3Body_SetTargetTransform( id, target, dt, true );
}

/** Lock roll and pitch, for a character that must stay on its feet; yaw stays free. */
void b3w_set_rotation_lock( int handle, int body, int locked )
{
	BODY_OR_RETURN( id, handle, body )
	b3MotionLocks locks = b3Body_GetMotionLocks( id );
	locks.angularX = locked != 0;
	locks.angularZ = locked != 0;
	b3Body_SetMotionLocks( id, locks );
	// Box3D re-derives the mass from shapes when the locks change.
	World* world = GetWorld( handle );
	int slot = SlotOf( world, body );
	if ( world->massStated[slot] != 0 )
	{
		ApplyMass( world, slot );
	}
}

void b3w_set_damping( int handle, int body, float linear, float angular )
{
	BODY_OR_RETURN( id, handle, body )
	b3Body_SetLinearDamping( id, linear );
	b3Body_SetAngularDamping( id, angular );
}

void b3w_set_linear_velocity( int handle, int body, float x, float y, float z, int wake )
{
	BODY_OR_RETURN( id, handle, body )
	bool wasAsleep = b3Body_IsAwake( id ) == false;

	b3Body_SetLinearVelocity( id, ( b3Vec3 ){ x, y, z } );

	if ( wake == 0 && wasAsleep )
	{
		b3Body_SetAwake( id, false );
	}
}

void b3w_set_angular_velocity( int handle, int body, float x, float y, float z, int wake )
{
	BODY_OR_RETURN( id, handle, body )
	bool wasAsleep = b3Body_IsAwake( id ) == false;

	b3Body_SetAngularVelocity( id, ( b3Vec3 ){ x, y, z } );

	if ( wake == 0 && wasAsleep )
	{
		b3Body_SetAwake( id, false );
	}
}

void b3w_apply_impulse( int handle, int body, float x, float y, float z )
{
	BODY_OR_RETURN( id, handle, body )
	b3Body_ApplyLinearImpulse( id, ( b3Vec3 ){ x, y, z }, b3Body_GetWorldCenter( id ), true );
}

void b3w_apply_force( int handle, int body, float x, float y, float z, int wake )
{
	BODY_OR_RETURN( id, handle, body )
	b3Body_ApplyForce( id, ( b3Vec3 ){ x, y, z }, b3Body_GetWorldCenter( id ), wake != 0 );
}

void b3w_apply_torque( int handle, int body, float x, float y, float z, int wake )
{
	BODY_OR_RETURN( id, handle, body )
	b3Body_ApplyTorque( id, ( b3Vec3 ){ x, y, z }, wake != 0 );
}

float b3w_mass( int handle, int body )
{
	BODY_OR_RETURN( id, handle, body, 0.0f )
	return b3Body_GetMass( id );
}

/// Diagonal of the local rotational inertia, which is what the force pipeline
/// reads to judge whether a torque could actually turn the body.
void b3w_principal_inertia( int handle, int body, float* out )
{
	out[0] = out[1] = out[2] = 0.0f;
	BODY_OR_RETURN( id, handle, body )

	b3Matrix3 I = b3Body_GetLocalRotationalInertia( id );
	out[0] = I.cx.x;
	out[1] = I.cy.y;
	out[2] = I.cz.z;
}

int b3w_body_type( int handle, int body )
{
	BODY_OR_RETURN( id, handle, body, 0 )
	return (int)b3Body_GetType( id );
}

void b3w_set_body_type( int handle, int body, int type )
{
	BODY_OR_RETURN( id, handle, body )
	b3Body_SetType( id, (b3BodyType)type );
	// Box3D re-derives the mass from shapes on a type change.
	World* world = GetWorld( handle );
	int slot = SlotOf( world, body );
	if ( world->massStated[slot] != 0 )
	{
		ApplyMass( world, slot );
	}
}

void b3w_set_gravity_scale( int handle, int body, float scale )
{
	BODY_OR_RETURN( id, handle, body )
	b3Body_SetGravityScale( id, scale );
}

void b3w_keep_awake( int handle, int body )
{
	BODY_OR_RETURN( id, handle, body )
	b3Body_KeepAwake( id );
}

void b3w_wake( int handle, int body )
{
	BODY_OR_RETURN( id, handle, body )
	b3Body_SetAwake( id, true );
}

int b3w_is_valid( int handle, int body )
{
	return B3_IS_NULL( BodyId( handle, body ) ) ? 0 : 1;
}

/// A capsule along local Y, the shape the player uses. `halfHeight` is the
/// distance from the centre to a hemisphere's centre, matching the game's
/// CapsuleColliderConfig rather than Box3D's two-point form.
int b3w_add_capsule_shape( int handle, int body, float halfHeight, float radius, float ox, float oy, float oz,
						   float density, float friction, float restitution, int isSensor )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return -1;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.updateBodyMass = MassFromShapes( world, body );
	def.enableCustomFiltering = true;
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
	b3BodyId bodyId = BodyOf( world, body );
	if ( B3_IS_NULL( bodyId ) )
	{
		return -1;
	}

	int slot = ClaimShapeSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	return RetainShape( world, slot, b3CreateCapsuleShape( bodyId, &def, &capsule ) );
}

// --- terrain and queries -------------------------------------------------

/// Allocate a float buffer in the wasm heap for the host to fill. Returned as a
/// byte offset; free with `free`. Throws rather than return 0 — see the heap.
int b3w_alloc_floats( int count )
{
	return (int)(intptr_t)CheckedAlloc( sizeof( float ) * (size_t)( count > 0 ? count : 1 ) );
}

/// The same for bytes: every buffer the host allocates in this heap comes from
/// one of these two, so none of them can be a 0 the host then writes through.
int b3w_alloc_bytes( int count )
{
	return (int)(intptr_t)CheckedAlloc( (size_t)( count > 0 ? count : 1 ) );
}

/**
 * Bytes the allocator has handed out and not had back.
 *
 * The heap's LENGTH only ever grows — wasm memory cannot shrink — so it is a
 * high-water mark and cannot tell a leak from a peak. This can: under steady
 * churn it is flat when everything allocated is also freed.
 */
int b3w_heap_in_use( void )
{
	struct mallinfo info = mallinfo();
	return (int)info.uordblks;
}

/**
 * Attach a height field to a body.
 *
 * The host passes heights as a pointer into the wasm heap it filled itself, so
 * a terrain upload is one crossing rather than one per sample. Box3D keeps a
 * REFERENCE to the height field (see b3CreateHeightFieldShape's warning), so
 * the field must outlive the shape: the shim owns it and frees it when the
 * shape dies (`OwnedGeometry`). The heights themselves are copied into the
 * field by `b3CreateHeightField`, so they are read straight from the host's
 * buffer.
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
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return -1;
	}

	// Resolved BEFORE the field is built, so a refusal has nothing to free.
	b3BodyId bodyId = BodyOf( world, body );
	if ( B3_IS_NULL( bodyId ) )
	{
		return -1;
	}

	int sampleCount = countX * countZ;
	const float* heights = (const float*)(intptr_t)heightsPtr;

	float minHeight = heights[0];
	float maxHeight = heights[0];
	for ( int i = 1; i < sampleCount; ++i )
	{
		if ( heights[i] < minHeight ) minHeight = heights[i];
		if ( heights[i] > maxHeight ) maxHeight = heights[i];
	}

	b3HeightFieldDef def = { 0 };
	def.heights = (float*)heights; // read only: b3CreateHeightField copies them
	def.materialIndices = NULL;
	def.scale = ( b3Vec3 ){ scaleX, scaleY, scaleZ };
	def.countX = countX;
	def.countZ = countZ;
	def.globalMinimumHeight = minHeight;
	def.globalMaximumHeight = maxHeight;

	int slot = ClaimShapeSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	// Not held, for the reason the mesh is not (see b3w_add_mesh_shape).
	b3HeightFieldData* field = b3CreateHeightField( &def );

	// Terrain is a surface the game tunes like any other: it states a friction
	// and a collision group, and taking the library defaults instead left the
	// ground with neither. A retained handle is what lets the caller apply the
	// group through the same path every other shape uses.
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.enableCustomFiltering = true;
	shapeDef.baseMaterial.friction = friction;
	shapeDef.baseMaterial.restitution = restitution;

	b3ShapeId shape = b3CreateHeightFieldShape( bodyId, &shapeDef, field );
	return RetainShape( world, slot, AttachOwned( shape, b3_heightShape, field ) );
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

/*
 * The same split the shape table uses, for the same reason: the slot field has
 * to address every joint a world can hold at once, and the generation only has
 * to outlast the reuse of one slot.
 */
#define JOINT_SLOT_BITS 21
#define JOINT_SLOT_MASK 0x1FFFFF
/** 10 bits of generation above 21 bits of slot: a handle is at most 0x7FFFFFFF,
 *  so it is always a positive int32 and a negative value is never a real one.
 *  The trade the width buys: a slot recycled 1024 times re-issues a handle a
 *  stale reference could still hold, and that aliasing would be SILENT. At 21
 *  bits of slot that needs ~262k creations landing on one slot, which no
 *  session reaches — but it is a bound, not an impossibility. */
#define JOINT_GEN_MASK 0x3FF

/**
 * Make room for one more joint slot, doubling. False only when the slot field
 * cannot address another one (a full heap throws — see the heap).
 *
 * Generations are PRESERVED across a grow: realloc keeps the existing ones and
 * only the new tail is zeroed. A generation that reset would hand a stale
 * handle exactly the number it was waiting for.
 */
static bool ReserveJointSlot( World* world )
{
	if ( world->jointCount < world->jointCapacity )
	{
		return true;
	}

	int prev = world->jointCapacity;
	int next = prev > 0 ? prev * 2 : 256;
	if ( next > JOINT_SLOT_MASK + 1 )
	{
		next = JOINT_SLOT_MASK + 1;
	}
	if ( next <= prev )
	{
		return false;
	}

	world->joints = (b3JointId*)CheckedRealloc( world->joints, (size_t)next * sizeof( b3JointId ) );
	world->jointGen = (uint16_t*)CheckedRealloc( world->jointGen, (size_t)next * sizeof( uint16_t ) );

	memset( world->joints + prev, 0, (size_t)( next - prev ) * sizeof( b3JointId ) );
	memset( world->jointGen + prev, 0, (size_t)( next - prev ) * sizeof( uint16_t ) );

	world->jointCapacity = next;
	return true;
}

/**
 * The slot the next joint will take, or -1 when the slot field cannot address
 * another one — claimed before the joint is created, for the reason
 * `ClaimShapeSlot` gives: a throw while the table grows must not leave a joint
 * constraining two bodies with no handle to remove it by.
 *
 * Recycling is tried BEFORE growing, deliberately: it holds the table at the
 * high-water mark of what is actually alive rather than at everything ever
 * created, which is the whole difference between this and the bump allocator
 * that leaked.
 */
static int ClaimJointSlot( World* world )
{
	if ( world->jointCount < world->jointCapacity )
	{
		return world->jointCount;
	}

	// Full. Sweep for a slot whose joint has since died, resuming where the
	// last sweep stopped so a steady state costs about one probe.
	for ( int n = 0; n < world->jointCapacity; n++ )
	{
		int i = world->jointRover;
		world->jointRover = ( i + 1 ) % world->jointCapacity;
		if ( !b3Joint_IsValid( world->joints[i] ) )
		{
			return i;
		}
	}

	// Every slot holds a LIVE joint, so the table has to get bigger.
	return ReserveJointSlot( world ) ? world->jointCount : -1;
}

/** Record a joint in the slot `ClaimJointSlot` named and return its handle. Allocates nothing. */
static int RetainJoint( World* world, int slot, b3JointId joint )
{
	if ( slot == world->jointCount )
	{
		world->jointCount++;
	}
	else
	{
		// Reusing the slot invalidates every handle that pointed at it.
		world->jointGen[slot] = (uint16_t)( ( world->jointGen[slot] + 1 ) & JOINT_GEN_MASK );
	}

	world->joints[slot] = joint;
	return (int)( ( (uint32_t)world->jointGen[slot] << JOINT_SLOT_BITS ) | (uint32_t)slot );
}

/**
 * Resolve a joint handle to its slot, or NULL if it is stale, out of range or
 * malformed. The generation check is the whole point: a handle held across the
 * reuse of its slot resolves to NULL here rather than to the joint that took
 * its place.
 */
static b3JointId* ResolveJoint( World* world, int handle )
{
	if ( world == NULL || handle < 0 )
	{
		return NULL;
	}

	int slot = handle & JOINT_SLOT_MASK;
	int generation = ( handle >> JOINT_SLOT_BITS ) & JOINT_GEN_MASK;

	if ( slot >= world->jointCount || world->jointGen[slot] != (uint16_t)generation )
	{
		return NULL;
	}

	return &world->joints[slot];
}

/**
 * Resolve a joint to a live id, or null.
 *
 * The counterpart of `BodyOf`, and it exists because joints were missed when
 * that one was written. A slot passing a bounds check does NOT mean the joint
 * exists: the host drops a joint on every rebuild, and Box3D destroys a body's
 * joints with the body. Handing a dead or null id to Box3D indexes an array
 * from -1 under `-DNDEBUG`, which is the same silent heap corruption
 * `BODY_OR_RETURN` was written to end.
 */
static b3JointId JointOf( World* world, int joint )
{
	b3JointId* slot = ResolveJoint( world, joint );
	if ( slot == NULL || b3Joint_IsValid( *slot ) == false )
	{
		return b3_nullJointId;
	}

	return *slot;
}

static b3JointId JointId( int handle, int joint )
{
	return JointOf( GetWorld( handle ), joint );
}

/**
 * Take the answer `JointOf` already worked out — the joint half of
 * `BODY_OR_RETURN`, and it is here for exactly the same reason.
 *
 * A dead joint reads as zero and ignores what it is told. The caller asked
 * about something that is not there; that is not a reason to kill the
 * simulation.
 */
#define JOINT_OR_RETURN( id, handle, joint, ... )                                                                      \
	b3JointId id = JointId( handle, joint );                                                                           \
	if ( B3_IS_NULL( id ) )                                                                                            \
	{                                                                                                                  \
		return __VA_ARGS__;                                                                                            \
	}

/** Joint slots ever allocated: the table's length, which never falls. */
int b3w_joint_count( int handle )
{
	World* world = GetWorld( handle );
	return world == NULL ? 0 : world->jointCount;
}

/**
 * Joints that EXIST, as opposed to slots ever handed out.
 *
 * The probe this bug needed and did not have. `jointCount` above is the
 * high-water mark of simultaneous liveness — with the slots recycled it stops
 * being a leak counter and becomes a budget — and the gap between the two is
 * how much of the table is reclaimable right now.
 */
int b3w_live_joint_count( int handle )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return 0;
	}

	int live = 0;
	for ( int i = 0; i < world->jointCount; ++i )
	{
		if ( b3Joint_IsValid( world->joints[i] ) )
		{
			++live;
		}
	}
	return live;
}

int b3w_add_revolute_joint( int handle, int bodyA, int bodyB, float ax, float ay, float az, float bx, float by,
							float bz, float axisX, float axisY, float axisZ )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return -1;
	}

	// A joint to a body that is not there is not a joint. Both ends are resolved
	// before anything is built, so a stale handle costs a joint rather than a
	// world holding a constraint onto nothing.
	b3BodyId idA = BodyId( handle, bodyA );
	b3BodyId idB = BodyId( handle, bodyB );
	if ( B3_IS_NULL( idA ) || B3_IS_NULL( idB ) )
	{
		return -1;
	}

	b3Vec3 axis = { axisX, axisY, axisZ };

	b3RevoluteJointDef def = b3DefaultRevoluteJointDef();
	def.base.bodyIdA = idA;
	def.base.bodyIdB = idB;
	def.base.localFrameA = FrameAlong( ( b3Vec3 ){ ax, ay, az }, AXIS_Z, axis );
	def.base.localFrameB = FrameAlong( ( b3Vec3 ){ bx, by, bz }, AXIS_Z, axis );

	int slot = ClaimJointSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	return RetainJoint( world, slot, b3CreateRevoluteJoint( world->id, &def ) );
}

/**
 * A motor joint: drag body B toward body A's frame with a bounded force.
 *
 * Exposed for a replicated body whose POSITION is dictated from outside the
 * solver while its RESPONSE stays the solver's. Placing such a body teleports
 * it, so it never travels between estimates and generates no contact; stepping
 * it from a remote's stale input makes it drift; making it kinematic stops it
 * being pushable at all. A motor is the fourth option and the only one that
 * separates the two: B remains a dynamic body that collides and can be shoved,
 * and a spring with a hertz, a damping ratio and a MAXIMUM FORCE pulls it
 * toward where it is supposed to be.
 *
 * The maximum force is the whole point. It is what lets a collision win locally
 * for as long as the collision lasts, and lets the correction win again once it
 * is over, without either being switched off.
 */
int b3w_add_motor_joint( int handle, int bodyA, int bodyB, float hertz, float dampingRatio, float maxForce,
						 float angularHertz, float angularDampingRatio, float maxTorque )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return -1;
	}

	b3BodyId idA = BodyId( handle, bodyA );
	b3BodyId idB = BodyId( handle, bodyB );
	if ( B3_IS_NULL( idA ) || B3_IS_NULL( idB ) )
	{
		return -1;
	}

	b3MotorJointDef def = b3DefaultMotorJointDef();
	def.base.bodyIdA = idA;
	def.base.bodyIdB = idB;
	def.linearHertz = hertz;
	def.linearDampingRatio = dampingRatio;
	def.maxSpringForce = maxForce;
	def.angularHertz = angularHertz;
	def.angularDampingRatio = angularDampingRatio;
	def.maxSpringTorque = maxTorque;

	int slot = ClaimJointSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	return RetainJoint( world, slot, b3CreateMotorJoint( world->id, &def ) );
}

int b3w_add_prismatic_joint( int handle, int bodyA, int bodyB, float ax, float ay, float az, float bx, float by,
							 float bz, float axisX, float axisY, float axisZ )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return -1;
	}

	// A joint to a body that is not there is not a joint. Both ends are resolved
	// before anything is built, so a stale handle costs a joint rather than a
	// world holding a constraint onto nothing.
	b3BodyId idA = BodyId( handle, bodyA );
	b3BodyId idB = BodyId( handle, bodyB );
	if ( B3_IS_NULL( idA ) || B3_IS_NULL( idB ) )
	{
		return -1;
	}

	b3Vec3 axis = { axisX, axisY, axisZ };

	b3PrismaticJointDef def = b3DefaultPrismaticJointDef();
	def.base.bodyIdA = idA;
	def.base.bodyIdB = idB;
	def.base.localFrameA = FrameAlong( ( b3Vec3 ){ ax, ay, az }, AXIS_X, axis );
	def.base.localFrameB = FrameAlong( ( b3Vec3 ){ bx, by, bz }, AXIS_X, axis );

	int slot = ClaimJointSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	return RetainJoint( world, slot, b3CreatePrismaticJoint( world->id, &def ) );
}

/**
 * Weld two bodies together where they are now.
 *
 * The joint frame is the world point (px, py, pz) with the world axes, and each
 * body's local frame is that point seen from the body — so the weld holds the
 * relative pose the bodies have at the moment of the call, whatever it is.
 * Both frames are derived here rather than passed in because "hold what you
 * are touching" is the only pose a caller ever means; a weld that had to be
 * given two local frames would recompute exactly this on the other side.
 *
 * Zero hertz is Box3D's "as rigid as the solver allows". Two bodies is the case
 * the docs say the approximate solver does hold; the warning is about chains.
 */
int b3w_add_weld_joint( int handle, int bodyA, int bodyB, float px, float py, float pz, float linearHertz,
						float angularHertz, float dampingRatio )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return -1;
	}

	b3BodyId idA = BodyId( handle, bodyA );
	b3BodyId idB = BodyId( handle, bodyB );
	if ( B3_IS_NULL( idA ) || B3_IS_NULL( idB ) )
	{
		return -1;
	}

	b3Transform anchor = { { px, py, pz }, b3Quat_identity };
	b3WorldTransform ta = b3Body_GetTransform( idA );
	b3WorldTransform tb = b3Body_GetTransform( idB );
	b3Transform fa = { { (float)ta.p.x, (float)ta.p.y, (float)ta.p.z }, ta.q };
	b3Transform fb = { { (float)tb.p.x, (float)tb.p.y, (float)tb.p.z }, tb.q };

	b3WeldJointDef def = b3DefaultWeldJointDef();
	def.base.bodyIdA = idA;
	def.base.bodyIdB = idB;
	def.base.localFrameA = b3InvMulTransforms( fa, anchor );
	def.base.localFrameB = b3InvMulTransforms( fb, anchor );
	def.linearHertz = linearHertz;
	def.angularHertz = angularHertz;
	def.linearDampingRatio = dampingRatio;
	def.angularDampingRatio = dampingRatio;

	int slot = ClaimJointSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	return RetainJoint( world, slot, b3CreateWeldJoint( world->id, &def ) );
}

/**
 * A ball joint between two bodies at the world point (px, py, pz), holding
 * the relative pose they have now — the tow-hitch counterpart of the weld
 * above: position pinned, every rotation free.
 */
int b3w_add_ball_joint( int handle, int bodyA, int bodyB, float px, float py, float pz )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return -1;
	}

	b3BodyId idA = BodyId( handle, bodyA );
	b3BodyId idB = BodyId( handle, bodyB );
	if ( B3_IS_NULL( idA ) || B3_IS_NULL( idB ) )
	{
		return -1;
	}

	b3Transform anchor = { { px, py, pz }, b3Quat_identity };
	b3WorldTransform ta = b3Body_GetTransform( idA );
	b3WorldTransform tb = b3Body_GetTransform( idB );
	b3Transform fa = { { (float)ta.p.x, (float)ta.p.y, (float)ta.p.z }, ta.q };
	b3Transform fb = { { (float)tb.p.x, (float)tb.p.y, (float)tb.p.z }, tb.q };

	b3SphericalJointDef def = b3DefaultSphericalJointDef();
	def.base.bodyIdA = idA;
	def.base.bodyIdB = idB;
	def.base.localFrameA = b3InvMulTransforms( fa, anchor );
	def.base.localFrameB = b3InvMulTransforms( fb, anchor );

	int slot = ClaimJointSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	return RetainJoint( world, slot, b3CreateSphericalJoint( world->id, &def ) );
}

/**
 * A distance joint between two bodies, anchored at two separate world points.
 *
 * Unlike the weld and ball joints above, the two anchors are NOT the same
 * point — a rope spans a gap, so each end is placed on its own body.
 *
 * The rope configuration is `hertz == 0` with the limit enabled: the spring
 * block is skipped entirely (`distance_joint.c` guards it on `hertz > 0`), and
 * the upper-limit block does the work. That block is already exactly a rope —
 * it clamps its impulse at zero, so it pulls the bodies together once they are
 * further apart than `maxLength` and does nothing at all below it. Slack below
 * the limit, hard stop at it, and no compression by construction.
 *
 * A spring rope (a bungee) is the same call with `hertz > 0`. Note the sign
 * convention if you go there: `axis` runs A to B and tension is the NEGATIVE
 * impulse, so `lowerSpringForce` caps tension and `upperSpringForce` caps
 * compression — set the upper to zero for a spring that can only pull.
 */
int b3w_add_distance_joint( int handle, int bodyA, int bodyB, float ax, float ay, float az, float bx, float by, float bz,
							float minLength, float maxLength, float hertz, float dampingRatio )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return -1;
	}

	b3BodyId idA = BodyId( handle, bodyA );
	b3BodyId idB = BodyId( handle, bodyB );
	if ( B3_IS_NULL( idA ) || B3_IS_NULL( idB ) )
	{
		return -1;
	}

	b3Transform anchorA = { { ax, ay, az }, b3Quat_identity };
	b3Transform anchorB = { { bx, by, bz }, b3Quat_identity };
	b3WorldTransform ta = b3Body_GetTransform( idA );
	b3WorldTransform tb = b3Body_GetTransform( idB );
	b3Transform fa = { { (float)ta.p.x, (float)ta.p.y, (float)ta.p.z }, ta.q };
	b3Transform fb = { { (float)tb.p.x, (float)tb.p.y, (float)tb.p.z }, tb.q };

	b3DistanceJointDef def = b3DefaultDistanceJointDef();
	def.base.bodyIdA = idA;
	def.base.bodyIdB = idB;
	def.base.localFrameA = b3InvMulTransforms( fa, anchorA );
	def.base.localFrameB = b3InvMulTransforms( fb, anchorB );
	def.length = maxLength;
	def.minLength = minLength;
	def.maxLength = maxLength;
	def.enableLimit = true;
	// Required: the limit and the motor are both solved inside the soft branch,
	// so a joint with the spring disabled is rigid and ignores them.
	def.enableSpring = true;
	def.hertz = hertz;
	def.dampingRatio = dampingRatio;
	// A rope only limits how far apart its ends get. The joint default switches
	// off contact between the joined bodies, which let a towed trailer pass
	// straight through the truck; same-machine filtering is the shape group's job.
	def.base.collideConnected = true;

	int slot = ClaimJointSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	return RetainJoint( world, slot, b3CreateDistanceJoint( world->id, &def ) );
}

/**
 * Move a rope's limits. Reeling a winch in is a shrinking maxLength, so the
 * rest length tracks it — otherwise a spring rope would fight its own limit.
 */
void b3w_set_distance_limits( int handle, int joint, float minLength, float maxLength )
{
	JOINT_OR_RETURN( id, handle, joint )

	b3DistanceJoint_SetLengthRange( id, minLength, maxLength );
	b3DistanceJoint_SetLength( id, maxLength );
}

/**
 * Drive the rope's length under power — the winch. Speed is metres per second,
 * positive paying out; force is the budget in newtons, the same (speed, budget)
 * pair every motor in the game takes. A zero force disables the motor so a
 * winch at rest leaves the rope a plain limit again.
 */
void b3w_set_distance_motor( int handle, int joint, float speed, float maxForce )
{
	JOINT_OR_RETURN( id, handle, joint )

	b3DistanceJoint_EnableMotor( id, maxForce > 0.0f );
	b3DistanceJoint_SetMotorSpeed( id, speed );
	b3DistanceJoint_SetMaxMotorForce( id, maxForce );
}

/**
 * The distance between a rope's two anchors right now, in metres.
 *
 * This is what tells the renderer whether a rope is taut or slack — sag is the
 * difference between this and the rope's length — and it is a measurement of
 * the world, not of what we asked for. B3_HUGE for a joint that is gone, so a
 * caller cannot read a missing rope as a rope of length zero.
 */
float b3w_distance_length( int handle, int joint )
{
	JOINT_OR_RETURN( id, handle, joint, B3_HUGE )

	return b3DistanceJoint_GetCurrentLength( id );
}

/**
 * The force a joint carried in the last step, as a magnitude in newtons.
 *
 * Read from the game side rather than through Box3D's threshold events, so the
 * hold limit — a game-design number — stays in the game and the engine keeps
 * reporting a fact. Zero for a joint that is gone.
 */
float b3w_joint_force( int handle, int joint )
{
	JOINT_OR_RETURN( id, handle, joint, 0.0f )

	return b3Length( b3Joint_GetConstraintForce( id ) );
}

/**
 * Attach a static triangle mesh.
 *
 * Internal edges are identified (`identifyEdges`). `mesh_contact.c` rejects a
 * sphere's edge or vertex contact when an adjacent triangle already claimed it,
 * but a hull's or capsule's only on an edge the mesh has marked FLAT, and only
 * `b3IdentifyEdges` marks them: without it every box sliding over two coplanar
 * triangles met the edge between them (a box at 30 m/s on a flat plane of
 * 1 m quads rose 3.5 m). Adjacency is by vertex INDEX, so the caller shares
 * vertices by position; nothing is welded here, so no vertex moves.
 *
 * Box3D keeps a reference to the mesh (see b3CreateMeshShape), so the mesh is
 * the shim's: it outlives the shape and is freed when the shape dies
 * (`OwnedGeometry`).
 *
 * `surfacesPtr`, when non-zero, is one byte per triangle naming the SURFACE it
 * is — mud, rock, road — and 0 means untagged. Each id becomes a material whose
 * `userMaterialId` is that id, which is how a cast reports what it struck: GTA's
 * colpoint carries a surface type per triangle for exactly this. Every material
 * keeps the shape's friction and restitution, so a surface changes what a TYRE
 * makes of the ground (see `b3w_set_surface` in vehicle.c) and nothing about
 * how a crate slides on it.
 *
 * The last `ghostTriangleCount` triangles are GHOSTS (`b3MeshDef::ghostTriangleCount`):
 * a neighbouring mesh's faces along a cut, there only so the classification
 * above sees across it. The rejection works within one mesh, so a surface the
 * caller cut into tiles loses it at every cut, and a shape sliding over one
 * meets the shared edge itself, along a normal no face has. Ghosts never
 * collide and no query returns them. Their surfaces are read like any other
 * triangle's, so `surfacesPtr` covers them too.
 */
int b3w_add_mesh_shape( int handle, int body, int verticesPtr, int vertexCount, int indicesPtr, int triangleCount,
						 float friction, float restitution, int group, int isSensor, int surfacesPtr, int ghostTriangleCount )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return -1;
	}

	// Resolved BEFORE the mesh is built, so a refusal has nothing to free.
	b3BodyId bodyId = BodyOf( world, body );
	if ( B3_IS_NULL( bodyId ) )
	{
		return -1;
	}

	int slot = ClaimShapeSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}

	b3Vec3* vertices = (b3Vec3*)AllocHeld( sizeof( b3Vec3 ) * vertexCount );
	const float* src = (const float*)(intptr_t)verticesPtr;
	for ( int i = 0; i < vertexCount; ++i )
	{
		vertices[i] = ( b3Vec3 ){ src[3 * i], src[3 * i + 1], src[3 * i + 2] };
	}

	int32_t* indices = (int32_t*)AllocHeld( sizeof( int32_t ) * 3 * triangleCount );
	memcpy( indices, (const void*)(intptr_t)indicesPtr, sizeof( int32_t ) * 3 * triangleCount );

	// The highest surface named decides how many materials there are: a mesh
	// whose triangles are all untagged needs none, and keeps the single-material
	// path every mesh took before surfaces existed.
	const uint8_t* surfaces = (const uint8_t*)(intptr_t)surfacesPtr;
	int surfaceCount = 0;
	if ( surfaces != NULL )
	{
		for ( int i = 0; i < triangleCount; ++i )
		{
			surfaceCount = surfaces[i] + 1 > surfaceCount ? surfaces[i] + 1 : surfaceCount;
		}
	}
	bool tagged = surfaceCount > 1;

	b3MeshDef def = { 0 };
	def.vertices = vertices;
	def.indices = indices;
	def.materialIndices = tagged ? (uint8_t*)surfaces : NULL;
	def.weldTolerance = 1e-4f;
	def.vertexCount = vertexCount;
	def.triangleCount = triangleCount;
	def.ghostTriangleCount = ghostTriangleCount;
	def.identifyEdges = true;

	// Not held: nothing allocates between here and the shape create, and that
	// call takes the reference before it allocates — releasing the mesh on a
	// throw from inside it would free geometry a shape already points at.
	b3MeshData* mesh = b3CreateMesh( &def, NULL, 0 );
	FreeHeld( vertices );
	FreeHeld( indices );
	if ( mesh == NULL )
	{
		return -1;
	}

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.enableCustomFiltering = true;
	shapeDef.baseMaterial.friction = friction;
	shapeDef.baseMaterial.restitution = restitution;

	// Copied by the shape on creation, so a stack array is enough.
	b3SurfaceMaterial materials[256];
	if ( tagged )
	{
		for ( int i = 0; i < surfaceCount; ++i )
		{
			materials[i] = shapeDef.baseMaterial;
			materials[i].userMaterialId = (uint64_t)i;
		}
		shapeDef.materials = materials;
		shapeDef.materialCount = surfaceCount;
	}
	shapeDef.isSensor = isSensor != 0;
	shapeDef.enableSensorEvents = isSensor != 0;
	shapeDef.filter.groupIndex = group;

	/*
	 * Speculative contacts against the terrain triangles: ON.
	 *
	 * Box3D's own doc for the flag says to leave it on "unless you care about
	 * reducing ghost collision more than continuous collision under rotation",
	 * and notes it only affects hulls against triangles. That is exactly this
	 * shape: every machine in the game is hulls, and it drives on this.
	 *
	 * It was turned OFF to kill ghost collisions — the "sudden stop", where a
	 * body sliding over the mesh catches on the internal edge between two
	 * nearly coplanar triangles and takes an impulse the surface does not
	 * justify. Measured driving the Car across real terrain, one-tick losses of
	 * forward speed above 1 m/s fell from 19 to 11 over 1440 ticks, worst single
	 * loss 12.63 m/s -> 9.95.
	 *
	 * What that traded away was not "continuous collision under rotation" — the
	 * terrain never rotates, so the comment here read as free. It was the
	 * approach contact: without it a body must OVERLAP the ground before the
	 * solver has anything to solve, so it visibly sinks and is then pushed back
	 * out. Measured on a block dropped onto terrain: 62.5 mm of penetration —
	 * a quarter of a block — at 0.5 m, 2 m and 5 m alike, against 0.0 mm with
	 * the flag on. Speed-independent, because it is the allowed overlap and not
	 * a tunnelling failure.
	 *
	 * Nick's call, knowing the cost: sinking into the ground is worse to look at
	 * than the occasional catch, so ghost collisions come back and are somebody
	 * else's problem to solve properly — with a real fix, not by letting bodies
	 * interpenetrate.
	 */
	shapeDef.enableSpeculativeContact = true;

	b3ShapeId shape = b3CreateMeshShape( bodyId, &shapeDef, mesh, ( b3Vec3 ){ 1.0f, 1.0f, 1.0f } );
	return RetainShape( world, slot, AttachOwned( shape, b3_meshShape, mesh ) );
}

// --- joint motors and limits ---------------------------------------------
//
// The game drives every wheel, pivot and piston through a motor, so this is
// the rest of the joint surface. Revolute and prismatic take the same shape
// deliberately — one host-side call site rather than two — even though Box3D
// names the force cap differently for each (torque vs force).

/// Drive a revolute joint at `speed` rad/s, capped at `maxTorque`. A zero or
/// negative cap disables the motor rather than fighting with zero authority.
void b3w_set_revolute_motor( int handle, int joint, float speed, float maxTorque )
{
	JOINT_OR_RETURN( id, handle, joint )

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
	JOINT_OR_RETURN( id, handle, joint )

	b3Joint_SetConstraintTuning( id, hertz, dampingRatio );
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
	JOINT_OR_RETURN( id, handle, joint )

	b3Joint_WakeBodies( id );
}

void b3w_set_revolute_limits( int handle, int joint, int enable, float lower, float upper )
{
	JOINT_OR_RETURN( id, handle, joint )

	b3RevoluteJoint_EnableLimit( id, enable != 0 );
	if ( enable != 0 )
	{
		b3RevoluteJoint_SetLimits( id, lower, upper );
	}
}

/// As above, for a prismatic joint: `speed` in m/s, capped at `maxForce`.
void b3w_set_prismatic_motor( int handle, int joint, float speed, float maxForce )
{
	JOINT_OR_RETURN( id, handle, joint )

	b3PrismaticJoint_EnableMotor( id, maxForce > 0.0f );
	b3PrismaticJoint_SetMotorSpeed( id, speed );
	b3PrismaticJoint_SetMaxMotorForce( id, maxForce );
}

void b3w_set_prismatic_limits( int handle, int joint, int enable, float lower, float upper )
{
	JOINT_OR_RETURN( id, handle, joint )

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
		// A cast into a world that is not there is not a MISS. `out` is left
		// untouched, so a caller that ignores this reads stale floats rather
		// than a fabricated clean miss.
		return B3W_UNKNOWN;
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
// Hulls built by b3CreateCylinder and b3CreateHull are heap-allocated, and
// freed as soon as the shape exists: `b3CreateHullShape` clones the hull into
// the world's ref-counted hull database, so nothing references ours after it.

int b3w_add_sphere_shape( int handle, int body, float radius, float ox, float oy, float oz, float density,
						  float friction, float restitution, int group, int isSensor )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return -1;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.updateBodyMass = MassFromShapes( world, body );
	def.enableCustomFiltering = true;
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;
	def.filter.groupIndex = group;
	def.isSensor = isSensor != 0;
	def.enableSensorEvents = isSensor != 0;

	b3Sphere sphere = { .center = { ox, oy, oz }, .radius = radius };
	b3BodyId bodyId = BodyOf( world, body );
	if ( B3_IS_NULL( bodyId ) )
	{
		return -1;
	}

	int slot = ClaimShapeSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}
	return RetainShape( world, slot, b3CreateSphereShape( bodyId, &def, &sphere ) );
}

/// A cylinder about local Y, matching how the game orients wheels.
int b3w_add_cylinder_shape( int handle, int body, float halfHeight, float radius, int sides, float density,
							 float friction, float restitution, int group, int isSensor )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return -1;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.updateBodyMass = MassFromShapes( world, body );
	def.enableCustomFiltering = true;
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;
	def.filter.groupIndex = group;
	def.isSensor = isSensor != 0;
	def.enableSensorEvents = isSensor != 0;

	// The body is resolved BEFORE the hull is built: bailing out afterwards
	// used to strand a hull that nothing would ever free.
	b3BodyId bodyId = BodyOf( world, body );
	if ( B3_IS_NULL( bodyId ) )
	{
		return -1;
	}

	int slot = ClaimShapeSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}

	b3HullData* hull = Hold( b3CreateCylinder( 2.0f * halfHeight, radius, -halfHeight, sides ), ReleaseHull );
	b3ShapeId shape = b3CreateHullShape( bodyId, &def, hull );
	Unhold( hull );
	b3DestroyHull( hull );

	return RetainShape( world, slot, shape );
}

/// A convex hull over a point cloud the host uploaded.
int b3w_add_hull_shape( int handle, int body, int pointsPtr, int pointCount, int maxVertices, float density,
						 float friction, float restitution, int group, int isSensor )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return -1;
	}

	int slot = ClaimShapeSlot( world );
	if ( slot < 0 )
	{
		return -1;
	}

	b3Vec3* points = (b3Vec3*)AllocHeld( sizeof( b3Vec3 ) * pointCount );
	const float* src = (const float*)(intptr_t)pointsPtr;
	for ( int i = 0; i < pointCount; ++i )
	{
		points[i] = ( b3Vec3 ){ src[3 * i], src[3 * i + 1], src[3 * i + 2] };
	}

	b3HullData* hull = Hold( b3CreateHull( points, pointCount, maxVertices ), ReleaseHull );
	FreeHeld( points );
	if ( hull == NULL )
	{
		return -1;
	}

	b3ShapeDef def = b3DefaultShapeDef();
	def.updateBodyMass = MassFromShapes( world, body );
	def.enableCustomFiltering = true;
	def.density = density;
	def.baseMaterial.friction = friction;
	def.baseMaterial.restitution = restitution;
	def.filter.groupIndex = group;
	def.isSensor = isSensor != 0;
	def.enableSensorEvents = isSensor != 0;

	b3BodyId bodyId = BodyOf( world, body );
	if ( B3_IS_NULL( bodyId ) )
	{
		Unhold( hull );
		b3DestroyHull( hull );
		return -1;
	}

	b3ShapeId shape = b3CreateHullShape( bodyId, &def, hull );
	Unhold( hull );
	b3DestroyHull( hull );

	return RetainShape( world, slot, shape );
}

// --- lifecycle and world state -------------------------------------------

/// Destroy a body and everything attached to it, geometry included. The handle
/// is retired, never reissued, so a stale reference fails a validity check
/// rather than addressing another body — `b3w_is_valid` is the way to ask. Its
/// slot goes back to the free list for the next body.
void b3w_destroy_body( int handle, int body )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return;
	}

	// The slot is released even if the engine no longer knows the body: the
	// handle is the host's to retire, and it is retired here. After the
	// destroy, not before: gathering a big body's shapes allocates, and a full
	// heap must not leave a live body mapped to a slot the free list hands on.
	b3BodyId id = BodyOf( world, body );
	if ( B3_IS_NON_NULL( id ) )
	{
		DestroyBodyAndGeometry( id );
	}
	ReleaseBodySlot( world, body );
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

/** Defined below with the ray-cast helpers — the tagged-userData scheme. */
static int BodyIndexOfShape( b3ShapeId shapeId );

/**
 * The pair filter behind `excludedFrom`: a body never collides with the one
 * body it is excluded from, and with everything else as its group and mask
 * bits say. Called by Box3D only for pairs where a shape enabled custom
 * filtering, which every shape created here does.
 */
/** Defined in vehicle.c: a tyre's own contact rule. */
int b3w_tyre_presolve( b3ShapeId shapeIdA, b3ShapeId shapeIdB, b3Pos point, b3Vec3 normal );

/** Defined in vehicle.c: is this a vehicle's tyre? Nothing stands on, picks or places onto one. */
bool b3w_is_tyre( b3ShapeId shapeId );
/** Defined in vehicle.c: a tyre never pairs with terrain. */
bool b3w_tyre_should_pair( b3ShapeId shapeIdA, b3ShapeId shapeIdB );

static bool CustomFilterExclusion( b3ShapeId shapeIdA, b3ShapeId shapeIdB, void* context )
{
	if ( !b3w_tyre_should_pair( shapeIdA, shapeIdB ) )
	{
		return false;
	}

	World* world = (World*)context;
	int a = BodyIndexOfShape( shapeIdA );
	int b = BodyIndexOfShape( shapeIdB );
	if ( a < 0 || b < 0 || a >= world->bodySlotCount || b >= world->bodySlotCount )
	{
		return true;
	}
	// Slots in, handles compared: an exclusion names the other body's handle.
	return world->excludedFrom[a] != world->slotHandle[b] && world->excludedFrom[b] != world->slotHandle[a];
}

/**
 * Exclude `body` from colliding with `other` (-1 clears). Existing contacts
 * between them are dropped by re-filtering the body's shapes, the same way a
 * group change does.
 */
void b3w_set_body_exclusion( int handle, int body, int other )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return;
	}
	if ( world->excludedFrom[body] == other )
	{
		return;
	}
	world->excludedFrom[body] = other;
	b3BodyId id = world->bodies[body];
	int count = b3Body_GetShapeCount( id );
	b3ShapeId stack[64];
	b3ShapeId* shapes = count <= 64 ? stack : (b3ShapeId*)AllocHeld( sizeof( b3ShapeId ) * (size_t)count );
	count = b3Body_GetShapes( id, shapes, count );
	for ( int i = 0; i < count; ++i )
	{
		// b3Shape_SetFilter returns early when nothing in the filter changed,
		// so the same filter twice never touches the proxy. A group index the
		// shape does not have, then its own back: two resets, and the pair is
		// re-queried under the new exclusion.
		b3Filter filter = b3Shape_GetFilter( shapes[i] );
		b3Filter nudge = filter;
		nudge.groupIndex = filter.groupIndex == 0 ? 1 : 0;
		b3Shape_SetFilter( shapes[i], nudge, true );
		b3Shape_SetFilter( shapes[i], filter, true );
	}
	if ( shapes != stack )
	{
		FreeHeld( shapes );
	}
}

/**
 * Drop a contact from the solver while still reporting it.
 *
 * Returning false here is Box3D's "detect but do not solve". The game marks
 * vehicle chassis bodies and resolves their contacts positionally after the
 * step, so a machine slides along what it hits rather than being stopped dead
 * by an impulse — which is the jarring stop when a car clips terrain at an
 * angle.
 *
 * Chassis vs chassis is left to the solver deliberately: two machines colliding
 * should push each other, and the positional pass only knows how to move one.
 */
static bool PreSolveDetectOnly( b3ShapeId shapeIdA, b3ShapeId shapeIdB, b3Pos point, b3Vec3 normal, void* context )
{
	// A tyre's contacts under its wheel are the suspension's (see vehicle.c). The
	// rest go on to every other rule here, as any contact does.
	if ( b3w_tyre_presolve( shapeIdA, shapeIdB, point, normal ) == 0 )
	{
		return false;
	}

	World* world = (World*)context;
	int a = BodyIndexOfShape( shapeIdA );
	int b = BodyIndexOfShape( shapeIdB );

	if ( a >= 0 && b >= 0 && a < world->bodySlotCount && b < world->bodySlotCount && world->placed[a] && world->placed[b] )
	{
		return false;
	}

	bool aDetect = a >= 0 && a < world->bodySlotCount && world->detectOnly[a] != 0;
	bool bDetect = b >= 0 && b < world->bodySlotCount && world->detectOnly[b] != 0;

	if ( !aDetect && !bDetect )
	{
		return true;
	}

	/*
	 * Detect-only applies to the WORLD, not to other machines.
	 *
	 * It exists so a chassis clipping a hillside slides along it instead of
	 * stopping dead: the solver leaves that contact alone and the game resolves
	 * it by translating the chassis out. Applied to every pair, it also switched
	 * off collision between a car and every other machine — and returning false
	 * here does not merely skip the impulse, it drops the manifold, so the
	 * positional pass that was supposed to cover it saw nothing at all. Measured:
	 * a car driven at a parked machine reported ZERO manifolds and passed clean
	 * through it, 110 m past where it should have stopped. Rapier keeps the
	 * manifold in the same situation, which is why the engine that shipped
	 * before did not do this.
	 *
	 * Terrain has no owner; every machine's bodies carry their structure's id.
	 */
	uint32_t ownerA = ( a >= 0 && a < world->bodySlotCount ) ? world->bodyOwner[a] : 0;
	uint32_t ownerB = ( b >= 0 && b < world->bodySlotCount ) ? world->bodyOwner[b] : 0;

	if ( ownerA != 0 && ownerB != 0 )
	{
		return true;
	}

	return false;
}

/**
 * Mark the bodies whose contacts the solver must not resolve.
 *
 * Pre-solve events are per SHAPE, so every shape on a marked body has to opt
 * in; the callback itself is registered once, lazily, because a world with no
 * vehicles should not pay for it.
 */
/** Mark a body as placed from the network (or not). Contacts between two placed bodies are detected, never solved. */
void b3w_set_body_placed( int handle, int body, int on )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return;
	}
	if ( world->placed[body] == ( on != 0 ) )
	{
		return;
	}
	world->placed[body] = on != 0;
	b3World_SetPreSolveCallback( world->id, PreSolveDetectOnly, world );
	b3BodyId id = world->bodies[body];
	int count = b3Body_GetShapeCount( id );
	b3ShapeId shapes[64];
	int n = b3Body_GetShapes( id, shapes, count < 64 ? count : 64 );
	for ( int k = 0; k < n; ++k )
	{
		b3Shape_EnablePreSolveEvents( shapes[k], true );
	}
}

void b3w_set_detect_only( int handle, int bodiesPtr, int count )
{
	World* world = GetWorld( handle );
	if ( world == NULL )
	{
		return;
	}

	/*
	 * Per WORLD, every time.
	 *
	 * This was a process-wide `static bool`: the first world to ask registered
	 * the callback and every later world silently got none, while the callback
	 * that was registered kept a pointer to a world that may since have been
	 * destroyed and its arrays freed. A second world is not exotic — the game
	 * makes one per reload and the tests make one per case — and the dangling
	 * context read freed memory, which is how a solver ends up calling through a
	 * pointer that was never a function.
	 *
	 * Registering is idempotent and costs a store, so there is nothing to guard.
	 */
	b3World_SetPreSolveCallback( world->id, PreSolveDetectOnly, world );

	const int32_t* src = (const int32_t*)(intptr_t)bodiesPtr;

	for ( int i = 0; i < world->bodySlotCount; ++i )
	{
		if ( world->detectOnly[i] == 0 )
		{
			continue;
		}
		world->detectOnly[i] = 0;
		b3BodyId clearBody = BodyOf( world, i );
		if ( B3_IS_NON_NULL( clearBody ) )
		{
			int shapeCount = b3Body_GetShapeCount( clearBody );
			b3ShapeId shapes[64];
			int n = b3Body_GetShapes( clearBody, shapes, shapeCount < 64 ? shapeCount : 64 );
			for ( int k = 0; k < n; ++k )
			{
				b3Shape_EnablePreSolveEvents( shapes[k], false );
			}
		}
	}

	for ( int i = 0; i < count; ++i )
	{
		int body = SlotOf( world, src[i] );
		b3BodyId markBody = BodyOf( world, body );
		if ( B3_IS_NULL( markBody ) )
		{
			continue;
		}
		world->detectOnly[body] = 1;

		int shapeCount = b3Body_GetShapeCount( markBody );
		b3ShapeId shapes[64];
		int n = b3Body_GetShapes( markBody, shapes, shapeCount < 64 ? shapeCount : 64 );
		for ( int k = 0; k < n; ++k )
		{
			b3Shape_EnablePreSolveEvents( shapes[k], true );
		}
	}
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
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return B3W_UNKNOWN;
	}

	// A destroyed body keeps its slot — indices are never compacted — so it
	// passes the range check above and only resolves to null HERE. This was the
	// second 0-return in one function, and the one that actually fired.
	b3BodyId id = BodyOf( world, body );
	return B3_IS_NULL( id ) ? B3W_UNKNOWN : b3Body_GetShapeCount( id );
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

	// A tyre is not a surface: nothing stands on it, picks it or places onto it.
	if ( b3w_is_tyre( shapeId ) )
	{
		return -1.0f;
	}

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
	World* world;
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

	if ( b3w_is_tyre( shapeId ) )
	{
		return -1.0f;
	}

	// The exclusions are HANDLES, a run of them for one structure, so the hit is
	// compared as one.
	int slot = BodyIndexOfShape( shapeId );
	int body = slot >= 0 ? HandleOfSlot( ctx->world, slot ) : -1;
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
		// A cast into a world that is not there is not a MISS. `out` is left
		// untouched, so a caller that ignores this reads stale floats rather
		// than a fabricated clean miss.
		return B3W_UNKNOWN;
	}

	b3QueryFilter filter = b3DefaultQueryFilter();
	if ( maskBits != 0 )
	{
		filter.categoryBits = (uint64_t)(uint32_t)categoryBits;
		filter.maskBits = (uint64_t)(uint32_t)maskBits;
	}

	FilteredRayHit ctx = { .closest = { .fraction = 1.0f, .body = -1, .hit = false },
						   .world = world,
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
		// A cast into a world that is not there is not a MISS. `out` is left
		// untouched, so a caller that ignores this reads stale floats rather
		// than a fabricated clean miss.
		return B3W_UNKNOWN;
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
	// The callback recorded the SLOT; the host names bodies by handle.
	out[7] = (float)( closest.body >= 0 ? HandleOfSlot( world, closest.body ) : -1 );
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
	JOINT_OR_RETURN( id, handle, joint )

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
/**
 * Continuous collision on or off for one body, after creation.
 *
 * Continuous collision assumes a body MOVED between two transforms and sweeps
 * the space it crossed. A networked replica does not move: it is re-positioned
 * every tick from an estimate, and a placement half a metre from the last one
 * is not travel, it is a new opinion about where the car has been all along.
 * Swept as though it were travel, it manufactures a time of impact against
 * geometry the car never passed through, and the contact resolves against that
 * phantom instead of against the real one.
 *
 * Measured: a car bulldozing another across open ground pushed it 96 m when
 * both were simulated, and 1.4 m when the pusher was a replica being placed.
 * The push was being spent on impacts that never happened.
 *
 * So the flag has to be per-tick rather than per-body: set while a body is
 * stepped by the solver, cleared while it is being placed.
 */
void b3w_body_set_bullet( int handle, int body, int flag )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return;
	}

	b3Body_SetBullet( BodyOf( world, body ), flag != 0 );
}

// Lets one body turn faster than a quarter turn per step. Box3D clamps every
// body's angular speed to B3_MAX_ROTATION * inv_dt so a fast spinner cannot
// tunnel through what it sweeps; a rotor head has to spin past that to be a
// rotor at all, and is the only body the game asks this for.
void b3w_body_allow_fast_rotation( int handle, int body, int flag )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return;
	}

	b3Body_AllowFastRotation( BodyOf( world, body ), flag != 0 );
}

void b3w_body_set_owner( int handle, int body, unsigned int owner )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return;
	}

	world->bodyOwner[body] = (uint32_t)owner;
}

/** The owner of a body, by SLOT — what a body's userData carries. 0 when
 *  unowned. Read by the wheel cast in vehicle.c, never by the host. */
uint32_t b3w_owner_of_slot( int handle, int slot )
{
	World* world = GetWorld( handle );
	if ( world == NULL || slot < 0 || slot >= world->bodySlotCount )
	{
		return 0;
	}

	return world->bodyOwner[slot];
}

/** The handle of the body in a slot, -1 for none: how vehicle.c names a body
 *  it found by userData when it reports one to the host. */
int b3w_handle_of_slot( int handle, int slot )
{
	World* world = GetWorld( handle );
	return world == NULL ? -1 : HandleOfSlot( world, slot );
}

/// Current joint angle in radians, which the servo and the game's readouts need.
float b3w_revolute_angle( int handle, int joint )
{
	JOINT_OR_RETURN( id, handle, joint, 0.0f )

	return b3RevoluteJoint_GetAngle( id );
}

/**
 * Defer a body's mass update while shapes are attached to it, and apply it
 * once when the deferral ends. Deferring on is free; deferring off recomputes
 * the mass from every shape, once, which is what every add did before.
 */
void b3w_body_defer_mass( int handle, int body, int deferred )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return;
	}

	world->massDeferred[body] = deferred != 0;
	if ( deferred != 0 )
	{
		return;
	}

	ApplyMass( world, body );
}

// --- shape mutation ------------------------------------------------------

/**
 * Destroy one shape, leaving its body alone.
 *
 * The body's mass is updated because the game rebuilds a body's colliders in
 * place — a structure that loses a block must lose its mass with it, or every
 * force applied afterwards is computed against a body heavier than it looks.
 * While the body's mass is deferred (`b3w_body_defer_mass`) the update waits for
 * the deferral to end, as it does for shapes added: tearing down a batch of
 * shapes is then linear, not quadratic, in the body's shape count.
 */
void b3w_destroy_shape( int handle, int shape )
{
	World* world = GetWorld( handle );
	b3ShapeId* id = ResolveShape( world, shape );
	if ( id == NULL )
	{
		return;
	}

	if ( b3Shape_IsValid( *id ) )
	{
		int body = BodyIndexOfShape( *id );
		DestroyShapeAndGeometry( *id, MassFromShapes( world, body ) );
		if ( body >= 0 && body < world->bodySlotCount && world->massStated[body] != 0 && world->massDeferred[body] == 0 )
		{
			ApplyMass( world, body );
		}
	}
}

int b3w_shape_is_valid( int handle, int shape )
{
	b3ShapeId* id = ResolveShape( GetWorld( handle ), shape );
	return id != NULL && b3Shape_IsValid( *id ) ? 1 : 0;
}

/// Destroy a joint. Joints otherwise live until their world does, which is
/// wrong for a game that rebuilds structures on every placement.
void b3w_destroy_joint( int handle, int joint )
{
	b3JointId* slot = ResolveJoint( GetWorld( handle ), joint );
	if ( slot == NULL )
	{
		return;
	}

	if ( b3Joint_IsValid( *slot ) )
	{
		b3DestroyJoint( *slot, true );
	}

	// Nulling is what makes the slot reclaimable on the spot rather than at the
	// next sweep, and it is also the cheap, exact answer for a joint the engine
	// would still call valid at this instant.
	*slot = b3_nullJointId;
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
/** A negative group never collides with itself: a told rider shares its parent's. */
void b3w_set_shape_group( int handle, int shape, int group )
{
	b3ShapeId* id = ResolveShape( GetWorld( handle ), shape );
	if ( id == NULL || b3Shape_IsValid( *id ) == false )
	{
		return;
	}
	b3Filter filter = b3Shape_GetFilter( *id );
	filter.groupIndex = group;
	b3Shape_SetFilter( *id, filter, true );
}

void b3w_set_shape_filter( int handle, int shape, int category, int mask )
{
	b3ShapeId* id = ResolveShape( GetWorld( handle ), shape );
	if ( id == NULL || b3Shape_IsValid( *id ) == false )
	{
		return;
	}

	b3Filter filter = b3Shape_GetFilter( *id );
	filter.categoryBits = (uint64_t)(uint32_t)category;
	filter.maskBits = (uint64_t)(uint32_t)mask;
	b3Shape_SetFilter( *id, filter, true );
}

/// Current extension of a prismatic joint. Box3D tracks this; Rapier's binding
/// exposes no equivalent, which is why its side re-derives it.
float b3w_prismatic_translation( int handle, int joint )
{
	JOINT_OR_RETURN( id, handle, joint, 0.0f )

	return b3PrismaticJoint_GetTranslation( id );
}

// ---------------------------------------------------------------------------
// Vehicle controller
//
// The controller itself lives in vehicle.c. These wrappers exist only to turn
// JavaScript's world handle and body index into the ids it works in.
// ---------------------------------------------------------------------------

int b3w_vehicle_create( b3WorldId world, int worldHandle, b3BodyId chassis, b3Filter filter );
/** Defined in vehicle.c: the host set a body's mass; a chassis re-shares to its wheels. */
void b3w_vehicle_chassis_mass_set( b3BodyId body );
void b3w_vehicle_prepare( void* vehicle, float dt );
void b3w_vehicle_substep( float h, void* context );
void* b3w_vehicle_get( int handle );

/// The machine's collision filter is stated here, in the same terms the host
/// gives its structure shapes (category, mask, group), and its tyres wear it.
int b3w_create_vehicle( int handle, int chassis, int category, int mask, int group )
{
	World* world = GetWorld( handle );
	b3BodyId chassisBody = BodyOf( world, SlotOf( world, chassis ) );
	if ( B3_IS_NULL( chassisBody ) )
	{
		return -1;
	}
	b3Filter filter = b3DefaultFilter();
	filter.categoryBits = (uint64_t)(uint32_t)category;
	filter.maskBits = (uint64_t)(uint32_t)mask;
	filter.groupIndex = group;
	return b3w_vehicle_create( world->id, handle, chassisBody, filter );
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
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 || threshold < 0.0f )
	{
		return;
	}

	b3BodyId id = BodyOf( world, body );
	if ( B3_IS_NULL( id ) )
	{
		return;
	}

	b3Body_SetSleepThreshold( id, threshold );
}

void b3w_set_body_mass( int handle, int body, float mass )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 || mass <= 0.0f )
	{
		return;
	}

	// A stated mass is scaled as stated: its shapes carry none to derive from.
	if ( world->massStated[body] != 0 )
	{
		b3MassData* md = &world->statedMass[body];
		float scale = mass / md->mass;
		md->mass = mass;
		md->inertia.cx = b3MulSV( scale, md->inertia.cx );
		md->inertia.cy = b3MulSV( scale, md->inertia.cy );
		md->inertia.cz = b3MulSV( scale, md->inertia.cz );
		ApplyMass( world, body );
		return;
	}

	// From the shapes, every time, never from the mass the body has now. That
	// is only the shapes' when they were just re-derived; a vehicle's chassis
	// sits at the whole less its wheels' share, so scaling from it inflated the
	// inertia by whole/share on every write that re-derived nothing first — a
	// car released from the physgun turned twice as heavily each time.
	b3BodyId id = BodyOf( world, body );
	b3Body_ApplyMassFromShapes( id );
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

	// A chassis's mass is the whole machine's: its wheels take their share now.
	b3w_vehicle_chassis_mass_set( id );
}

/**
 * State a body's mass outright: total mass, centre of mass in the body frame,
 * and the symmetric inertia tensor about that centre (the tensor's entries, so
 * the off-diagonals are minus the products of inertia).
 *
 * The body keeps it whatever shapes are added or destroyed afterwards, deferred
 * or not: its shapes carry collision only. A mass of zero or less hands the
 * mass back to the shapes.
 */
void b3w_set_body_mass_data( int handle, int body, float mass, float cx, float cy, float cz, float ixx, float iyy,
							 float izz, float ixy, float ixz, float iyz )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return;
	}

	if ( mass <= 0.0f )
	{
		world->massStated[body] = 0;
		ApplyMass( world, body );
		return;
	}

	b3MassData md;
	md.mass = mass;
	md.center = ( b3Vec3 ){ cx, cy, cz };
	md.inertia.cx = ( b3Vec3 ){ ixx, ixy, ixz };
	md.inertia.cy = ( b3Vec3 ){ ixy, iyy, iyz };
	md.inertia.cz = ( b3Vec3 ){ ixz, iyz, izz };
	world->statedMass[body] = md;
	world->massStated[body] = 1;
	ApplyMass( world, body );
}

/**
 * Flatten a body's contact manifolds into `out`, returning how many were
 * written (capped at `capacity`).
 *
 * 8 floats per manifold: normal xyz (pointing AWAY from the queried body),
 * deepest penetration depth, normal impulse, tangent impulse, other body index,
 * approach speed.
 *
 * Approach speed is the pre-solve closing speed along the normal, the same
 * quantity Box3D's own hit events report (solver.c) — how hard the two shapes
 * were coming together before the solver stopped them. It is what a "hit" is:
 * mass-independent, and untouched by whatever forces the caller applied that
 * step, which is what makes it the right trigger for anything shock-sensitive.
 * Only points the solver actually pushed on count, as in the hit events.
 *
 * The normal is flipped here when the queried body is shape B, because Box3D
 * reports A->B and the caller must not have to know which end it is on: a
 * chassis pushed along an inverted normal moves INTO what it hit.
 */
int b3w_body_contacts( int handle, int body, float* out, int capacity )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 || capacity <= 0 )
	{
		return 0;
	}

	b3BodyId id = BodyOf( world, body );
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
		int otherIndex = tagged > 0 ? HandleOfSlot( world, (int)( tagged - 1 ) ) : -1;

		for ( int m = 0; m < data[i].manifoldCount && written < capacity; m++ )
		{
			const b3Manifold* manifold = data[i].manifolds + m;
			if ( manifold->pointCount == 0 )
			{
				continue;
			}

			float depth = 0.0f;
			float normalImpulse = 0.0f;
			float approachSpeed = 0.0f;
			for ( int p = 0; p < manifold->pointCount; p++ )
			{
				const b3ManifoldPoint* mp = manifold->points + p;
				float separation = mp->separation;
				if ( separation < 0.0f && -separation > depth )
				{
					depth = -separation;
				}
				normalImpulse += mp->normalImpulse;
				if ( mp->totalNormalImpulse > 0.0f && -mp->normalVelocity > approachSpeed )
				{
					approachSpeed = -mp->normalVelocity;
				}
			}

			float* o = out + written * 8;
			o[0] = manifold->normal.x * sign;
			o[1] = manifold->normal.y * sign;
			o[2] = manifold->normal.z * sign;
			o[3] = depth;
			o[4] = normalImpulse;
			o[5] = b3Length( manifold->frictionImpulse );
			o[6] = (float)otherIndex;
			o[7] = approachSpeed;
			written++;
		}
	}

	return written;
}

/**
 * The linear impulse `body` received from its contacts with `other` over the
 * last step, into `out[0..2]`, and how many touching contacts the two share.
 *
 * Every contact between the two, not the first 64 manifolds: a block-built
 * machine touching another carries hundreds. Read from each manifold's
 * `stepImpulse`, which is what the contact did and nothing else — a body's
 * change of velocity across a step also holds its brakes, its tyres and
 * gravity, and a collision measured that way is a collision plus all of them.
 */
int b3w_body_pair_impulse( int handle, int body, int other, float* out )
{
	out[0] = out[1] = out[2] = 0.0f;
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 || ( other = SlotOf( world, other ) ) < 0 )
	{
		return 0;
	}

	b3BodyId id = BodyOf( world, body );
	b3BodyId otherId = BodyOf( world, other );
	int capacity = b3Body_GetContactCapacity( id );
	if ( capacity <= 0 )
	{
		return 0;
	}

	b3ContactData stack[64];
	b3ContactData* data = capacity <= 64 ? stack : (b3ContactData*)CheckedAlloc( sizeof( b3ContactData ) * (size_t)capacity );
	int count = b3Body_GetContactData( id, data, capacity );

	int touching = 0;
	for ( int i = 0; i < count; i++ )
	{
		b3BodyId a = b3Shape_GetBody( data[i].shapeIdA );
		b3BodyId b = b3Shape_GetBody( data[i].shapeIdB );
		bool queriedIsA = B3_ID_EQUALS( a, id );
		if ( !B3_ID_EQUALS( queriedIsA ? b : a, otherId ) )
		{
			continue;
		}
		touching++;
		// A manifold's step impulse is shape B's; shape A received its negative.
		float sign = queriedIsA ? -1.0f : 1.0f;
		for ( int m = 0; m < data[i].manifoldCount; m++ )
		{
			b3Vec3 j = data[i].manifolds[m].stepImpulse;
			out[0] += sign * j.x;
			out[1] += sign * j.y;
			out[2] += sign * j.z;
		}
	}

	if ( data != stack )
	{
		free( data );
	}
	return touching;
}

/**
 * The hardest this body was struck this step: the largest approach speed over
 * EVERY contact point it has, in m/s. Zero for a body resting or separating.
 *
 * Its own entry point rather than a read of `b3w_body_contacts`, because that
 * one caps at 64 manifolds and a large chassis on several terrain chunks can
 * carry more — a strike hidden past the cap would simply not be reported. This
 * walks the whole set, taking a heap buffer only when the stack one is short.
 */
float b3w_body_max_approach_speed( int handle, int body )
{
	World* world = GetWorld( handle );
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return 0.0f;
	}

	b3BodyId id = BodyOf( world, body );
	int count = b3Body_GetContactCapacity( id );
	if ( count <= 0 )
	{
		return 0.0f;
	}

	b3ContactData local[64];
	b3ContactData* data = local;
	if ( count > 64 )
	{
		data = (b3ContactData*)CheckedAlloc( (size_t)count * sizeof( b3ContactData ) );
	}
	count = b3Body_GetContactData( id, data, count );

	float best = 0.0f;
	for ( int i = 0; i < count; i++ )
	{
		for ( int m = 0; m < data[i].manifoldCount; m++ )
		{
			const b3Manifold* manifold = data[i].manifolds + m;
			for ( int p = 0; p < manifold->pointCount; p++ )
			{
				const b3ManifoldPoint* mp = manifold->points + p;
				if ( mp->totalNormalImpulse > 0.0f && -mp->normalVelocity > best )
				{
					best = -mp->normalVelocity;
				}
			}
		}
	}

	if ( data != local )
	{
		free( data );
	}
	return best;
}

/** A body's world AABB, as min xyz then max xyz. */
void b3w_body_aabb( int handle, int body, float* out )
{
	b3BodyId aabbBody = BodyId( handle, body );
	if ( B3_IS_NULL( aabbBody ) )
	{
		return;
	}

	b3AABB aabb = b3Body_ComputeAABB( aabbBody );
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
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return;
	}

	b3BodyId id = BodyOf( world, body );
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
	if ( world == NULL || ( body = SlotOf( world, body ) ) < 0 )
	{
		return 0;
	}

	b3BodyId id = BodyOf( world, body );
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
		// A cast into a world that is not there is not a MISS. `out` is left
		// untouched, so a caller that ignores this reads stale floats rather
		// than a fabricated clean miss.
		return B3W_UNKNOWN;
	}

	b3Vec3 point = { 0.0f, 0.0f, 0.0f };
	b3ShapeProxy proxy = { .points = &point, .count = 1, .radius = radius };

	FilteredCastHit ctx = { .closest = { .fraction = 1.0f, .normal = { 0.0f, 1.0f, 0.0f }, .hit = false },
							.exclude = BodyOf( world, SlotOf( world, exclude ) ) };

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

	for ( int i = 0; i < world->bodySlotCount; ++i )
	{
		if ( world->slotHandle[i] == excludeBody )
		{
			continue;
		}

		b3BodyId bodyId = BodyOf( world, i );
		if ( B3_IS_NULL( bodyId ) )
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
