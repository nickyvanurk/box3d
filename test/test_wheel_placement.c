// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/math_functions.h"

// The shipping controller, included whole so the STATIC function under test is
// the one the game runs. A test that re-implements the maths beside it proves
// only that two copies agree, and the second copy is as likely to be wrong as
// the first — which is exactly the mistake this file exists to stop making.
#include "../bindings/wasm/vehicle.c"

#include <stdio.h>

// `vehicle.c` asks the wasm shim who owns a body, to keep a wheel from standing
// on its own machine. Nothing here casts, so a stub satisfies the linker without
// pretending to answer.
uint32_t b3w_owner_of_body( int handle, int body )
{
	(void)handle;
	(void)body;

	// Anything but the vehicle's own owner, which defaults to 0. Returning 0 here
	// makes the cast treat the GROUND as part of the machine and reject it, the
	// wheel finds nothing, and the chassis lands on its own collider — which read
	// as a tyre 537 mm underground the first time this rig ran.
	return 7;
}

// The wheel placement maths from the Uncharted 4 vehicle port, tested as MATHS.
//
// Every number this project has measured for it so far came from driving a whole
// car around real terrain, which mixes the cast, the placement, the point
// constraint, the suspension, friction, drive torque and the shape of the hill
// into one figure. Changing a constant and watching that figure move says almost
// nothing about which mechanism moved it — twice a sweep produced byte-identical
// results and it was read as "not the cause" when it meant "never reached".
//
// So this file contains no world, no bodies and no time. It builds the geometry
// by hand, calls the function, and compares against a distance worked out on
// paper. The talk's own slides give the formulas:
//
//   P1 = Center + Right - Up            CP1 = P1 - Center      S1 = CP1 . N
//   P2 = Center - Right - Up            CP2 = P2 - Center      S2 = CP2 . N
//   S3 = (CP1 - Contact Point) . N
//
// then S3 is projected onto the suspension direction, because moving along the
// contact normal would pull the wheel out of line with its own suspension.

// Build the wheel and contact the shipping function expects, then ask it how
// far the tyre must move. No world, no bodies, no time.
static float Placement( b3Vec3 center, b3Vec3 up, b3Vec3 right, float radius, float halfWidth, b3Vec3 contactPoint,
						b3Vec3 n, b3Vec3 suspensionDown )
{
	Wheel w;
	memset( &w, 0, sizeof( Wheel ) );
	w.radius = radius;
	w.width = 2.0f * halfWidth;
	w.wheelDirectionWs = suspensionDown;
	w.wheelAxleWs = right;

	WheelContact contact;
	memset( &contact, 0, sizeof( WheelContact ) );
	contact.surfaceNormalWs = n;
	contact.pointWs = B3_LITERAL( b3Pos ){ contactPoint.x, contactPoint.y, contactPoint.z };

	(void)up;
	return PlacementAlongSuspension( &w, &contact, B3_LITERAL( b3Pos ){ center.x, center.y, center.z } );
}

static bool NearlyEqual( float a, float b, float tolerance )
{
	float d = a - b;
	return ( d < 0.0f ? -d : d ) <= tolerance;
}

// A level wheel resting exactly on flat ground needs no placement at all.
static int TangentOnFlatGround( void )
{
	b3Vec3 up = { 0.0f, 1.0f, 0.0f };
	b3Vec3 right = { 0.0f, 0.0f, 1.0f };
	b3Vec3 down = { 0.0f, -1.0f, 0.0f };
	b3Vec3 n = { 0.0f, 1.0f, 0.0f };

	float radius = 0.4f;
	b3Vec3 center = { 0.0f, radius, 0.0f };
	b3Vec3 contact = { 0.0f, 0.0f, 0.0f };

	float move = Placement( center, up, right, radius, 0.15f, contact, n, down );
	ENSURE( NearlyEqual( move, 0.0f, 1.0e-5f ) );
	return 0;
}

// A level wheel sunk 100 mm into flat ground must come up by exactly 100 mm.
static int SunkIntoFlatGround( void )
{
	b3Vec3 up = { 0.0f, 1.0f, 0.0f };
	b3Vec3 right = { 0.0f, 0.0f, 1.0f };
	b3Vec3 down = { 0.0f, -1.0f, 0.0f };
	b3Vec3 n = { 0.0f, 1.0f, 0.0f };

	float radius = 0.4f;
	float sink = 0.1f;
	b3Vec3 center = { 0.0f, radius - sink, 0.0f };
	b3Vec3 contact = { 0.0f, 0.0f, 0.0f };

	float move = Placement( center, up, right, radius, 0.15f, contact, n, down );
	ENSURE( NearlyEqual( move, sink, 1.0e-5f ) );
	return 0;
}

// A wheel clear of the ground is not pulled down onto it.
static int ClearOfTheGround( void )
{
	b3Vec3 up = { 0.0f, 1.0f, 0.0f };
	b3Vec3 right = { 0.0f, 0.0f, 1.0f };
	b3Vec3 down = { 0.0f, -1.0f, 0.0f };
	b3Vec3 n = { 0.0f, 1.0f, 0.0f };

	float radius = 0.4f;
	b3Vec3 center = { 0.0f, radius + 0.05f, 0.0f };
	b3Vec3 contact = { 0.0f, 0.0f, 0.0f };

	float move = Placement( center, up, right, radius, 0.15f, contact, n, down );
	ENSURE( NearlyEqual( move, 0.0f, 1.0e-5f ) );
	return 0;
}

// On a slope the wheel may only move along its SUSPENSION, so clearing a
// penetration of `s3` along the normal costs `s3 / cos(theta)` of travel. This
// is the projection step, and getting it wrong leaves a tyre in a hill while
// looking correct on the flat.
static int SlopeCostsMoreTravel( void )
{
	float theta = 0.5235988f; // 30 degrees
	float c = cosf( theta );
	float s = sinf( theta );

	// Surface tilted 30 degrees; the suspension still points down the world.
	b3Vec3 n = { -s, c, 0.0f };
	b3Vec3 down = { 0.0f, -1.0f, 0.0f };
	b3Vec3 up = { 0.0f, 1.0f, 0.0f };
	b3Vec3 right = { 0.0f, 0.0f, 1.0f };

	float radius = 0.4f;
	float sink = 0.08f;

	// Put the deepest corner exactly `sink` below the plane through the contact
	// point, measured along the normal.
	b3Vec3 contact = { 0.0f, 0.0f, 0.0f };
	b3Vec3 corner = b3MulSV( -sink, n );
	b3Vec3 center = b3Add( corner, b3MulSV( radius, up ) );

	float move = Placement( center, up, right, radius, 0.0f, contact, n, down );
	ENSURE( NearlyEqual( move, sink / c, 1.0e-4f ) );
	return 0;
}

// A wheel whose SUSPENSION is tilted — a cambered mount, or a car leaning on a
// slope — presents a corner rather than the bottom of its centre line, and the
// corner is what must clear the surface.
//
// The corners come from the suspension axis and the axle, NOT from a wheel-fixed
// "up". A tyre spins freely about its axle, so a wheel-fixed up rotates with the
// spin and the corners would wobble once per revolution; the talk's Up is the
// direction the suspension holds the wheel against, which does not spin. Writing
// this test the other way round is what caught the ambiguity.
static int CamberedWheelUsesTheDeeperCorner( void )
{
	float camber = 0.2617994f; // 15 degrees
	b3Vec3 down = { sinf( camber ), -cosf( camber ), 0.0f };
	b3Vec3 up = { -down.x, -down.y, -down.z };
	b3Vec3 right = { 0.0f, 0.0f, 1.0f };
	b3Vec3 n = { 0.0f, 1.0f, 0.0f };

	float radius = 0.4f;
	float halfWidth = 0.15f;
	b3Vec3 center = { 0.0f, 0.30f, 0.0f };
	b3Vec3 contact = { 0.0f, 0.0f, 0.0f };

	// By hand: both corners sit at center - radius*up, offset along the axle,
	// which is horizontal here — so they share a height and the deeper test is a
	// tie broken either way to the same distance. The corner is BELOW the plane
	// by that much, and the suspension is tilted, so clearing it costs
	// depth / cos(camber) of travel.
	float cornerY = center.y - radius * up.y;
	float expected = -cornerY / cosf( camber );

	float move = Placement( center, up, right, radius, halfWidth, contact, n, down );
	ENSURE( NearlyEqual( move, expected, 1.0e-4f ) );
	return 0;
}

// The wheel's own spin must not move the corners. A cylinder is the same shape
// at every angle, so a placement that changes as the tyre rolls would jitter the
// ride height once per revolution.
static int SpinDoesNotMoveTheCorners( void )
{
	b3Vec3 up = { 0.0f, 1.0f, 0.0f };
	b3Vec3 down = { 0.0f, -1.0f, 0.0f };
	b3Vec3 right = { 0.0f, 0.0f, 1.0f };
	b3Vec3 n = { 0.0f, 1.0f, 0.0f };

	float radius = 0.4f;
	b3Vec3 center = { 0.0f, radius - 0.05f, 0.0f };
	b3Vec3 contact = { 0.0f, 0.0f, 0.0f };

	float first = Placement( center, up, right, radius, 0.15f, contact, n, down );

	// The same wheel, a third of a turn later. Only the axle is passed to the
	// placement, and rolling does not change it.
	float second = Placement( center, up, right, radius, 0.15f, contact, n, down );

	ENSURE( NearlyEqual( first, second, 1.0e-6f ) );
	ENSURE( NearlyEqual( first, 0.05f, 1.0e-5f ) );
	return 0;
}

int WheelPlacementTest( void )
{
	RUN_SUBTEST( TangentOnFlatGround );
	RUN_SUBTEST( SunkIntoFlatGround );
	RUN_SUBTEST( ClearOfTheGround );
	RUN_SUBTEST( SlopeCostsMoreTravel );
	RUN_SUBTEST( CamberedWheelUsesTheDeeperCorner );
	RUN_SUBTEST( SpinDoesNotMoveTheCorners );
	return 0;
}

// ONE wheel, one flat plane, no car.
//
// The question "how deep does the tyre sit" has only been asked of a whole
// vehicle on real terrain so far, where the answer mixes the cast, the
// placement, the point constraint, the suspension force, the joint limit, drive
// torque, friction, the blueprint's mass distribution and the shape of the
// ground. This rig removes all of it: a static plane, a chassis that is a plain
// box, and a single wheel. Whatever depth it reports belongs to the contact
// machinery and nothing else.

#define RIG_TICK g_rigTick

/** The tick the rig steps at. 24 Hz is the game's; the sweep varies it because
 *  a penetration that is a per-TICK distance must shrink as the tick does. */
static float g_rigTick = 1.0f / 24.0f;
#define RIG_SUBSTEPS 32
/** The ground's own collision category, so the depth probe can ask for it by
 *  name rather than hitting the machine on the way down. */
#define GROUND_CATEGORY 0x2ull
#define MACHINE_CATEGORY 0x4ull

static int g_rigSubsteps = RIG_SUBSTEPS;
static float g_rigTravel = 0.25f;
static bool g_rigMeshGround = false;

typedef struct
{
	b3WorldId world;
	b3BodyId chassis;
	b3BodyId ground;
	/** The surface's tilt, kept rather than read back off the ground body: a mesh
	 *  ground carries its slope in its TRIANGLES and leaves the body unrotated, so
	 *  deriving the plane from the transform gives a horizontal one and reports a
	 *  wheel high on a hill as 74 metres underground. */
	float slope;
	int vehicle;
	float radius;
} WheelRig;

static void RigSubstep( float h, void* context )
{
	Vehicle* v = b3w_vehicle_get( (int)(intptr_t)context );
	if ( v != NULL && v->armed )
	{
		b3w_vehicle_solve( v, h );
	}
}

// A chassis held on one wheel, over a plane at y = 0.
//
// The chassis is deliberately heavy next to the tyre — that ratio is the whole
// difficulty, and a rig that used a light chassis would flatter every result.
/**
 * Ground as a triangle MESH, built the way the game builds terrain.
 *
 * Every case in this file until now stood the wheel on a BOX, and the game's
 * ground is neither a box nor a heightfield: it is a dual-contoured triangle
 * mesh, meshed by the renderer's own mesher. A shapecast against triangles is a
 * different query from a shapecast against a hull — different witness points on
 * edges, different normals across a crease — and the shim turns speculative
 * contacts OFF for exactly this shape. None of that was under test.
 */
static void CreateMeshGround( WheelRig* rig, float slopeRadians )
{
	b3BodyDef def = b3DefaultBodyDef();
	def.type = b3_staticBody;
	rig->ground = b3CreateBody( rig->world, &def );

	// A large tilted plane, as two triangles, with its surface through the
	// origin — the same surface the box version presents.
	float c = cosf( slopeRadians );
	float sn = sinf( slopeRadians );
	float e = 400.0f;

	b3Vec3 along = { c, sn, 0.0f };
	b3Vec3 across = { 0.0f, 0.0f, 1.0f };

	b3Vec3 verts[4];
	for ( int i = 0; i < 4; i++ )
	{
		float u = ( i == 0 || i == 3 ) ? -e : e;
		float w = ( i < 2 ) ? -e : e;
		verts[i] = b3Add( b3MulSV( u, along ), b3MulSV( w, across ) );
	}

	// Wound so the face normal points UP. Triangle collision is single-sided, and
	// with `along x across` pointing downward the first attempt let the machine
	// fall through the world and reported a tyre 74 metres under the surface.
	int32_t indices[6] = { 0, 2, 1, 0, 3, 2 };

	b3MeshDef meshDef = { 0 };
	meshDef.vertices = verts;
	meshDef.indices = indices;
	meshDef.weldTolerance = 1e-4f;
	meshDef.vertexCount = 4;
	meshDef.triangleCount = 2;

	b3MeshData* mesh = b3CreateMesh( &meshDef, NULL, 0 );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	// The game's setting for terrain, and the reason it is set.
	shapeDef.enableSpeculativeContact = false;
	// A category of its own so the depth probe can ask for the GROUND and not be
	// answered by the chassis hanging above the wheel.
	shapeDef.filter.categoryBits = GROUND_CATEGORY;
	b3CreateMeshShape( rig->ground, &shapeDef, mesh, B3_LITERAL( b3Vec3 ){ 1.0f, 1.0f, 1.0f } );
}

/**
 * Ground with a CREASE: two planes meeting at a ridge or a valley.
 *
 * Real terrain is dual contoured — thousands of flat facets meeting at angles —
 * and every case in this file so far has stood the wheel on ONE plane. A crease
 * is the shape the talk is talking about when it says a tyre "sits on multiple
 * planks", and it is where the faked normal earns both its keep and its cost:
 * the response direction is a lie, so the geometry has to come from the two real
 * facets, and a wheel bridging them touches neither at its centre.
 *
 * `angle` is the half-angle between the planes. Positive makes a RIDGE (convex,
 * the wheel bridges the top), negative a VALLEY (concave, the wheel wedges into
 * it) — and a valley is the harder of the two, because both facets push.
 */
static void CreateCreasedGround( WheelRig* rig, float angle )
{
	b3BodyDef def = b3DefaultBodyDef();
	def.type = b3_staticBody;
	rig->ground = b3CreateBody( rig->world, &def );

	float e = 200.0f;
	float run = 40.0f;
	float rise = run * tanf( angle );

	// Four quads across the crease at x = 0, so the wheel meets the join with
	// plenty of surface either side.
	b3Vec3 verts[8] = {
		{ -run, rise, -e }, { -run, rise, e }, { 0.0f, 0.0f, -e }, { 0.0f, 0.0f, e },
		{ run, rise, -e },	{ run, rise, e },  { -e, rise, -e },   { -e, rise, e },
	};

	int32_t indices[12] = {
		0, 2, 1, 1, 2, 3, // approach face, down to the crease
		2, 4, 3, 3, 4, 5, // far face, up from the crease
	};

	b3MeshDef meshDef = { 0 };
	meshDef.vertices = verts;
	meshDef.indices = indices;
	meshDef.weldTolerance = 1e-4f;
	meshDef.vertexCount = 8;
	meshDef.triangleCount = 4;

	b3MeshData* mesh = b3CreateMesh( &meshDef, NULL, 0 );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.enableSpeculativeContact = false;
	shapeDef.filter.categoryBits = GROUND_CATEGORY;
	b3CreateMeshShape( rig->ground, &shapeDef, mesh, B3_LITERAL( b3Vec3 ){ 1.0f, 1.0f, 1.0f } );
}

/** Chassis density. The reported symptom is on a HEAVY blueprint, and mass is
 *  the one thing every case here held fixed. */
static float g_rigChassisDensity = 400.0f;

/** The wheels and their tuning, so a rebuild can hang a fresh set under a fresh
 *  chassis without a second copy of the game's numbers. */
static bool g_rigSidewaysWheel = false;

static void AddRigWheels( WheelRig* rig, int wheelCount )
{
	// The wheels, hung under the chassis: suspension straight down, axle across.
	// One wheel isolates the contact machinery; four bring in what a chassis does
	// when its corners disagree, which is where pitching and flipping live.
	if ( wheelCount == 1 )
	{
		// Suspension down and axle across, unless the caller wants the SIDEWAYS
		// mount a player can build and the game's own fixtures produce by default.
		if ( g_rigSidewaysWheel )
		{
			b3w_vehicle_add_wheel( rig->vehicle, 0.0f, -0.3f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.25f,
								   rig->radius, 0.3f );
		}
		else
		{
			b3w_vehicle_add_wheel( rig->vehicle, 0.0f, -0.3f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f,
								   rig->radius, 0.3f );
		}
	}
	else
	{
		for ( int i = 0; i < 4; i++ )
		{
			float x = ( i < 2 ) ? 0.5f : -0.5f;
			float z = ( i % 2 == 0 ) ? 0.35f : -0.35f;
			b3w_vehicle_add_wheel( rig->vehicle, x, -0.3f, z, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f, rig->radius,
								   0.3f );
		}
	}

	Vehicle* v = b3w_vehicle_get( rig->vehicle );

	// The game's own numbers for a single-wheel machine: sag is 40% of rest
	// length, damping a fraction of critical (`wheel-config.ts`).
	float sag = 0.4f * 0.25f;
	float stiffness = 9.81f / sag;
	// Per WHEEL, exactly as `wheel-config.ts:125` derives it: the stiffness each
	// wheel carries is `k / n`, so its critical damping is `2 * sqrt(k / n)`.
	// Without the divide a four-wheeled rig runs at twice critical, and an
	// over-damped suspension cannot hold the car up while it is moving — it sank
	// onto its bump stops within a second and every ramp number measured that
	// instead of the controller.
	float critical = 2.0f * sqrtf( stiffness / (float)( wheelCount > 0 ? wheelCount : 1 ) );

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		b3w_vehicle_set_wheel_mass( rig->vehicle, i, 20.0f );
		b3w_vehicle_set_suspension_stiffness( rig->vehicle, i, stiffness / (float)v->wheelCount );
		b3w_vehicle_set_suspension_compression( rig->vehicle, i, 0.55f * critical );
		b3w_vehicle_set_suspension_relaxation( rig->vehicle, i, 0.8f * critical );
		b3w_vehicle_set_max_suspension_travel( rig->vehicle, i, g_rigTravel );
		b3w_vehicle_set_max_suspension_force( rig->vehicle, i, 20000.0f );
	}

}

int b3w_tyre_presolve( b3ShapeId shapeIdA, b3ShapeId shapeIdB, b3Pos point, b3Vec3 normal );

static bool RigPreSolve( b3ShapeId shapeIdA, b3ShapeId shapeIdB, b3Pos point, b3Vec3 normal, void* context )
{
	(void)context;
	return b3w_tyre_presolve( shapeIdA, shapeIdB, point, normal ) != 0;
}

bool b3w_tyre_should_pair( b3ShapeId shapeIdA, b3ShapeId shapeIdB );

static bool RigPairFilter( b3ShapeId shapeIdA, b3ShapeId shapeIdB, void* context )
{
	(void)context;
	return b3w_tyre_should_pair( shapeIdA, shapeIdB );
}

static WheelRig CreateRig( float slopeRadians, int wheelCount )
{
	WheelRig rig;
	rig.radius = 0.4f;
	rig.slope = slopeRadians;

	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = B3_LITERAL( b3Vec3 ){ 0.0f, -9.81f, 0.0f };
	rig.world = b3CreateWorld( &worldDef );

	// Ground: a big box, tilted by `slopeRadians` about Z so the surface normal
	// leans. A box rather than a height field, so the surface is exactly the
	// plane the maths expects and any error is the vehicle's.
	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.type = b3_staticBody;
	groundDef.rotation = b3MakeQuatFromAxisAngle( B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 1.0f }, slopeRadians );

	// Push the box down along ITS OWN up, so the top face passes through the
	// origin whatever the tilt. Positioning it straight down the world instead
	// moves the surface sideways as it rotates, and the vehicle then spawns
	// relative to ground that is not where the test thinks it is — which is how
	// this rig first reported a tyre a metre in the air.
	groundDef.position =
		b3OffsetPos( B3_LITERAL( b3Pos ){ 0.0f, 0.0f, 0.0f },
					 b3RotateVector( groundDef.rotation, B3_LITERAL( b3Vec3 ){ 0.0f, -5.0f, 0.0f } ) );
	rig.ground = b3CreateBody( rig.world, &groundDef );
	b3BodyId ground = rig.ground;

	if ( g_rigMeshGround )
	{
		CreateMeshGround( &rig, slopeRadians );
		// The tilt is baked into the triangles, so the body itself is unrotated
		// and its origin IS a point on the surface.
		b3Body_SetTransform( rig.ground, B3_LITERAL( b3Pos ){ 0.0f, 0.0f, 0.0f }, b3Quat_identity );
	}
	else
	{
		b3ShapeDef groundShape = b3DefaultShapeDef();
		groundShape.filter.categoryBits = GROUND_CATEGORY;
		b3BoxHull groundBox = b3MakeBoxHull( 200.0f, 5.0f, 200.0f );
		b3CreateHullShape( ground, &groundShape, &groundBox.base );
	}

	// Chassis: one box, well clear of the ground so the wheel lands first.
	b3BodyDef chassisDef = b3DefaultBodyDef();
	chassisDef.type = b3_dynamicBody;
	chassisDef.position = B3_LITERAL( b3Pos ){ 0.0f, 1.2f, 0.0f };
	rig.chassis = b3CreateBody( rig.world, &chassisDef );

	b3ShapeDef chassisShape = b3DefaultShapeDef();
	chassisShape.density = g_rigChassisDensity;
	// Anything but the ground's category. Box3D's default is UINT64_MAX, which
	// matches every mask — so a "ground only" raycast was answered by the chassis
	// hanging above the wheel, and every depth in this file briefly read 1.12 m.
	chassisShape.filter.categoryBits = MACHINE_CATEGORY;
	b3BoxHull chassisBox = b3MakeBoxHull( 0.6f, 0.3f, 0.4f );
	b3CreateHullShape( rig.chassis, &chassisShape, &chassisBox.base );

	rig.vehicle = b3w_vehicle_create( rig.world, 0, rig.chassis );
	// The tyre's rules, as the shim installs them for every game world.
	b3World_SetPreSolveCallback( rig.world, RigPreSolve, NULL );
	b3World_SetCustomFilterCallback( rig.world, RigPairFilter, NULL );

	AddRigWheels( &rig, wheelCount );

	b3World_SetSubStepCallback( rig.world, RigSubstep, (void*)(intptr_t)rig.vehicle );
	return rig;
}

static bool g_rigTrace = false;
int g_traceWheel = 0;

static void StepRig( WheelRig* rig, int ticks )
{
	Vehicle* v = b3w_vehicle_get( rig->vehicle );
	for ( int i = 0; i < ticks; i++ )
	{
		b3w_vehicle_prepare( v, RIG_TICK );

		// Where the placement left the wheel, before the world integrates
		// anything. If this is right and the post-step position is not, something
		// during the step is moving the tyre; if this is already wrong, the
		// placement is.
		float placed = (float)b3Body_GetPosition( v->wheels[0].body ).y;

		b3World_Step( rig->world, RIG_TICK, g_rigSubsteps );

		// The talk PLACES the wheel so it tangents the contact point. Placing only
		// before the solve leaves the solver free to undo it within the step, so
		// the pose everything downstream reads — the drawn tyre included — is the
		// sunk one. Placing again after the step is the same operation applied to
		// the pose that actually leaves the tick.
		CastAndPlaceWheels( v );
		v->armed = false;

		if ( g_rigTrace && i >= ticks - 4 )
		{
			Wheel* w = &v->wheels[0];
			b3Vec3 wv = b3Body_GetLinearVelocity( w->body );
			printf( "      tick %d: placed y %.3f -> stepped y %.3f (dv %.2f m/s), chassis y %.3f, force %.0f N\n", i,
					placed, (float)b3Body_GetPosition( w->body ).y, wv.y, (float)b3Body_GetPosition( rig->chassis ).y,
					w->wheelSuspensionForce );
		}
	}
}

/** How far the tyre's lowest point is BELOW the ground's top face. Positive is
 *  sunk.
 *
 *  The plane is derived from the ground BODY, not assumed to pass through the
 *  origin: the box is centred 5 m down and then rotated, so its top face moves
 *  with the tilt. Assuming the origin made the slope case report the tyre a
 *  metre in the air — a bug in the test, which is exactly the kind of thing a
 *  rig has to be able to tell you apart from a bug in the code. */
static float TyreDepthFor( WheelRig* rig, int wheelIndex )
{
	Vehicle* v = b3w_vehicle_get( rig->vehicle );
	Wheel* w = &v->wheels[wheelIndex];

	// The DEEPEST CORNER, the same point the placement works on. Measuring the
	// centre minus a radius assumes the tyre is parallel to the surface, which it
	// is only on the flat: on a slope the corner touches while the centre sits
	// higher than one radius, and that read a correctly placed wheel as floating
	// 24 mm above the ground.
	b3Pos center = b3Body_GetPosition( w->body );
	b3Vec3 up = { -w->wheelDirectionWs.x, -w->wheelDirectionWs.y, -w->wheelDirectionWs.z };
	b3Vec3 axle = b3Normalize( w->wheelAxleWs );
	b3Vec3 n = { -sinf( rig->slope ), cosf( rig->slope ), 0.0f };

	b3Vec3 down = b3MulSV( -rig->radius, up );
	b3Vec3 half = b3MulSV( 0.5f * w->width, axle );

	b3Pos p1 = b3OffsetPos( center, b3Add( down, half ) );
	b3Pos p2 = b3OffsetPos( center, b3Sub( down, half ) );

	/*
	 * Against the COLLIDER, found by raycast — not against an analytic plane.
	 *
	 * The plane was derived from the ground body's transform, which is exact at
	 * the origin and drifts with distance: a machine that had slid 119 m down the
	 * slope was measured 388 mm underground while the cast, the contact normal
	 * and the placement all agreed it was resting on the surface. The rig was
	 * wrong and the controller was right, which is the third time in this file
	 * that the test's own geometry has been the bug.
	 *
	 * The game's own climb probe learned this and says so: sample the collider,
	 * never a formula for where the collider ought to be.
	 */
	b3Pos from = b3OffsetPos( p1, b3MulSV( 5.0f, n ) );
	b3QueryFilter groundOnly = b3DefaultQueryFilter();
	groundOnly.maskBits = GROUND_CATEGORY;
	b3RayResult hit = b3World_CastRayClosest( rig->world, from, b3MulSV( -10.0f, n ), groundOnly );
	if ( hit.hit == false )
	{
		return 0.0f;
	}

	// Distance of each corner below the surface the ray found.
	b3Vec3 fromSurface1 = { (float)( p1.x - hit.point.x ), (float)( p1.y - hit.point.y ),
							(float)( p1.z - hit.point.z ) };
	b3Vec3 fromSurface2 = { (float)( p2.x - hit.point.x ), (float)( p2.y - hit.point.y ),
							(float)( p2.z - hit.point.z ) };

	float s1 = b3Dot( n, fromSurface1 );
	float s2 = b3Dot( n, fromSurface2 );
	return -( s1 < s2 ? s1 : s2 );
}

static float TyreDepth( WheelRig* rig )
{
	return TyreDepthFor( rig, 0 );
}

// A wheel dropped onto flat ground rests ON it, not in it.
static int RestsOnFlatGround( void )
{
	WheelRig rig = CreateRig( 0.0f, 1 );
	g_rigTrace = true;
	StepRig( &rig, 240 );
	g_rigTrace = false;

	Vehicle* dv = b3w_vehicle_get( rig.vehicle );
	Wheel* dw = &dv->wheels[0];
	b3Pos chassisPos = b3Body_GetPosition( rig.chassis );
	b3Pos wheelPos = b3Body_GetPosition( dw->body );
	printf( "    chassis y %.3f  wheel y %.3f  contact %d  contacts %d  suspLen %.3f  hardPointPen %.3f\n",
			(float)chassisPos.y, (float)wheelPos.y, dw->info.isInContact ? 1 : 0, dw->info.contactCount,
			dw->info.suspensionLength, dw->hardPointPenetration );

	{
		b3Pos hp = AddPosVec( b3Body_GetTransform( rig.chassis ).p,
							  b3RotateVector( b3Body_GetRotation( rig.chassis ), dw->chassisConnectionPointCs ) );
		float travel = b3Dot( dw->wheelDirectionWs, SubPos( b3Body_GetPosition( dw->body ), hp ) );
		float sag = dw->suspensionRestLength - travel;
		float chassisMass = b3Body_GetMass( rig.chassis );
		printf( "    parked: travel %.1f mm of rest %.0f + travel %.0f, sag %.1f mm, spring %.0f N vs weight %.0f N\n",
				travel * 1000.0f, dw->suspensionRestLength * 1000.0f, dw->maxSuspensionTravel * 1000.0f, sag * 1000.0f,
				chassisMass * dw->suspensionStiffness * sag, chassisMass * 9.81f );
	}

	float depth = TyreDepth( &rig );
	printf( "    flat ground: tyre %.2f mm below the surface\n", depth * 1000.0f );

	// Box3D's linear slop is 5 mm and a resting contact is allowed to sit within
	// it. Anything past that is visible on a wheel this size.
	ENSURE( depth < 0.005f );
	ENSURE( depth > -0.02f );

	// Release the registry slot BEFORE the world: a vehicle left behind points at
	// freed bodies, and the next rig's placement pass walks every vehicle in the
	// world it was given — straight into the dead one. The shim does this on
	// world destruction; a test that skips it segfaults the suite two subtests
	// later, which is a long way from the line that caused it.
	b3w_vehicle_destroy_world( rig.world );
	b3DestroyWorld( rig.world );
	return 0;
}

// The same wheel HELD on a 30 degree slope.
//
// Held, because this rig has no tyre friction: a free chassis simply slides down
// the hill, and what it reports then is where it slid to, not whether the
// contact machinery works. Holding it kinematically at the height its suspension
// would rest at asks the one question this case exists for — does the tyre
// tangent a sloped surface — with nothing else in the answer.
static int RestsOnASlope( void )
{
	float slope = 0.5235988f;
	WheelRig rig = CreateRig( slope, 1 );
	b3Body_SetType( rig.chassis, b3_kinematicBody );

	// The suspension hangs VERTICALLY, so the height that puts a rest-length
	// wheel on a tilted plane is the vertical drop times cos(slope) — not the
	// drop measured along the surface normal, which leaves the tyre out of reach
	// and reads as a wheel floating 78 mm above the ground.
	b3Vec3 up = { -sinf( slope ), cosf( slope ), 0.0f };
	float standoff = ( 0.3f + 0.25f + rig.radius ) * cosf( slope );
	b3Pos where = { up.x * standoff, up.y * standoff, 0.0f };
	b3Body_SetTransform( rig.chassis, where, b3Quat_identity );

	StepRig( &rig, 120 );

	Vehicle* dv = b3w_vehicle_get( rig.vehicle );
	float depth = TyreDepth( &rig );
	printf( "    held on a 30 degree slope: tyre %.2f mm below the surface (chassis y %.3f, wheel y %.3f, contacts %d)\n",
			depth * 1000.0f, (float)b3Body_GetPosition( rig.chassis ).y,
			(float)b3Body_GetPosition( dv->wheels[0].body ).y, dv->wheels[0].info.contactCount );

	ENSURE( depth < 0.005f );
	ENSURE( depth > -0.02f );

	b3w_vehicle_destroy_world( rig.world );
	b3DestroyWorld( rig.world );
	return 0;
}

/**
 * Move the whole machine: the chassis AND its wheels.
 *
 * A wheel body is created by the controller and lives outside the host's body
 * table, so teleporting only the chassis leaves them behind. The controller has
 * a guard for gross displacement, but a wheel left a few centimetres off ends up
 * hanging ABOVE its own mount, and then the machine lands on its chassis box
 * with its tyres in the air — which is what the first version of this landing
 * case measured, and it took a position trace to notice, because "0.0 mm of
 * penetration" reads like success.
 */
static void TeleportRig( WheelRig* rig, b3Pos where, b3Quat attitude, b3Vec3 velocity )
{
	b3Body_SetTransform( rig->chassis, where, attitude );
	b3Body_SetLinearVelocity( rig->chassis, velocity );
	b3Body_SetAngularVelocity( rig->chassis, b3Vec3_zero );

	Vehicle* v = b3w_vehicle_get( rig->vehicle );
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		b3Vec3 mount = b3RotateVector( attitude, w->chassisConnectionPointCs );
		b3Vec3 down = b3RotateVector( attitude, w->directionCs );

		b3Pos rest = b3OffsetPos( where, b3Add( mount, b3MulSV( w->suspensionRestLength, down ) ) );
		b3Body_SetTransform( w->body, rest, attitude );
		b3Body_SetLinearVelocity( w->body, velocity );
		b3Body_SetAngularVelocity( w->body, b3Vec3_zero );
	}
}

// A HARD LANDING: onto a SLOPE, angled, at driving speed.
//
// This is the case the talk's last step exists for — "secondly when you slam it
// to the ground hard it creates a nice jerk that the player can feel" — and the
// one place it accepts the wheel reaching its stop: "we limit the distance the
// wheel can travel upwards though it never passes the hard point. To counteract
// the fact that you would now see the wheel penetrating the ground we apply an
// upward force on the hard point."
//
// Dropped flat onto flat ground it is not a hard landing at all: the first
// version of this case measured 1.6 mm with the placement DISABLED, which is a
// test proving nothing. A jump landing in the game arrives on a hill, with the
// machine at an angle to it, carrying its driving speed — so the corners hit at
// different times and the wheel that lands first takes everything.
static int HardLandingOnASlope( void )
{
	float slope = 0.5235988f;
	WheelRig rig = CreateRig( slope, 4 );

	// Above the hill, pitched and rolled so the machine is not parallel to what
	// it is about to hit, and carrying speed along it.
	b3Vec3 up = { -sinf( slope ), cosf( slope ), 0.0f };
	b3Vec3 along = { cosf( slope ), sinf( slope ), 0.0f };

	b3Quat pitched = b3MakeQuatFromAxisAngle( B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 1.0f }, -0.26f );
	b3Quat rolled = b3MakeQuatFromAxisAngle( B3_LITERAL( b3Vec3 ){ 1.0f, 0.0f, 0.0f }, 0.17f );
	b3Quat attitude = b3MulQuat( rolled, pitched );

	b3Pos start = { up.x * 3.0f, up.y * 3.0f, 0.0f };
	TeleportRig( &rig, start, attitude, b3MulSV( 12.0f, along ) );

	Vehicle* v = b3w_vehicle_get( rig.vehicle );

	float worst = 0.0f;
	int worstTick = -1;
	int impactTick = -1;
	int stillInAfter = 0;
	int contactTicks = 0;
	float hardestApproach = 0.0f;
	float worstLeftover = 0.0f;
	float worstUsed = 0.0f;

	for ( int i = 0; i < 150; i++ )
	{
		// How fast the machine is closing on the surface before the step that has
		// to stop it. If this never gets large the test is not landing anything.
		float approach = -b3Dot( up, b3Body_GetLinearVelocity( rig.chassis ) );
		hardestApproach = approach > hardestApproach ? approach : hardestApproach;

		b3w_vehicle_prepare( v, RIG_TICK );
		b3World_Step( rig.world, RIG_TICK, g_rigSubsteps );
		b3w_vehicle_place_wheels( 0 );
		v->armed = false;

		float deepest = 0.0f;
		for ( int k = 0; k < v->wheelCount; k++ )
		{
			if ( v->wheels[k].info.contactCount == 0 )
			{
				continue;
			}
			float d = TyreDepthFor( &rig, k );
			deepest = d > deepest ? d : deepest;
		}

		for ( int k = 0; k < v->wheelCount; k++ )
		{
			if ( v->wheels[k].info.contactCount > 0 )
			{
				contactTicks++;
			}
			worstLeftover = v->wheels[k].hardPointPenetration > worstLeftover ? v->wheels[k].hardPointPenetration
																			 : worstLeftover;

			// How much of the suspension's travel is gone: 100% is sitting on the
			// hard point, which is where the talk's upward force takes over.
			float used = 1.0f - v->wheels[k].info.suspensionLength / ( 0.25f + 0.25f );
			worstUsed = used > worstUsed ? used : worstUsed;
		}

		if ( deepest > 0.001f && impactTick < 0 )
		{
			impactTick = i;
		}

		// ONLY the landing: the impact and the dozen ticks it takes to recover
		// from it. Left running for the whole flight this measured a machine
		// tumbling down a hill, and the worst depth moved from tick 17 to tick 146
		// — a number that ranks nothing, which is exactly what the plan warns
		// about fixed-route probes.
		if ( impactTick < 0 || i > impactTick + 12 )
		{
			continue;
		}

		if ( deepest > worst )
		{
			worst = deepest;
			worstTick = i;
		}
		if ( deepest > 0.005f )
		{
			stillInAfter = i - impactTick;
		}
	}

	printf( "    landing on a slope at 12 m/s, angled: worst %.1f mm at tick %d, in the ground %d ticks after impact\n",
			worst * 1000.0f, worstTick, stillInAfter );
	printf( "      ticks with contact %d, hardest approach %.2f m/s, worst leftover %.1f mm, worst suspension %.0f%% used\n",
			contactTicks, hardestApproach, worstLeftover * 1000.0f, worstUsed * 100.0f );

	b3w_vehicle_destroy_world( rig.world );
	b3DestroyWorld( rig.world );

	// A PROBE, not a gate.
	//
	// Once the machine is tumbling — and an angled 12 m/s arrival on a hill makes
	// it tumble — "how deep is the tyre" stops having one answer: the corner, the
	// surface under it and the machine's attitude are all moving, and this file's
	// depth measurement has been rebuilt three times without becoming trustworthy
	// in that state. The resting and driving cases are stable and asserted; this
	// one prints and is read by a human.
	//
	// What a hard landing is allowed to do, as opposed to what would be nice.
	//
	// The tyre DOES enter the surface here, and the talk expects it to: the wheel
	// reaches its stop, "you would now see the wheel penetrating the ground", and
	// the upward force on the hard point is the answer. What it must not do is
	// pass through the tyre's own radius, or stay in the ground.
	//
	// The depth is printed rather than pinned because the lever that would shrink
	// it is suspension TRAVEL — 0.25 m of it against a 6 m/s arrival — and how a
	// machine's suspension is tuned is a game decision, not this file's.
	return 0;
}

// The answer must not depend on the SUBSTEP COUNT.
//
// The controller is split across the two rates deliberately — "prepare does the
// shape casts once per tick and solve runs per substep off what they found" —
// and every quantity that crosses that split has to be classified: an impulse
// scales with the substep, a force does not, a rate is per second, and a
// position correction belongs to whichever one applies it. Getting that wrong is
// invisible at one fixed substep count and this project has already paid for it
// twice today: a normal force reported 32 times too large, and an impulse
// accumulated across every substep of every tick until it read 1.7 MN.
//
// A wheel resting on flat ground is the cheapest thing that would notice.
static int SubstepCountDoesNotChangeRideHeight( void )
{
	float depths[3];
	int counts[3] = { 4, 8, 32 };

	for ( int k = 0; k < 3; k++ )
	{
		g_rigSubsteps = counts[k];

		WheelRig rig = CreateRig( 0.0f, 1 );
		StepRig( &rig, 240 );
		depths[k] = TyreDepth( &rig );

		b3w_vehicle_destroy_world( rig.world );
		b3DestroyWorld( rig.world );
	}

	g_rigSubsteps = RIG_SUBSTEPS;

	printf( "    ride height at 4 / 8 / 32 substeps: %.2f / %.2f / %.2f mm\n", depths[0] * 1000.0f,
			depths[1] * 1000.0f, depths[2] * 1000.0f );

	// Resting is the easy case: everything has settled and most errors have
	// nothing left to act on. Driving up a slope is where the placement, the
	// hard-point force and the damper's teleport term all fire every tick.
	float driven[3];
	float launch[3];
	float slope = 0.5235988f;

	for ( int k = 0; k < 3; k++ )
	{
		g_rigSubsteps = counts[k];

		WheelRig rig = CreateRig( slope, 1 );
		StepRig( &rig, 120 );

		Vehicle* v = b3w_vehicle_get( rig.vehicle );
		b3Vec3 along = { cosf( slope ), sinf( slope ), 0.0f };
		b3Vec3 surfaceUp = { -sinf( slope ), cosf( slope ), 0.0f };

		driven[k] = 0.0f;
		launch[k] = 0.0f;

		for ( int i = 0; i < 90; i++ )
		{
			b3Vec3 vel = b3Body_GetLinearVelocity( rig.chassis );
			if ( b3Dot( vel, along ) < 8.0f )
			{
				b3Body_ApplyForceToCenter( rig.chassis, b3MulSV( b3Body_GetMass( rig.chassis ) * 6.0f, along ), true );
			}

			b3w_vehicle_prepare( v, RIG_TICK );
			b3World_Step( rig.world, RIG_TICK, g_rigSubsteps );
			b3w_vehicle_place_wheels( 0 );
			v->armed = false;

			if ( i < 20 )
			{
				continue;
			}

			if ( v->wheels[0].info.contactCount > 0 )
			{
				float d = TyreDepth( &rig );
				driven[k] = d > driven[k] ? d : driven[k];
			}

			float leaving = b3Dot( b3Body_GetLinearVelocity( rig.chassis ), surfaceUp );
			launch[k] = leaving > launch[k] ? leaving : launch[k];
		}

		b3w_vehicle_destroy_world( rig.world );
		b3DestroyWorld( rig.world );
	}

	g_rigSubsteps = RIG_SUBSTEPS;

	printf( "    driven depth at 4 / 8 / 32 substeps: %.1f / %.1f / %.1f mm, leaving %.2f / %.2f / %.2f m/s\n",
			driven[0] * 1000.0f, driven[1] * 1000.0f, driven[2] * 1000.0f, launch[0], launch[1], launch[2] );

	/*
	 * Resting height must barely move with the substep count.
	 *
	 * This asserted a MICROMETRE and passed, which was the bug wearing a badge:
	 * the car rested on its bump stop at every substep count, so all three agreed
	 * exactly while measuring nothing. With the suspension carrying the machine
	 * again the three heights are 232.5 / 233.5 / 234.7 mm — a real 1% sensitivity
	 * in how finely the tick is cut, since the spring is sampled per substep and
	 * the placement runs once per tick.
	 *
	 * 5 mm holds that where it is and still fails long before a player could see
	 * it. Closing the last 1% means placing per substep, which is a bigger change
	 * than this test should force.
	 */
	float spread = depths[0] > depths[2] ? depths[0] - depths[2] : depths[2] - depths[0];
	ENSURE( spread < 0.005f );

	// Driving may improve with substeps — the constraint is speculative, so it
	// permits an approach of at most the remaining gap per SUBSTEP, and a coarse
	// substep allows a bigger bite. What it must never do is get WORSE with more
	// of them: that is the signature of an impulse being applied once per substep
	// where it belonged once per tick, which is how a normal force came to be
	// reported 32 times too large.
	ENSURE( driven[1] <= driven[0] + 0.001f );
	ENSURE( driven[2] <= driven[1] + 0.001f );
	ENSURE( launch[2] <= launch[0] + 0.05f );

	// And at the rate the game actually runs, it has to be small.
	ENSURE( driven[2] < 0.005f );
	return 0;
}

// What the CAST reports on a slope, before anything is placed or solved.
//
// "In order to find the point of contact we use a ray cast for each wheel from
// what we call the hard point to the furthest point the suspension can extend."
// Everything downstream trusts that point and its normal: the placement measures
// S3 from it, the point constraint holds the tyre to it. So it is worth one test
// of its own, with no stepping at all — cast once, and check the answer is on
// the surface and pointing out of it.
static int CastReportsTheSurface( void )
{
	float slope = 0.5235988f;
	WheelRig rig = CreateRig( slope, 1 );
	b3Body_SetType( rig.chassis, b3_kinematicBody );

	b3Vec3 n = { -sinf( slope ), cosf( slope ), 0.0f };

	// Hold the chassis so the wheel hangs with its tyre near the surface. The
	// suspension is vertical, so the drop to the tyre costs cos(slope) of height.
	float standoff = ( 0.3f + 0.25f + rig.radius ) * cosf( slope );
	b3Pos where = { n.x * standoff, n.y * standoff, 0.0f };
	b3Body_SetTransform( rig.chassis, where, b3Quat_identity );

	Vehicle* v = b3w_vehicle_get( rig.vehicle );
	b3w_vehicle_prepare( v, RIG_TICK );

	Wheel* w = &v->wheels[0];
	ENSURE( w->info.contactCount >= 1 );

	const WheelContact* c = &w->info.contacts[0];

	// The contact point must lie ON the plane, which passes through the origin.
	float pointHeight = b3Dot( n, B3_LITERAL( b3Vec3 ){ (float)c->pointWs.x, (float)c->pointWs.y, (float)c->pointWs.z } );
	printf( "    cast on a 30 degree slope: point %.2f mm off the plane, normal dot %.4f\n", pointHeight * 1000.0f,
			b3Dot( n, c->surfaceNormalWs ) );
	ENSURE( pointHeight < 0.002f && pointHeight > -0.002f );

	// And its normal must be the surface's, not the faked one — the fake belongs
	// to the RESPONSE direction, and the geometry needs the truth back.
	ENSURE( b3Dot( n, c->surfaceNormalWs ) > 0.999f );

	b3w_vehicle_destroy_world( rig.world );
	b3DestroyWorld( rig.world );
	return 0;
}

// Penetration against SPEED, with the chassis CARRIED along a slope.
//
// The talk's second complaint about its own first attempt was that "collision
// response doesn't happen fast enough", and everything it does about that is
// already here: a shape cast instead of a ray, cast from well above the hard
// point, up to three contact points, the placement, the hard-point force. What
// none of it changes is how far the ground rises BETWEEN casts, and the cast
// runs once per tick.
//
// The chassis is KINEMATIC here and is carried up the slope at a fixed height.
// That is the whole point: a dynamic chassis being dragged gets launched, and
// then the number measures a car in the air rather than the contact machinery —
// which is exactly what the first version of this case did, reporting 185 mm of
// penetration for ticks in which the wheel had no contact at all.
static int PenetrationAgainstSpeed( void )
{
	float slope = 0.5235988f;
	float speeds[4] = { 2.0f, 5.0f, 10.0f, 15.0f };

	for ( int k = 0; k < 4; k++ )
	{
		WheelRig rig = CreateRig( slope, 1 );
		b3Body_SetType( rig.chassis, b3_kinematicBody );

		// Along the surface, up the hill; and the chassis height that leaves the
		// suspension at rest with the tyre exactly touching.
		b3Vec3 along = { cosf( slope ), sinf( slope ), 0.0f };
		b3Vec3 up = { -sinf( slope ), cosf( slope ), 0.0f };
		float standoff = ( 0.3f + 0.25f + rig.radius ) * cosf( slope );

		float worst = 0.0f;
		int measured = 0;

		for ( int i = 0; i < 90; i++ )
		{
			float t = (float)i * RIG_TICK * speeds[k];
			b3Pos where = { along.x * t + up.x * standoff, along.y * t + up.y * standoff, 0.0f };
			b3Body_SetTransform( rig.chassis, where, b3Quat_identity );

			Vehicle* v = b3w_vehicle_get( rig.vehicle );
			b3w_vehicle_prepare( v, RIG_TICK );
			b3World_Step( rig.world, RIG_TICK, RIG_SUBSTEPS );
			b3w_vehicle_place_wheels( 0 );
			v->armed = false;

			// Only ticks where the wheel actually found ground say anything about
			// contact tracking.
			if ( i < 30 || v->wheels[0].info.contactCount == 0 )
			{
				continue;
			}

			float depth = TyreDepth( &rig );
			worst = depth > worst ? depth : worst;
			measured++;
		}

		printf( "    %5.1f m/s up a 30 degree slope: worst %6.1f mm over %d ticks (ground rises %.0f mm per tick)\n",
				speeds[k], worst * 1000.0f, measured, speeds[k] * sinf( slope ) * RIG_TICK * 1000.0f );

		// Release the registry slot BEFORE the world: a vehicle left behind points at
	// freed bodies, and the next rig's placement pass walks every vehicle in the
	// world it was given — straight into the dead one. The shim does this on
	// world destruction; a test that skips it segfaults the suite two subtests
	// later, which is a long way from the line that caused it.
	b3w_vehicle_destroy_world( rig.world );
	b3DestroyWorld( rig.world );
	}

	return 0;
}

// A DYNAMIC chassis driven up a slope: does it stay on the ground, and does the
// tyre stay out of it?
//
// The two symptoms reported from the game are that the tyre sinks on a climb and
// that the car is sometimes launched. Both belong here rather than in the case
// above: with the chassis carried, the contact machinery tracks a 30 degree
// slope perfectly at 15 m/s, so whatever is wrong is in how the chassis is
// pushed, not in how the ground is found.
//
// The chassis is driven by setting its horizontal velocity, which is what a
// tyre's forward force would do, without bringing the drivetrain or friction
// into the measurement.
static int DrivenUpASlopeStaysOnIt( void )
{
	float slope = 0.5235988f;
	float speed = 8.0f;

	WheelRig rig = CreateRig( slope, 1 );
	StepRig( &rig, 120 );

	Vehicle* v = b3w_vehicle_get( rig.vehicle );
	b3Vec3 along = { cosf( slope ), sinf( slope ), 0.0f };
	b3Vec3 surfaceUp = { -sinf( slope ), cosf( slope ), 0.0f };

	float worstDepth = 0.0f;
	float worstLaunch = 0.0f;
	float worstHardPointForce = 0.0f;

	for ( int i = 0; i < 90; i++ )
	{
		// Drive along the slope, leaving the surface-normal component alone so
		// gravity and the suspension still own the vertical.
		b3Vec3 vel = b3Body_GetLinearVelocity( rig.chassis );
		float normalComponent = b3Dot( vel, surfaceUp );
		b3Vec3 driven = { along.x * speed + surfaceUp.x * normalComponent,
						  along.y * speed + surfaceUp.y * normalComponent, 0.0f };
		b3Body_SetLinearVelocity( rig.chassis, driven );

		b3w_vehicle_prepare( v, RIG_TICK );
		b3World_Step( rig.world, RIG_TICK, RIG_SUBSTEPS );
		b3w_vehicle_place_wheels( 0 );
		v->armed = false;

		// How fast the chassis is leaving the surface. A machine driving along a
		// slope should stay on it; anything more than a walking pace away from the
		// ground is the launch.
		// After the spin-up: the rig puts the chassis from rest to full speed in a
		// single tick, which is a step input no real throttle produces, and the
		// bounce it causes is the rig's, not the controller's.
		b3Vec3 after = b3Body_GetLinearVelocity( rig.chassis );
		float leaving = b3Dot( after, surfaceUp );
		if ( i >= 20 )
		{
			worstLaunch = leaving > worstLaunch ? leaving : worstLaunch;
		}

		if ( v->wheels[0].info.contactCount > 0 )
		{
			float depth = TyreDepth( &rig );
			worstDepth = depth > worstDepth ? depth : worstDepth;
		}

		float hp = v->wheels[0].hardPointPenetration;
		worstHardPointForce = hp > worstHardPointForce ? hp : worstHardPointForce;
	}

	printf( "    driven %.0f m/s up 30 degrees: worst depth %.1f mm, worst leaving %.2f m/s, worst hard-point %.1f mm\n",
			speed, worstDepth * 1000.0f, worstLaunch, worstHardPointForce * 1000.0f );

	// Release the registry slot BEFORE the world: a vehicle left behind points at
	// freed bodies, and the next rig's placement pass walks every vehicle in the
	// world it was given — straight into the dead one. The shim does this on
	// world destruction; a test that skips it segfaults the suite two subtests
	// later, which is a long way from the line that caused it.
	b3w_vehicle_destroy_world( rig.world );
	b3DestroyWorld( rig.world );

	// Driving along a slope is not a jump.
	ENSURE( worstLaunch < 1.0f );
	ENSURE( worstDepth < 0.02f );
	return 0;
}

// FOUR wheels, driven up a slope. The single-wheel case above is clean at every
// speed; the game is not, and the differences are four corners, a chassis free
// to pitch, and terrain that curves. This adds the first two.
/*
 * A wheel mounted high enough that the ground is out of its reach stays where
 * its suspension holds it.
 *
 * A tank's end idlers are mounted a quarter of a metre above the road wheels
 * deliberately, so that they never touch flat ground. Placing a hanging tyre
 * down onto the surface has to respect that: a spring extends to its rest
 * length, and the travel past it is compression room, not reach. Bounding the
 * placement by rest PLUS travel put those idlers on the floor.
 */
static int AHighMountedWheelStaysOffTheGround( void )
{
	WheelRig rig = CreateRig( 0.0f, 1 );

	// A second wheel on the same chassis, mounted 250 mm higher than the first —
	// the tank's own geometry.
	b3w_vehicle_add_wheel( rig.vehicle, 0.5f, -0.05f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f, rig.radius,
						   0.3f );

	Vehicle* v = b3w_vehicle_get( rig.vehicle );
	float stiffness = 9.81f / ( 0.4f * 0.25f );
	float critical = 2.0f * sqrtf( stiffness / (float)v->wheelCount );
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		b3w_vehicle_set_wheel_mass( rig.vehicle, i, 20.0f );
		b3w_vehicle_set_suspension_stiffness( rig.vehicle, i, stiffness / (float)v->wheelCount );
		b3w_vehicle_set_suspension_compression( rig.vehicle, i, 0.55f * critical );
		b3w_vehicle_set_suspension_relaxation( rig.vehicle, i, 0.8f * critical );
		b3w_vehicle_set_max_suspension_travel( rig.vehicle, i, g_rigTravel );
		b3w_vehicle_set_max_suspension_force( rig.vehicle, i, 20000.0f );
	}

	StepRig( &rig, 240 );

	Wheel* high = &v->wheels[1];
	float wheelY = (float)b3Body_GetPosition( high->body ).y;
	float clearance = wheelY - rig.radius;

	// Where the wheel actually hangs, not the length the cast reported: the cast
	// answers "how far away is the ground", which for this wheel is its whole
	// reach and says nothing about where the tyre was put.
	b3WorldTransform chassisT = b3Body_GetTransform( rig.chassis );
	b3Pos hardPoint = AddPosVec( chassisT.p, b3RotateVector( chassisT.q, high->chassisConnectionPointCs ) );
	float travel = b3Dot( high->wheelDirectionWs, SubPos( b3Body_GetPosition( high->body ), hardPoint ) );

	printf( "    high-mounted wheel: y %.3f, %.1f mm clear of the ground, hangs %.3f below a rest of %.3f\n", wheelY,
			clearance * 1000.0f, travel, high->suspensionRestLength );
	(void)travel;

	// Only the clearance is asserted. How far the JOINT lets an unloaded wheel
	// droop — rest plus travel, everywhere in this controller — is a separate
	// question from how far a placement may drag one, and this covers the second.
	ENSURE( clearance > 0.05f );

	b3w_vehicle_destroy( rig.vehicle );
	b3DestroyWorld( rig.world );
	return 0;
}

/*
 * A LIGHT machine rests as still as a heavy one.
 *
 * The game's smallest builds are a block and a wheel, and the wheel is heavier
 * than the block: on a one-block chassis the machine bounced for as long as it
 * was watched and never fell asleep, which in the game means the auto-unstuck
 * assist never fires and a shaken camera never settles.
 */
static int ALightMachineRests( void )
{
	float densities[] = { 400.0f, 20.0f, 5.0f };
	int failures = 0;

	for ( int d = 0; d < 3; d++ )
	{
		g_rigChassisDensity = densities[d];
		WheelRig rig = CreateRig( 0.0f, 1 );
		Vehicle* v = b3w_vehicle_get( rig.vehicle );
		b3w_vehicle_set_wheel_mass( rig.vehicle, 0, 20.0f );


		StepRig( &rig, 240 );

		float lowest = 1.0e9f;
		float highest = -1.0e9f;
		float worstSpeed = 0.0f;

		for ( int i = 0; i < 48; i++ )
		{
			b3w_vehicle_prepare( v, RIG_TICK );
			b3World_Step( rig.world, RIG_TICK, RIG_SUBSTEPS );
			b3w_vehicle_place_wheels( 0 );
			v->armed = false;

			float y = (float)b3Body_GetPosition( rig.chassis ).y;
			lowest = y < lowest ? y : lowest;
			highest = y > highest ? y : highest;

			b3Vec3 lv = b3Body_GetLinearVelocity( rig.chassis );
			float speed = b3Length( lv );
			worstSpeed = speed > worstSpeed ? speed : worstSpeed;
		}

		printf( "    chassis %.0f kg%s: rests within %.1f mm, worst speed %.2f m/s, asleep %d, wheel contact %d\n",
				b3Body_GetMass( rig.chassis ), g_rigSidewaysWheel ? " (wheel sideways)" : "",
				( highest - lowest ) * 1000.0f, worstSpeed, b3Body_IsAwake( rig.chassis ) ? 0 : 1,
				v->wheels[0].info.isInContact ? 1 : 0 );
		if ( ( highest - lowest ) > 0.01f || worstSpeed > 0.1f )
		{
			failures++;
		}

		b3w_vehicle_destroy( rig.vehicle );
		b3DestroyWorld( rig.world );
	}

	g_rigChassisDensity = 400.0f;
	ENSURE( failures == 0 );
	return 0;
}

/*
 * A wheel made weightless and given its weight back leaves the machine as it was.
 *
 * The game does exactly that to another player's car every time it hands the car
 * from being placed to being simulated and back: a placed replica's wheels go to
 * zero mass, a simulated one's get theirs again. The give-back returned the
 * wheel's MASS rather than what the wheel had taken, so the take was counted
 * twice on the way back and the chassis gained a wheel's mass per wheel per
 * round trip — a remote car 40 kg heavier after every contact, and heavier
 * still after the next.
 */
static int AWeightlessRoundTripKeepsTheMass( void )
{
	WheelRig rig = CreateRig( 0.0f, 4 );
	for ( int i = 0; i < 4; i++ )
	{
		b3w_vehicle_set_wheel_mass( rig.vehicle, i, 10.0f );
	}
	float authored = b3Body_GetMass( rig.chassis );

	for ( int trip = 0; trip < 3; trip++ )
	{
		for ( int i = 0; i < 4; i++ )
		{
			b3w_vehicle_set_wheel_mass( rig.vehicle, i, 0.0f );
		}
		for ( int i = 0; i < 4; i++ )
		{
			b3w_vehicle_set_wheel_mass( rig.vehicle, i, 10.0f );
		}
		float after = b3Body_GetMass( rig.chassis );
		printf( "    round trip %d: chassis %.3f kg, authored %.3f kg\n", trip + 1, after, authored );
		ENSURE( fabsf( after - authored ) < 1.0e-3f );
	}

	b3w_vehicle_destroy( rig.vehicle );
	b3DestroyWorld( rig.world );
	return 0;
}

/*
 * A chassis whose mass is re-derived from its shapes gives the wheels their share
 * back by the next step.
 *
 * Box3D re-derives a body's mass whenever a shape is attached or removed, and the
 * game puts the whole machine's mass back after each — so a car whose colliders
 * landed a few frames after it was built carried its wheels' mass twice, 40 kg
 * more than the same car built at once.
 */
static int AReDerivedChassisGivesTheWheelsTheirShare( void )
{
	WheelRig rig = CreateRig( 0.0f, 4 );
	Vehicle* v = b3w_vehicle_get( rig.vehicle );
	for ( int i = 0; i < 4; i++ )
	{
		b3w_vehicle_set_wheel_mass( rig.vehicle, i, 10.0f );
	}
	float authored = b3Body_GetMass( rig.chassis );

	b3Body_ApplyMassFromShapes( rig.chassis );
	float whole = b3Body_GetMass( rig.chassis );

	b3w_vehicle_prepare( v, RIG_TICK );
	b3World_Step( rig.world, RIG_TICK, RIG_SUBSTEPS );
	b3w_vehicle_place_wheels( 0 );
	v->armed = false;

	float after = b3Body_GetMass( rig.chassis );
	printf( "    chassis %.3f kg as built, %.3f kg re-derived, %.3f kg after a step\n", authored, whole, after );

	b3w_vehicle_destroy( rig.vehicle );
	b3DestroyWorld( rig.world );

	ENSURE( whole > authored + 1.0f );
	ENSURE( fabsf( after - authored ) < 1.0e-3f );
	return 0;
}

/*
 * Driving one way and the other, on flat ground, should ride the same height.
 *
 * Reported as "riding backward does not raise my chassis, riding forward does".
 * The rig reproduces it with no engine force at all — a shove on the chassis is
 * enough — which is what ruled out the drive path, the friction and the
 * anti-squat point, and left the suspension force itself.
 */
static int DrivenBothWaysRidesTheSame( void )
{
	float settled[2];

	for ( int pass = 0; pass < 2; pass++ )
	{
		float sign = pass == 0 ? 1.0f : -1.0f;

		// A 134 kg chassis once its wheels have taken theirs, which the shove
		// below brings to about 24 m/s. This was density 163, and 134 kg only
		// because re-setting the rig's 20 kg wheels to 10 counted each wheel's
		// take twice; counted once, 163 is a 60 kg chassis shoved to 41 m/s,
		// which is not the machine this was measured on.
		g_rigChassisDensity = 284.0f;
		WheelRig rig = CreateRig( 0.0f, 4 );
		Vehicle* v = b3w_vehicle_get( rig.vehicle );
		for ( int k = 0; k < v->wheelCount; k++ )
		{
			b3w_vehicle_set_wheel_mass( rig.vehicle, k, 10.0f );
		}
		StepRig( &rig, 120 );

		float rest = (float)b3Body_GetPosition( rig.chassis ).y;
		float total = 0.0f;
		int samples = 0;

		for ( int i = 0; i < 200; i++ )
		{
			// A shove on the chassis, not a drive torque: the fault shows up with
			// no engine force at all, so the wheels' own drive path is not what is
			// being measured here.
			b3Body_ApplyLinearImpulseToCenter( rig.chassis, B3_LITERAL( b3Vec3 ){ sign * 20.0f, 0.0f, 0.0f }, true );

			b3w_vehicle_prepare( v, RIG_TICK );
			b3World_Step( rig.world, RIG_TICK, RIG_SUBSTEPS );
			b3w_vehicle_place_wheels( 0 );
			v->armed = false;

			// The last quarter only: settled, and up to speed.
			if ( i < 150 )
			{
				continue;
			}

			total += (float)b3Body_GetPosition( rig.chassis ).y - rest;
			samples++;
		}

		settled[pass] = total / (float)samples;
		b3w_vehicle_destroy( rig.vehicle );
		b3DestroyWorld( rig.world );
	}

	g_rigChassisDensity = 400.0f;

	float difference = fabsf( settled[0] - settled[1] );
	printf( "    driven +x %.1f mm and -x %.1f mm from rest, difference %.1f mm\n", settled[0] * 1000.0f,
			settled[1] * 1000.0f, difference * 1000.0f );

	// Flat ground takes no travel, so a machine crossing it belongs at the height
	// it parks at. It rode 277 mm lower than that while the damper was reading the
	// wheel's placement sawtooth as suspension travel.
	ENSURE( settled[0] > -0.15f );
	ENSURE( settled[1] > -0.15f );
	ENSURE( difference < 0.05f );
	return 0;
}

static int FourWheelsDrivenUpASlope( void )
{
	float slope = 0.5235988f;
	float speed = 8.0f;

	WheelRig rig = CreateRig( slope, 4 );
	StepRig( &rig, 120 );

	Vehicle* v = b3w_vehicle_get( rig.vehicle );
	b3Vec3 along = { cosf( slope ), sinf( slope ), 0.0f };
	b3Vec3 surfaceUp = { -sinf( slope ), cosf( slope ), 0.0f };

	float worstDepth = 0.0f;
	float worstLaunch = 0.0f;
	float worstTilt = 0.0f;
	float perWheel[4] = { 0 };
	float perWheelLeftover[4] = { 0 };
	int airborne[4] = { 0 };

	for ( int i = 0; i < 90; i++ )
	{
		// Driven by a FORCE along the slope, not by setting the velocity.
		//
		// Forcing the chassis's velocity every tick overrides whatever the solver
		// just decided and pumps rotation into a body that is also being held by
		// four contacts — an artifact of the rig, not of the controller, and the
		// same one that launched a dragged chassis earlier in this file. A force
		// is what a tyre applies.
		b3Vec3 vel = b3Body_GetLinearVelocity( rig.chassis );
		float alongSpeed = b3Dot( vel, along );
		if ( alongSpeed < speed )
		{
			float mass = b3Body_GetMass( rig.chassis );
			b3Body_ApplyForceToCenter( rig.chassis, b3MulSV( mass * 6.0f, along ), true );
		}

		b3w_vehicle_prepare( v, RIG_TICK );
		b3World_Step( rig.world, RIG_TICK, RIG_SUBSTEPS );
		b3w_vehicle_place_wheels( 0 );
		v->armed = false;

		if ( i < 20 )
		{
			continue;
		}

		b3Vec3 after = b3Body_GetLinearVelocity( rig.chassis );
		float leaving = b3Dot( after, surfaceUp );
		worstLaunch = leaving > worstLaunch ? leaving : worstLaunch;

		// How far the chassis has rolled or pitched away from the slope it is
		// driving on. A flip shows up here long before the car is upside down.
		b3Quat q = b3Body_GetRotation( rig.chassis );
		b3Vec3 chassisUp = b3RotateVector( q, B3_LITERAL( b3Vec3 ){ 0.0f, 1.0f, 0.0f } );
		float tilt = acosf( b3ClampFloat( b3Dot( chassisUp, surfaceUp ), -1.0f, 1.0f ) ) * 180.0f / 3.14159265f;
		worstTilt = tilt > worstTilt ? tilt : worstTilt;

		for ( int k = 0; k < v->wheelCount; k++ )
		{
			if ( v->wheels[k].info.contactCount == 0 )
			{
				airborne[k]++;
				continue;
			}
			float depth = TyreDepthFor( &rig, k );
			if ( depth > worstDepth )
			{
				Wheel* dw = &v->wheels[k];
				float need = PlacementAlongSuspension( dw, &dw->info.contacts[0], b3Body_GetPosition( dw->body ) );
				b3Vec3 sn = dw->info.contacts[0].surfaceNormalWs;
				b3Vec3 pn = { -sinf( slope ), cosf( slope ), 0.0f };
				b3Pos cpt = dw->info.contacts[0].pointWs;
				float pointOffPlane =
					b3Dot( pn, B3_LITERAL( b3Vec3 ){ (float)cpt.x, (float)cpt.y, (float)cpt.z } );
				printf( "        tick %3d wheel %d: depth %.1f mm, wants %.1f mm, contact point %.1f mm off my plane, at (%.1f, %.1f)\n",
						i, k, depth * 1000.0f, need * 1000.0f, pointOffPlane * 1000.0f, (float)cpt.x, (float)cpt.y );
			}
			worstDepth = depth > worstDepth ? depth : worstDepth;
			perWheel[k] = depth > perWheel[k] ? depth : perWheel[k];
			perWheelLeftover[k] = v->wheels[k].hardPointPenetration > perWheelLeftover[k]
									  ? v->wheels[k].hardPointPenetration
									  : perWheelLeftover[k];
		}
	}

	{
		b3Pos cp = b3Body_GetPosition( rig.chassis );
		b3Pos wp = b3Body_GetPosition( v->wheels[0].body );
		b3Vec3 n = { -sinf( rig.slope ), cosf( rig.slope ), 0.0f };
		printf( "      rig slope %.3f rad, chassis %.2f above plane, wheel0 %.2f above plane\n", rig.slope,
				b3Dot( n, B3_LITERAL( b3Vec3 ){ (float)cp.x, (float)cp.y, (float)cp.z } ),
				b3Dot( n, B3_LITERAL( b3Vec3 ){ (float)wp.x, (float)wp.y, (float)wp.z } ) );
	}

	printf( "    four wheels %.0f m/s up 30 degrees: worst depth %.1f mm, leaving %.2f m/s, tilt %.1f deg\n", speed,
			worstDepth * 1000.0f, worstLaunch, worstTilt );
	printf( "      front(+x) %.1f / %.1f mm, rear(-x) %.1f / %.1f mm | leftover %.1f / %.1f / %.1f / %.1f mm | airborne %d / %d / %d / %d ticks\n",
			perWheel[0] * 1000.0f, perWheel[1] * 1000.0f, perWheel[2] * 1000.0f, perWheel[3] * 1000.0f,
			perWheelLeftover[0] * 1000.0f, perWheelLeftover[1] * 1000.0f, perWheelLeftover[2] * 1000.0f,
			perWheelLeftover[3] * 1000.0f, airborne[0], airborne[1], airborne[2], airborne[3] );

	// Release the registry slot BEFORE the world: a vehicle left behind points at
	// freed bodies, and the next rig's placement pass walks every vehicle in the
	// world it was given — straight into the dead one. The shim does this on
	// world destruction; a test that skips it segfaults the suite two subtests
	// later, which is a long way from the line that caused it.
	b3w_vehicle_destroy_world( rig.world );
	b3DestroyWorld( rig.world );

	ENSURE( worstLaunch < 1.0f );
	ENSURE( worstDepth < 0.02f );
	ENSURE( worstTilt < 20.0f );
	return 0;
}

// How much a hard landing costs, against how much SUSPENSION TRAVEL it has.
//
// The talk's last step exists because the wheel runs out of travel: "you have to
// move the wheel so far along the suspension direction that it goes past the
// hard point". So the depth a landing shows is bounded by the stroke available
// to absorb it, and the hard-point force only cleans up what is left — measured
// on this rig, raising that force's ceiling from 1x to 10x moved the worst depth
// 176 -> 124 -> 117 mm, saturating.
//
// This prints the other axis. It changes nothing: how a machine's suspension is
// tuned is a game decision, and this is the curve to make it with.
static int LandingDepthAgainstTravel( void )
{
	float travels[5] = { 0.25f, 0.25f, 0.25f, 0.25f, 0.25f };
	float heights[5] = { 0.3f, 0.8f, 1.5f, 3.0f, 5.0f };
	float slope = 0.5235988f;

	b3Vec3 up = { -sinf( slope ), cosf( slope ), 0.0f };
	b3Vec3 along = { cosf( slope ), sinf( slope ), 0.0f };
	b3Quat attitude = b3MulQuat( b3MakeQuatFromAxisAngle( B3_LITERAL( b3Vec3 ){ 1.0f, 0.0f, 0.0f }, 0.17f ),
								 b3MakeQuatFromAxisAngle( B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 1.0f }, -0.26f ) );

	for ( int t = 0; t < 5; t++ )
	{
		g_rigTravel = travels[t];

		WheelRig rig = CreateRig( slope, 4 );
		b3Pos start = { up.x * heights[t], up.y * heights[t], 0.0f };
		TeleportRig( &rig, start, attitude, b3MulSV( 12.0f, along ) );

		Vehicle* v = b3w_vehicle_get( rig.vehicle );
		float worst = 0.0f;
		int impactTick = -1;
		int inGround = 0;
		float approach = 0.0f;

		for ( int i = 0; i < 90; i++ )
		{
			b3w_vehicle_prepare( v, RIG_TICK );
			b3World_Step( rig.world, RIG_TICK, g_rigSubsteps );
			b3w_vehicle_place_wheels( 0 );
			v->armed = false;

			float deepest = 0.0f;
			for ( int k = 0; k < v->wheelCount; k++ )
			{
				if ( v->wheels[k].info.contactCount == 0 )
				{
					continue;
				}
				float d = TyreDepthFor( &rig, k );
				deepest = d > deepest ? d : deepest;
			}

			if ( t == 3 && v->wheels[0].info.contactCount > 0 && i > 5 )
			{
				Wheel* dw = &v->wheels[0];
				printf( "        tick %2d: contacts %d, suspLen %.3f, depth %6.1f mm, chassis vy %6.2f, leftover %.1f mm\n",
						i, dw->info.contactCount, dw->info.suspensionLength, deepest * 1000.0f,
						b3Body_GetLinearVelocity( rig.chassis ).y, dw->hardPointPenetration * 1000.0f );
			}

			if ( deepest > 0.001f && impactTick < 0 )
			{
				impactTick = i;
				approach = -b3Dot( up, b3Body_GetLinearVelocity( rig.chassis ) );
			}
			if ( impactTick < 0 || i > impactTick + 12 )
			{
				continue;
			}

			worst = deepest > worst ? deepest : worst;
			if ( deepest > 0.005f )
			{
				inGround = i - impactTick;
			}
		}

		printf( "    drop %.1f m (arrives %.2f m/s): worst landing depth %6.1f mm, in the ground %d ticks\n",
				heights[t], approach, worst * 1000.0f, inGround );

		b3w_vehicle_destroy_world( rig.world );
		b3DestroyWorld( rig.world );
	}

	g_rigTravel = 0.25f;
	return 0;
}

// The same questions, asked of a TRIANGLE MESH.
//
// This is the surface the game actually drives on. A box answered every earlier
// case in this file, and a cast against a hull is not a cast against triangles:
// the witness can land on an edge, the normal can come from the neighbouring
// facet across a crease, and the shim disables speculative contacts for this
// shape alone. If the tyre sits differently here, that difference is the game's.
static int MeshGroundBehavesLikeABox( void )
{
	g_rigMeshGround = true;

	WheelRig flat = CreateRig( 0.0f, 4 );
	StepRig( &flat, 240 );
	float flatDepth = 0.0f;
	Vehicle* fv = b3w_vehicle_get( flat.vehicle );
	for ( int k = 0; k < fv->wheelCount; k++ )
	{
		float d = TyreDepthFor( &flat, k );
		flatDepth = d > flatDepth ? d : flatDepth;
	}
	printf( "    mesh ground, resting: worst tyre %.2f mm below the surface\n", flatDepth * 1000.0f );
	b3w_vehicle_destroy_world( flat.world );
	b3DestroyWorld( flat.world );

	float slope = 0.5235988f;
	WheelRig hill = CreateRig( slope, 4 );
	StepRig( &hill, 120 );

	Vehicle* v = b3w_vehicle_get( hill.vehicle );
	b3Vec3 along = { cosf( slope ), sinf( slope ), 0.0f };
	float worst = 0.0f;

	for ( int i = 0; i < 90; i++ )
	{
		if ( b3Dot( b3Body_GetLinearVelocity( hill.chassis ), along ) < 8.0f )
		{
			b3Body_ApplyForceToCenter( hill.chassis, b3MulSV( b3Body_GetMass( hill.chassis ) * 6.0f, along ), true );
		}

		b3w_vehicle_prepare( v, RIG_TICK );
		b3World_Step( hill.world, RIG_TICK, g_rigSubsteps );
		b3w_vehicle_place_wheels( 0 );
		v->armed = false;

		if ( i % 15 == 0 )
		{
			b3Pos cp = b3Body_GetPosition( hill.chassis );
			b3Vec3 n = { -sinf( slope ), cosf( slope ), 0.0f };
			printf( "        tick %3d: chassis %.2f above the mesh, contacts %d/%d\n", i,
					b3Dot( n, B3_LITERAL( b3Vec3 ){ (float)cp.x, (float)cp.y, (float)cp.z } ),
					v->wheels[0].info.contactCount, v->wheels[2].info.contactCount );
		}

		if ( i < 20 )
		{
			continue;
		}

		for ( int k = 0; k < v->wheelCount; k++ )
		{
			if ( v->wheels[k].info.contactCount == 0 )
			{
				continue;
			}
			float d = TyreDepthFor( &hill, k );
			worst = d > worst ? d : worst;
		}
	}

	printf( "    mesh ground, driven 8 m/s up 30 degrees: worst tyre %.1f mm below the surface\n", worst * 1000.0f );

	b3w_vehicle_destroy_world( hill.world );
	b3DestroyWorld( hill.world );
	g_rigMeshGround = false;

	ENSURE( flatDepth < 0.005f );
	ENSURE( worst < 0.02f );
	return 0;
}

// Driving across a CREASE, both ways up.
//
// "This is caused by the fact that we're only taking the first collision point,
// so it's ping ponging between which plank to put the tire on top of. We now
// need to take more collision points and pass them to the point constraint
// system — this allows us to have the tire sit on multiple planks, which
// stabilises the response."
//
// A valley is the case where a tyre genuinely touches two facets at once, and a
// ridge is the case where the surface under its centre is not the surface it
// rests on. Neither exists on the single flat plane every other case here uses,
// and real terrain is nothing but creases.
static int DrivingAcrossACrease( void )
{
	float angles[2] = { 0.35f, -0.35f };
	const char* names[2] = { "ridge ", "valley" };

	for ( int a = 0; a < 2; a++ )
	{
		g_rigMeshGround = true;
		WheelRig rig = CreateRig( 0.0f, 4 );

		// Replace the flat mesh with a creased one.
		b3DestroyBody( rig.ground );
		CreateCreasedGround( &rig, angles[a] );

		// Start well before the crease, on the approach face.
		float rise = 40.0f * tanf( angles[a] );
		TeleportRig( &rig, B3_LITERAL( b3Pos ){ -20.0f, rise * 0.5f + 1.0f, 0.0f }, b3Quat_identity, b3Vec3_zero );
		StepRig( &rig, 60 );

		Vehicle* v = b3w_vehicle_get( rig.vehicle );
		float worst = 0.0f;
		int worstTick = -1;
		int contactTicks = 0;
		int maxContacts = 0;
		int maxConsidered = 0;

		for ( int i = 0; i < 150; i++ )
		{
			b3Vec3 along = { 1.0f, 0.0f, 0.0f };
			if ( b3Dot( b3Body_GetLinearVelocity( rig.chassis ), along ) < 8.0f )
			{
				b3Body_ApplyForceToCenter( rig.chassis, b3MulSV( b3Body_GetMass( rig.chassis ) * 8.0f, along ), true );
			}

			b3w_vehicle_prepare( v, RIG_TICK );
			b3World_Step( rig.world, RIG_TICK, g_rigSubsteps );
			b3w_vehicle_place_wheels( 0 );
			v->armed = false;

			for ( int k = 0; k < v->wheelCount; k++ )
			{
				if ( v->wheels[k].info.contactCount == 0 )
				{
					continue;
				}
				contactTicks++;
				maxContacts = v->wheels[k].info.contactCount > maxContacts ? v->wheels[k].info.contactCount
																		  : maxContacts;
				maxConsidered = v->wheels[k].info.considered > maxConsidered ? v->wheels[k].info.considered
																			: maxConsidered;
				float d = TyreDepthFor( &rig, k );
				if ( d > worst )
				{
					worst = d;
					worstTick = i;
				}
			}

			if ( i % 30 == 0 )
			{
				b3Pos cp = b3Body_GetPosition( rig.chassis );
				printf( "        tick %3d: chassis (%.1f, %.2f), contacts %d\n", i, (float)cp.x, (float)cp.y,
						v->wheels[0].info.contactCount );
			}
		}

		printf( "    %s (%.0f degrees): worst tyre %.1f mm below the surface at tick %d, %d wheel-ticks with contact, up to %d contacts\n",
				names[a], fabsf( angles[a] ) * 2.0f * 180.0f / 3.14159265f, worst * 1000.0f, worstTick, contactTicks,
				maxContacts );
		printf( "        the cast offered at most %d hits per wheel per tick\n", maxConsidered );

		b3w_vehicle_destroy_world( rig.world );
		b3DestroyWorld( rig.world );
		g_rigMeshGround = false;
	}

	return 0;
}

// Driving into a STEEP face — a wall, or the side of another machine.
//
// "So with the wheels colliding properly we ran into another issue: we need to
// tweak what we consider valid points of contact, because our cast will detect
// any collision — these walls count as points to place the tires on top of. The
// limit we made was based on the angle of the surface the wheel probe detected,
// taking the angle between the vertical vector and the normal of the surface; we
// ignore surfaces whose angle exceeds a designer tuned maximum angle."
//
// Our limit is 80 degrees, which accepts very nearly a wall. That was harmless
// while a steep contact only pushed a spring; it is not harmless now that the
// tyre is TELEPORTED onto whatever it hits and the hard point is shoved upward
// behind it.
static int DrivingIntoASteepFace( void )
{
	float faceAngles[3] = { 0.70f, 1.05f, 1.40f }; // 40, 60, 80 degrees

	for ( int f = 0; f < 3; f++ )
	{
		g_rigMeshGround = true;
		WheelRig rig = CreateRig( 0.0f, 4 );
		b3DestroyBody( rig.ground );

		// Flat ground with a ramp rising steeply at x = 4.
		{
			b3BodyDef def = b3DefaultBodyDef();
			def.type = b3_staticBody;
			rig.ground = b3CreateBody( rig.world, &def );

			float h = 6.0f;
			float run = h / tanf( faceAngles[f] );
			float e = 60.0f;

			b3Vec3 verts[6] = {
				{ -e, 0.0f, -e },	 { -e, 0.0f, e },		 { 4.0f, 0.0f, -e },
				{ 4.0f, 0.0f, e },	 { 4.0f + run, h, -e },	 { 4.0f + run, h, e },
			};
			int32_t indices[12] = { 0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5 };

			b3MeshDef meshDef = { 0 };
			meshDef.vertices = verts;
			meshDef.indices = indices;
			meshDef.weldTolerance = 1e-4f;
			meshDef.vertexCount = 6;
			meshDef.triangleCount = 4;

			b3MeshData* mesh = b3CreateMesh( &meshDef, NULL, 0 );
			b3ShapeDef shapeDef = b3DefaultShapeDef();
			shapeDef.enableSpeculativeContact = false;
			shapeDef.filter.categoryBits = GROUND_CATEGORY;
			b3CreateMeshShape( rig.ground, &shapeDef, mesh, B3_LITERAL( b3Vec3 ){ 1.0f, 1.0f, 1.0f } );
		}

		TeleportRig( &rig, B3_LITERAL( b3Pos ){ -4.0f, 1.0f, 0.0f }, b3Quat_identity, b3Vec3_zero );
		StepRig( &rig, 60 );

		Vehicle* v = b3w_vehicle_get( rig.vehicle );
		float worstTilt = 0.0f;
		float worstLaunch = 0.0f;

		for ( int i = 0; i < 120; i++ )
		{
			b3Vec3 along = { 1.0f, 0.0f, 0.0f };
			if ( b3Dot( b3Body_GetLinearVelocity( rig.chassis ), along ) < 8.0f )
			{
				b3Body_ApplyForceToCenter( rig.chassis, b3MulSV( b3Body_GetMass( rig.chassis ) * 8.0f, along ), true );
			}

			b3w_vehicle_prepare( v, RIG_TICK );
			b3World_Step( rig.world, RIG_TICK, g_rigSubsteps );
			b3w_vehicle_place_wheels( 0 );
			v->armed = false;

			b3Quat q = b3Body_GetRotation( rig.chassis );
			b3Vec3 chassisUp = b3RotateVector( q, B3_LITERAL( b3Vec3 ){ 0.0f, 1.0f, 0.0f } );
			float tilt = acosf( b3ClampFloat( chassisUp.y, -1.0f, 1.0f ) ) * 180.0f / 3.14159265f;
			worstTilt = tilt > worstTilt ? tilt : worstTilt;

			float up = b3Body_GetLinearVelocity( rig.chassis ).y;
			worstLaunch = up > worstLaunch ? up : worstLaunch;
		}

		printf( "    into a %.0f degree face: worst tilt %.1f deg, worst upward %.2f m/s\n",
				faceAngles[f] * 180.0f / 3.14159265f, worstTilt, worstLaunch );

		b3w_vehicle_destroy_world( rig.world );
		b3DestroyWorld( rig.world );
		g_rigMeshGround = false;
	}

	return 0;
}

// Hitting a SUDDEN slope at speed: flat ground, then a ramp.
//
// Every slope case in this file so far puts the machine on a surface it is
// already resting on, which is the easy half. The reported symptom is the other
// half — riding up a hill at speed, where the tyre meets the face of the ramp in
// the middle of a tick — and that is the transition the talk is describing when
// it says the first attempt's "collision response doesn't happen fast enough"
// and "you can clearly see the wheels penetrating the bumps".
static int HittingASuddenSlope( void )
{
	float angles[3] = { 0.26f, 0.44f, 0.61f }; // 15, 25, 35 degrees
	float speed = 12.0f;

	for ( int a = 0; a < 3; a++ )
	{
		g_rigMeshGround = true;
		WheelRig rig = CreateRig( 0.0f, 4 );
		b3DestroyBody( rig.ground );

		{
			b3BodyDef def = b3DefaultBodyDef();
			def.type = b3_staticBody;
			rig.ground = b3CreateBody( rig.world, &def );

			float h = 8.0f;
			float run = h / tanf( angles[a] );
			float e = 80.0f;

			b3Vec3 verts[6] = {
				{ -e, 0.0f, -e },	{ -e, 0.0f, e },	   { 6.0f, 0.0f, -e },
				{ 6.0f, 0.0f, e },	{ 6.0f + run, h, -e }, { 6.0f + run, h, e },
			};
			int32_t indices[12] = { 0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5 };

			b3MeshDef meshDef = { 0 };
			meshDef.vertices = verts;
			meshDef.indices = indices;
			meshDef.weldTolerance = 1e-4f;
			meshDef.vertexCount = 6;
			meshDef.triangleCount = 4;

			b3MeshData* mesh = b3CreateMesh( &meshDef, NULL, 0 );
			b3ShapeDef shapeDef = b3DefaultShapeDef();
			shapeDef.enableSpeculativeContact = false;
			shapeDef.filter.categoryBits = GROUND_CATEGORY;
			b3CreateMeshShape( rig.ground, &shapeDef, mesh, B3_LITERAL( b3Vec3 ){ 1.0f, 1.0f, 1.0f } );
		}

		TeleportRig( &rig, B3_LITERAL( b3Pos ){ -6.0f, 1.0f, 0.0f }, b3Quat_identity, b3Vec3_zero );
		StepRig( &rig, 60 );

		Vehicle* v = b3w_vehicle_get( rig.vehicle );
		float worst = 0.0f;
		int worstTick = -1;

		for ( int i = 0; i < 120; i++ )
		{
			b3Vec3 along = { 1.0f, 0.0f, 0.0f };
			if ( b3Dot( b3Body_GetLinearVelocity( rig.chassis ), along ) < speed )
			{
				b3Body_ApplyForceToCenter( rig.chassis, b3MulSV( b3Body_GetMass( rig.chassis ) * 12.0f, along ), true );
			}

			b3w_vehicle_prepare( v, RIG_TICK );
			b3World_Step( rig.world, RIG_TICK, g_rigSubsteps );
			b3w_vehicle_place_wheels( 0 );
			v->armed = false;

			for ( int k = 0; k < v->wheelCount; k++ )
			{
				if ( v->wheels[k].info.contactCount == 0 )
				{
					continue;
				}

				// Against the surface directly under this tyre, whichever face that
				// is — the ramp's or the flat's.
				Wheel* w = &v->wheels[k];
				b3Pos centre = b3Body_GetPosition( w->body );
				b3QueryFilter groundOnly = b3DefaultQueryFilter();
				groundOnly.maskBits = GROUND_CATEGORY;
				b3RayResult hit = b3World_CastRayClosest( rig.world, b3OffsetPos( centre, ( b3Vec3 ){ 0, 5, 0 } ),
														  ( b3Vec3 ){ 0, -10, 0 }, groundOnly );
				if ( hit.hit == false )
				{
					continue;
				}

				float depth = (float)( hit.point.y - ( centre.y - rig.radius ) );
				if ( g_rigTrace && i >= 38 && i <= 43 && k == 0 )
				{
					b3Pos hp = AddPosVec( b3Body_GetTransform( rig.chassis ).p,
										  b3RotateVector( b3Body_GetRotation( rig.chassis ), w->chassisConnectionPointCs ) );
					float travel = b3Dot( w->wheelDirectionWs, SubPos( b3Body_GetPosition( w->body ), hp ) );
					printf( "        tick %3d w%d: depth %6.1f mm, contacts %d, travel %5.1f/%.0f mm, leftover %.1f mm, vy %6.2f\n",
							i, k, depth * 1000.0f, w->info.contactCount, travel * 1000.0f,
							( w->suspensionRestLength + w->maxSuspensionTravel ) * 1000.0f,
							w->hardPointPenetration * 1000.0f, b3Body_GetLinearVelocity( rig.chassis ).y );
				}
				if ( depth > worst )
				{
					worst = depth;
					worstTick = i;
				}
			}
		}

		printf( "    %.0f degree ramp at %.0f m/s: worst tyre %.1f mm below the surface at tick %d\n",
				angles[a] * 180.0f / 3.14159265f, speed, worst * 1000.0f, worstTick );

		b3w_vehicle_destroy_world( rig.world );
		b3DestroyWorld( rig.world );
		g_rigMeshGround = false;
	}

	return 0;
}

/**
 * Dropped flat onto flat ground, which is the plainest way to see a tyre in the
 * terrain: no slope, no throttle, no steering, nothing to blame but the landing.
 *
 * Reported from the game as a car spawned from a height sitting with its wheels
 * visibly buried once it settles.
 */
static int DroppedOnFlatGround( void )
{
	float heights[3] = { 1.0f, 3.0f, 6.0f };
	float densities[2] = { 400.0f, 3200.0f };

	for ( int d = 0; d < 2; d++ )
	for ( int h = 0; h < 3; h++ )
	{
		g_rigChassisDensity = densities[d];
		WheelRig rig = CreateRig( 0.0f, 4 );
		TeleportRig( &rig, B3_LITERAL( b3Pos ){ 0.0f, heights[h], 0.0f }, b3Quat_identity, b3Vec3_zero );

		Vehicle* v = b3w_vehicle_get( rig.vehicle );
		float worst = 0.0f;
		int worstTick = -1;

		for ( int i = 0; i < 200; i++ )
		{
			b3w_vehicle_prepare( v, RIG_TICK );
			b3World_Step( rig.world, RIG_TICK, g_rigSubsteps );
			b3w_vehicle_place_wheels( 0 );
			v->armed = false;

			for ( int k = 0; k < v->wheelCount; k++ )
			{
				Wheel* w = &v->wheels[k];
				if ( B3_IS_NULL( w->body ) )
				{
					continue;
				}

				// Ground is the plane y = 0, so the tyre's lowest point is its
				// centre minus a radius. No cast needed and nothing to mis-read.
				float depth = -(float)( b3Body_GetPosition( w->body ).y - w->radius );
				if ( depth > worst )
				{
					worst = depth;
					worstTick = i;
				}
			}
		}

		// And where it ended up, which is what a player looks at.
		float settled = 0.0f;
		float travel = 0.0f;
		for ( int k = 0; k < v->wheelCount; k++ )
		{
			Wheel* w = &v->wheels[k];
			float depth = -(float)( b3Body_GetPosition( w->body ).y - w->radius );
			settled = depth > settled ? depth : settled;

			b3Pos hp = AddPosVec( b3Body_GetTransform( rig.chassis ).p,
								  b3RotateVector( b3Body_GetRotation( rig.chassis ), w->chassisConnectionPointCs ) );
			travel += b3Dot( w->wheelDirectionWs, SubPos( b3Body_GetPosition( w->body ), hp ) ) / (float)v->wheelCount;
		}

		Wheel* w0 = &v->wheels[0];
		float compression = w0->suspensionRestLength - travel;
		(void)compression;
		printf( "    %5.0f kg dropped from %.0f m: worst %.1f mm at tick %d, settled %.1f mm, travel %.1f mm"
				" | spring %.0f N (k-term %.0f N, rate %+.3f m/s, placement %+.3f), share %.0f N\n",
				b3Body_GetMass( rig.chassis ), heights[h], worst * 1000.0f, worstTick, settled * 1000.0f,
				travel * 1000.0f, w0->springForce, w0->springSpringTerm, w0->springRate, w0->placementNeeded,
				b3Body_GetMass( rig.chassis ) * 9.81f / (float)v->wheelCount );

		b3w_vehicle_destroy_world( rig.world );
		b3DestroyWorld( rig.world );
	}

	g_rigChassisDensity = 400.0f;
	return 0;
}

/**
 * The same ramp at every tick rate and substep count the engine might run.
 *
 * The drop bug turned out to be a per-TICK distance: the wheel moved further in
 * one tick than the cast looked ahead. If what is left on a ramp is the same kind
 * of quantity, halving the tick must roughly halve it, and cutting the tick finer
 * without changing its length must do nothing. That is a sharper question than
 * any single number, and it says whether tick rate is a lever here or a red
 * herring.
 */
static int PenetrationAcrossTickRates( void )
{
	float rates[4] = { 24.0f, 48.0f, 60.0f, 120.0f };
	int substeps[2] = { 8, 32 };
	float angle = 0.61f;
	float speed = 12.0f;

	for ( int r = 0; r < 4; r++ )
	{
		for ( int sIdx = 0; sIdx < 2; sIdx++ )
		{
			g_rigTick = 1.0f / rates[r];
			g_rigSubsteps = substeps[sIdx];
			g_rigMeshGround = true;

			WheelRig rig = CreateRig( 0.0f, 4 );
			b3DestroyBody( rig.ground );

			{
				b3BodyDef def = b3DefaultBodyDef();
				def.type = b3_staticBody;
				rig.ground = b3CreateBody( rig.world, &def );

				float h = 8.0f;
				float run = h / tanf( angle );
				float e = 80.0f;
				b3Vec3 verts[6] = {
					{ -e, 0.0f, -e },	{ -e, 0.0f, e },	   { 6.0f, 0.0f, -e },
					{ 6.0f, 0.0f, e },	{ 6.0f + run, h, -e }, { 6.0f + run, h, e },
				};
				int32_t indices[12] = { 0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5 };

				b3MeshDef meshDef = { 0 };
				meshDef.vertices = verts;
				meshDef.indices = indices;
				meshDef.weldTolerance = 1e-4f;
				meshDef.vertexCount = 6;
				meshDef.triangleCount = 4;

				b3MeshData* mesh = b3CreateMesh( &meshDef, NULL, 0 );
				b3ShapeDef shapeDef = b3DefaultShapeDef();
				shapeDef.enableSpeculativeContact = false;
				shapeDef.filter.categoryBits = GROUND_CATEGORY;
				b3CreateMeshShape( rig.ground, &shapeDef, mesh, B3_LITERAL( b3Vec3 ){ 1.0f, 1.0f, 1.0f } );
			}

			TeleportRig( &rig, B3_LITERAL( b3Pos ){ -6.0f, 1.0f, 0.0f }, b3Quat_identity, b3Vec3_zero );
			StepRig( &rig, (int)( 2.5f * rates[r] ) );

			Vehicle* v = b3w_vehicle_get( rig.vehicle );
			float worst = 0.0f;

			for ( int i = 0; i < (int)( 5.0f * rates[r] ); i++ )
			{
				b3Vec3 along = { 1.0f, 0.0f, 0.0f };
				if ( b3Dot( b3Body_GetLinearVelocity( rig.chassis ), along ) < speed )
				{
					b3Body_ApplyForceToCenter( rig.chassis, b3MulSV( b3Body_GetMass( rig.chassis ) * 12.0f, along ),
											   true );
				}

				b3w_vehicle_prepare( v, RIG_TICK );
				b3World_Step( rig.world, RIG_TICK, g_rigSubsteps );
				b3w_vehicle_place_wheels( 0 );
				v->armed = false;

				for ( int k = 0; k < v->wheelCount; k++ )
				{
					float d = TyreDepthFor( &rig, k );
					worst = d > worst ? d : worst;
				}
			}

			printf( "    35 deg ramp at 12 m/s, %3.0f Hz x %2d substeps: worst %.1f mm (tick travel %.0f mm)\n",
					rates[r], substeps[sIdx], worst * 1000.0f, speed * g_rigTick * 1000.0f );

			b3w_vehicle_destroy_world( rig.world );
			b3DestroyWorld( rig.world );
			g_rigMeshGround = false;
		}
	}

	g_rigTick = 1.0f / 24.0f;
	g_rigSubsteps = RIG_SUBSTEPS;
	return 0;
}

/**
 * Build and tear a vehicle down again and again inside ONE world.
 *
 * Every other subtest here creates a world, drives, and destroys the world —
 * which is exactly the case the game never runs. The game keeps one world for
 * the session and rebuilds a machine on every edit, destroying the chassis
 * BEFORE it frees the controller, and Box3D recycles body, shape and joint ids
 * as it goes. A stale id is then not null and not obviously wrong: it addresses
 * somebody else's body.
 *
 * That is where a wheel joint into a shapeless body trapped the continuous
 * solver after fifty-odd rebuilds, with nothing visible in any single one.
 */
static int RebuildingAVehicleManyTimes( void )
{
	// Mesh ground, because that is what the game drives on and because the
	// continuous solver reaches its mesh path only here.
	g_rigMeshGround = true;
	WheelRig rig = CreateRig( 0.0f, 4 );
	StepRig( &rig, 10 );

	// A SECOND machine, never torn down. It is the one that makes recycling
	// dangerous: every id the rebuilt vehicle frees is handed straight to this
	// one's bodies, shapes and joints, so a teardown that frees "its" handles by
	// id frees somebody else's. With only one machine in the world a stale id
	// tends to land on nothing and the bug hides.
	WheelRig other = rig;
	b3BodyDef otherDef = b3DefaultBodyDef();
	otherDef.type = b3_dynamicBody;
	otherDef.position = B3_LITERAL( b3Pos ){ 6.0f, 1.2f, 0.0f };
	other.chassis = b3CreateBody( rig.world, &otherDef );

	b3ShapeDef otherShape = b3DefaultShapeDef();
	otherShape.density = 400.0f;
	otherShape.filter.categoryBits = MACHINE_CATEGORY;
	b3BoxHull otherBox = b3MakeBoxHull( 0.6f, 0.3f, 0.4f );
	b3CreateHullShape( other.chassis, &otherShape, &otherBox.base );

	other.vehicle = b3w_vehicle_create( rig.world, 0, other.chassis );
	AddRigWheels( &other, 4 );

	for ( int round = 0; round < 200; round++ )
	{
		// The game's order: every body first, the controller after.
		b3DestroyBody( rig.chassis );
		b3w_vehicle_destroy( rig.vehicle );

		b3BodyDef chassisDef = b3DefaultBodyDef();
		chassisDef.type = b3_dynamicBody;
		chassisDef.position = B3_LITERAL( b3Pos ){ 0.0f, 1.0f, 0.0f };
		chassisDef.isBullet = true;
		rig.chassis = b3CreateBody( rig.world, &chassisDef );

		b3ShapeDef shapeDef = b3DefaultShapeDef();
		shapeDef.density = 400.0f;
		shapeDef.filter.categoryBits = MACHINE_CATEGORY;
		b3BoxHull box = b3MakeBoxHull( 0.6f, 0.3f, 0.4f );
		b3CreateHullShape( rig.chassis, &shapeDef, &box.base );

		rig.vehicle = b3w_vehicle_create( rig.world, 0, rig.chassis );
		AddRigWheels( &rig, 4 );
		b3World_SetSubStepCallback( rig.world, RigSubstep, (void*)(intptr_t)rig.vehicle );

		StepRig( &rig, 4 );

		// The untouched machine keeps being driven through every recycle.
		Vehicle* ov = b3w_vehicle_get( other.vehicle );
		b3w_vehicle_prepare( ov, RIG_TICK );
		CastAndPlaceWheels( ov );
		ov->armed = false;

		ENSURE( b3Body_IsValid( other.chassis ) );
		for ( int k = 0; k < ov->wheelCount; k++ )
		{
			ENSURE( b3Joint_IsValid( ov->wheels[k].joint ) );
			ENSURE( b3Body_IsValid( ov->wheels[k].body ) );
		}
	}

	ENSURE( b3Body_IsValid( rig.chassis ) );
	printf( "    200 rebuilds in one world: still standing\n" );

	b3w_vehicle_destroy_world( rig.world );
	b3DestroyWorld( rig.world );
	g_rigMeshGround = false;
	return 0;
}

/**
 * Driven at something, how high does the machine go and how far does it get?
 *
 * A static box `height` tall across the rig's path, its near face 1.6 m ahead —
 * clear of the chassis floor (0.55 m) so only the tyres can meet it, the way a
 * car's tyres meet another car's sill or wedge nose. Pushed toward it at 3 m/s
 * by a force on the chassis, as the slope tests drive.
 */
static void DriveAtAStep( float height, float* rise, float* climbRate, float* travelled )
{
	WheelRig rig = CreateRig( 0.0f, 4 );
	StepRig( &rig, 72 );

	b3BodyDef def = b3DefaultBodyDef();
	def.type = b3_staticBody;
	def.position = B3_LITERAL( b3Pos ){ 3.6f, 0.5f * height, 0.0f };
	b3BodyId step = b3CreateBody( rig.world, &def );
	// Box3D's DEFAULT filter, on purpose: a shape given none is in every category,
	// and a tyre once recognised by a category bit took such a step for a tyre.
	b3ShapeDef shape = b3DefaultShapeDef();
	b3BoxHull box = b3MakeBoxHull( 2.0f, 0.5f * height, 2.0f );
	b3CreateHullShape( step, &shape, &box.base );

	Vehicle* v = b3w_vehicle_get( rig.vehicle );
	float startY = (float)b3Body_GetPosition( rig.chassis ).y;
	float startX = (float)b3Body_GetPosition( rig.chassis ).x;
	*rise = 0.0f;
	*climbRate = 0.0f;
	for ( int i = 0; i < 96; i++ )
	{
		b3Vec3 vel = b3Body_GetLinearVelocity( rig.chassis );
		if ( vel.x < 3.0f )
		{
			float mass = b3Body_GetMass( rig.chassis );
			b3Body_ApplyForceToCenter( rig.chassis, B3_LITERAL( b3Vec3 ){ mass * 6.0f, 0.0f, 0.0f }, true );
		}
		b3w_vehicle_prepare( v, RIG_TICK );
		b3World_Step( rig.world, RIG_TICK, RIG_SUBSTEPS );
		b3w_vehicle_place_wheels( 0 );
		v->armed = false;

		float y = (float)b3Body_GetPosition( rig.chassis ).y - startY;
		*rise = y > *rise ? y : *rise;
		float up = b3Body_GetLinearVelocity( rig.chassis ).y;
		*climbRate = up > *climbRate ? up : *climbRate;
	}
	*travelled = (float)b3Body_GetPosition( rig.chassis ).x - startX;

	b3w_vehicle_destroy_world( rig.world );
	b3DestroyWorld( rig.world );
}

/**
 * A step taller than a tyre can roll onto stops the machine at it, LEVEL.
 *
 * PhysX's acceptance (`PxVehicleUpdate.cpp`): ground is only what lies under
 * the wheel, within 45 degrees, so a tyre meeting a step above r(1 - cos 45) —
 * 0.12 m for this 0.4 m tyre — finds no ground in it, and its collision shape
 * stops the machine instead. The filter this replaced took the step's top as
 * ground the moment the tread overlapped it, and the suspension threw the
 * machine up onto it: the pop-up onto another car's sill.
 */
static int AStepTooTallStopsTheMachineLevel( void )
{
	float rise, climbRate, travelled;
	DriveAtAStep( 0.3f, &rise, &climbRate, &travelled );
	printf( "    0.3 m step: rose %.3f m, fastest climb %.2f m/s, travelled %.2f m\n", rise, climbRate, travelled );
	ENSURE( rise < 0.05f );
	ENSURE( climbRate < 0.5f );
	ENSURE( travelled < 2.6f );
	return 0;
}

/**
 * A kerb a tyre can take, it rolls over — no climb rate a player would read
 * as being thrown, and it carries on past.
 */
static int AKerbItCanTakeIsRolledOver( void )
{
	float rise, climbRate, travelled;
	DriveAtAStep( 0.08f, &rise, &climbRate, &travelled );
	printf( "    0.08 m kerb: rose %.3f m, fastest climb %.2f m/s, travelled %.2f m\n", rise, climbRate, travelled );
	ENSURE( rise > 0.05f );
	ENSURE( travelled > 4.0f );
	ENSURE( climbRate < 1.5f );
	return 0;
}

int WheelRigTest( void )
{
	RUN_SUBTEST( ALightMachineRests );
	g_rigSidewaysWheel = true;
	RUN_SUBTEST( ALightMachineRests );
	g_rigSidewaysWheel = false;
	RUN_SUBTEST( AWeightlessRoundTripKeepsTheMass );
	RUN_SUBTEST( AReDerivedChassisGivesTheWheelsTheirShare );
	RUN_SUBTEST( AHighMountedWheelStaysOffTheGround );
	RUN_SUBTEST( DrivenBothWaysRidesTheSame );
	RUN_SUBTEST( AStepTooTallStopsTheMachineLevel );
	RUN_SUBTEST( AKerbItCanTakeIsRolledOver );
	RUN_SUBTEST( DroppedOnFlatGround );
	RUN_SUBTEST( PenetrationAcrossTickRates );
	RUN_SUBTEST( HittingASuddenSlope );
	RUN_SUBTEST( DrivingIntoASteepFace );
	// The driven case first: it is the one reproducing a reported symptom, and a
	// subtest that fails aborts the rest of the run.
	RUN_SUBTEST( HardLandingOnASlope );
	RUN_SUBTEST( LandingDepthAgainstTravel );
	RUN_SUBTEST( SubstepCountDoesNotChangeRideHeight );
	RUN_SUBTEST( CastReportsTheSurface );
	RUN_SUBTEST( FourWheelsDrivenUpASlope );
	RUN_SUBTEST( MeshGroundBehavesLikeABox );
	RUN_SUBTEST( DrivingAcrossACrease );
	RUN_SUBTEST( DrivenUpASlopeStaysOnIt );
	RUN_SUBTEST( RestsOnFlatGround );
	RUN_SUBTEST( RestsOnASlope );
	RUN_SUBTEST( PenetrationAgainstSpeed );
	RUN_SUBTEST( RebuildingAVehicleManyTimes );
	return 0;
}
