// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/math_functions.h"

#include <math.h>
#include <stdio.h>

// A body's transform going non-finite, reproduced without a game attached.
//
// Found by the game's simulation fuzzer: after a machine is cut into pieces
// several times, one of the fragments ends up at NaN and STAYS there. A body at
// NaN is not a body that behaves badly — it is absent. It renders from a stale
// snapshot, stops nothing, answers no ray, and falls through the world. Players
// reported it as "my build keeps glitching and then i cant place things on it or
// move it not even colide with it".
//
// The game-side reproducer is 92 edit operations and resists reduction: ddmin
// plus a greedy sweep cannot remove a single one, which is what an accumulated
// state looks like. So the mechanism has to be found here instead, where a
// scenario is a dozen lines and a step is microseconds.
//
// What the game-side evidence says the mechanism is NOT:
//   - not the vehicle solver (stage-probed: the chassis is already non-finite
//     when b3w_vehicle_solve is entered)
//   - not a degenerate body (the one that went NaN had 3 shapes, mass 3.0 and a
//     healthy inertia tensor)
//   - not an extreme mass ratio (max/min across the structure was 1.0)
//   - not wheels (the wheel count never rose above one)
// What it IS: many small dynamic bodies, created overlapping, stepped.

// The game's numbers, so a result here means something there.
#define TICK_RATE 24.0f
#define SUB_STEPS 32
#define TICKS 240

static bool IsFinitePos( b3Pos p )
{
	return isfinite( p.x ) && isfinite( p.y ) && isfinite( p.z );
}

static bool IsFiniteVec( b3Vec3 v )
{
	return isfinite( v.x ) && isfinite( v.y ) && isfinite( v.z );
}

// Ground, then `count` unit boxes all created at the SAME place — which is what
// splitting a machine repeatedly leaves behind, since every fragment inherits
// the transform of the body it came out of.
static int RunCoincident( int count, float spread, const char* label )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -9.81f, 0.0f };
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.type = b3_staticBody;
	groundDef.position = (b3Pos){ 0.0f, -0.5f, 0.0f };
	b3BodyId ground = b3CreateBody( worldId, &groundDef );
	b3ShapeDef groundShape = b3DefaultShapeDef();
	b3BoxHull groundBox = b3MakeBoxHull( 50.0f, 0.5f, 50.0f );
	b3CreateHullShape( ground, &groundShape, &groundBox.base );

	enum
	{
		MAX_BODIES = 64
	};
	b3BodyId bodies[MAX_BODIES];
	if ( count > MAX_BODIES ) count = MAX_BODIES;

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	b3BoxHull box = b3MakeBoxHull( 0.125f, 0.125f, 0.125f );

	for ( int i = 0; i < count; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		// spread = 0 puts every body exactly on top of the others.
		bodyDef.position = (b3Pos){ spread * (float)i, 0.125f, 0.0f };
		bodies[i] = b3CreateBody( worldId, &bodyDef );
		b3CreateHullShape( bodies[i], &shapeDef, &box.base );
	}

	int firstBadTick = -1;
	int badBody = -1;
	for ( int tick = 0; tick < TICKS; ++tick )
	{
		b3World_Step( worldId, 1.0f / TICK_RATE, SUB_STEPS );

		for ( int i = 0; i < count && firstBadTick < 0; ++i )
		{
			b3Pos p = b3Body_GetPosition( bodies[i] );
			b3Vec3 v = b3Body_GetLinearVelocity( bodies[i] );
			if ( !IsFinitePos( p ) || !IsFiniteVec( v ) )
			{
				firstBadTick = tick;
				badBody = i;
			}
		}
		if ( firstBadTick >= 0 ) break;
	}

	if ( firstBadTick >= 0 )
	{
		b3Pos p = b3Body_GetPosition( bodies[badBody] );
		b3Vec3 v = b3Body_GetLinearVelocity( bodies[badBody] );
		printf( "    %s: body %d NON-FINITE at tick %d  pos=(%g, %g, %g) vel=(%g, %g, %g)\n", label, badBody, firstBadTick,
				p.x, p.y, p.z, v.x, v.y, v.z );
	}
	else
	{
		// How far the pile drifted, so a run that stayed finite still says
		// something about whether it stayed sane.
		float worst = 0.0f;
		for ( int i = 0; i < count; ++i )
		{
			b3Pos p = b3Body_GetPosition( bodies[i] );
			float d = b3Length( (b3Vec3){ p.x, p.y, p.z } );
			if ( d > worst ) worst = d;
		}
		printf( "    %s: finite after %d ticks, furthest body %.2f m from the origin\n", label, TICKS, worst );
	}

	b3DestroyWorld( worldId );
	return firstBadTick;
}

// Increasing overlap pressure: a pile that merely touches, then piles created
// exactly on top of each other in growing numbers.
static int TestCoincidentFragmentsStayFinite( void )
{
	int bad = 0;
	bad |= RunCoincident( 8, 0.25f, "8 bodies, touching   " ) >= 0;
	bad |= RunCoincident( 8, 0.0f, "8 bodies, coincident " ) >= 0;
	bad |= RunCoincident( 25, 0.0f, "25 bodies, coincident" ) >= 0;
	bad |= RunCoincident( 64, 0.0f, "64 bodies, coincident" ) >= 0;

	ENSURE( bad == 0 );
	return 0;
}


// Fragments JOINED to each other, which is what the game actually has: a
// machine's rigid groups are separate bodies held together by pivots, and a
// split hands those bodies to new structures without moving them apart.
//
// `frameOffset` of zero puts both joint frames at the same point, which is the
// degenerate case — a revolute whose anchors coincide has no lever arm to
// solve against.
static int RunJointedPile( int count, float spread, float frameOffset, const char* label )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -9.81f, 0.0f };
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.type = b3_staticBody;
	groundDef.position = (b3Pos){ 0.0f, -0.5f, 0.0f };
	b3BodyId ground = b3CreateBody( worldId, &groundDef );
	b3ShapeDef groundShape = b3DefaultShapeDef();
	b3BoxHull groundBox = b3MakeBoxHull( 50.0f, 0.5f, 50.0f );
	b3CreateHullShape( ground, &groundShape, &groundBox.base );

	enum
	{
		MAX_JOINTED = 32
	};
	b3BodyId bodies[MAX_JOINTED];
	if ( count > MAX_JOINTED ) count = MAX_JOINTED;

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	b3BoxHull box = b3MakeBoxHull( 0.125f, 0.125f, 0.125f );

	for ( int i = 0; i < count; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = (b3Pos){ spread * (float)i, 0.125f, 0.0f };
		bodies[i] = b3CreateBody( worldId, &bodyDef );
		b3CreateHullShape( bodies[i], &shapeDef, &box.base );
	}

	for ( int i = 0; i + 1 < count; ++i )
	{
		b3RevoluteJointDef def = b3DefaultRevoluteJointDef();
		def.base.bodyIdA = bodies[i];
		def.base.bodyIdB = bodies[i + 1];
		def.base.localFrameA = (b3Transform){ { frameOffset, 0.0f, 0.0f }, b3Quat_identity };
		def.base.localFrameB = (b3Transform){ { -frameOffset, 0.0f, 0.0f }, b3Quat_identity };
		b3CreateRevoluteJoint( worldId, &def );
	}

	int firstBadTick = -1;
	int badBody = -1;
	for ( int tick = 0; tick < TICKS && firstBadTick < 0; ++tick )
	{
		b3World_Step( worldId, 1.0f / TICK_RATE, SUB_STEPS );
		for ( int i = 0; i < count; ++i )
		{
			b3Pos p = b3Body_GetPosition( bodies[i] );
			b3Vec3 v = b3Body_GetLinearVelocity( bodies[i] );
			if ( !IsFinitePos( p ) || !IsFiniteVec( v ) )
			{
				firstBadTick = tick;
				badBody = i;
				break;
			}
		}
	}

	if ( firstBadTick >= 0 )
	{
		b3Pos p = b3Body_GetPosition( bodies[badBody] );
		printf( "    %s: body %d NON-FINITE at tick %d  pos=(%g, %g, %g)\n", label, badBody, firstBadTick, p.x, p.y, p.z );
	}
	else
	{
		float worst = 0.0f;
		for ( int i = 0; i < count; ++i )
		{
			b3Pos p = b3Body_GetPosition( bodies[i] );
			float d = b3Length( (b3Vec3){ p.x, p.y, p.z } );
			if ( d > worst ) worst = d;
		}
		printf( "    %s: finite after %d ticks, furthest %.2f m\n", label, TICKS, worst );
	}

	b3DestroyWorld( worldId );
	return firstBadTick;
}

static int TestJointedFragmentsStayFinite( void )
{
	int bad = 0;
	bad |= RunJointedPile( 8, 0.25f, 0.125f, "8 jointed, apart          " ) >= 0;
	bad |= RunJointedPile( 8, 0.0f, 0.125f, "8 jointed, coincident     " ) >= 0;
	bad |= RunJointedPile( 25, 0.0f, 0.125f, "25 jointed, coincident    " ) >= 0;
	bad |= RunJointedPile( 8, 0.0f, 0.0f, "8 jointed, zero-length    " ) >= 0;
	bad |= RunJointedPile( 25, 0.0f, 0.0f, "25 jointed, zero-length   " ) >= 0;

	ENSURE( bad == 0 );
	return 0;
}

int SplitFragmentsTest( void )
{
	RUN_SUBTEST( TestCoincidentFragmentsStayFinite );
	RUN_SUBTEST( TestJointedFragmentsStayFinite );
	return 0;
}
