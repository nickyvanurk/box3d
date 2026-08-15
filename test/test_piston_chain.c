// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/math_functions.h"

#include <stdio.h>

// A chain of prismatic joints driven by motors while a large external force
// pulls on the far end — a machine built out of pistons, which is what a player
// makes when they build a crane arm.
//
// The failure this reproduces is not a joint yielding under load. A soft
// constraint yields in proportion to the force on it and settles. This one
// ACCELERATES: once the joint frames separate, the lever arm in the axial
// Jacobian grows with the separation, the effective mass goes ill-conditioned,
// and the correction feeds the error it is meant to remove. The joint ends up
// many times past its own hard stop and keeps going.

#define CHAIN_LINKS 7

// The game's numbers, so a fix here means something there: 24 Hz ticks with 32
// solver substeps, pistons limited to 2.5 m with 3000 N motors.
#define TICK_RATE 24.0f
#define SUB_STEPS 32
#define PISTON_LIMIT 2.5f
#define PISTON_FORCE 3000.0f
#define PISTON_SPEED 0.5f

// A quarter of the substep rate is the stiffest constraint the solver will
// accept; b3PrepareJoint clamps anything above it. Asking for the ceiling means
// a failure here cannot be answered with "tune it higher".
#define CONSTRAINT_HERTZ ( 0.25f * TICK_RATE * SUB_STEPS )

typedef struct PistonChain
{
	b3WorldId worldId;
	b3BodyId bodies[CHAIN_LINKS + 1];
	b3JointId joints[CHAIN_LINKS];
} PistonChain;

// Anchored base, then a vertical stack of links joined by driven prismatic
// joints. The top link is heavy — a grabber on the end of an arm.
static PistonChain CreatePistonChainOf( int links, float motorForce, float hertz )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -9.81f, 0.0f };

	PistonChain c;
	c.worldId = b3CreateWorld( &worldDef );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	b3BoxHull box = b3MakeBoxHull( 0.5f, 0.125f, 0.5f );

	for ( int i = 0; i <= links; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = i == 0 ? b3_staticBody : b3_dynamicBody;
		bodyDef.position = (b3Pos){ 0.0f, 0.25f * (float)i, 0.0f };
		c.bodies[i] = b3CreateBody( c.worldId, &bodyDef );

		b3ShapeDef s = shapeDef;
		// The grabber is the heavy thing the arm is carrying.
		s.density = i == links ? 20.0f : 1.0f;
		b3CreateHullShape( c.bodies[i], &s, &box.base );
	}

	for ( int i = 0; i < links; ++i )
	{
		b3PrismaticJointDef def = b3DefaultPrismaticJointDef();
		def.base.bodyIdA = c.bodies[i];
		def.base.bodyIdB = c.bodies[i + 1];
		def.base.localFrameA = (b3Transform){ { 0.0f, 0.125f, 0.0f }, b3Quat_identity };
		def.base.localFrameB = (b3Transform){ { 0.0f, -0.125f, 0.0f }, b3Quat_identity };
		def.base.constraintHertz = hertz;
		def.base.constraintDampingRatio = 2.0f;

		def.enableLimit = true;
		def.lowerTranslation = 0.0f;
		def.upperTranslation = PISTON_LIMIT;

		def.enableMotor = true;
		def.motorSpeed = PISTON_SPEED;
		def.maxMotorForce = motorForce;
		def.enableMotor = motorForce > 0.0f;

		c.joints[i] = b3CreatePrismaticJoint( c.worldId, &def );
	}

	return c;
}

static float RunChain( int links, float force, float motorForce, float hertz, int subSteps )
{
	PistonChain c = CreatePistonChainOf( links, motorForce, hertz );
	float worst = 0.0f;

	for ( int step = 0; step < 400; ++step )
	{
		b3Body_ApplyForceToCenter( c.bodies[links], (b3Vec3){ 0.0f, force, 0.0f }, true );
		b3World_Step( c.worldId, 1.0f / TICK_RATE, subSteps );

		for ( int i = 0; i < links; ++i )
		{
			float t = b3PrismaticJoint_GetTranslation( c.joints[i] );
			if ( t > worst ) worst = t;
		}
	}

	b3DestroyWorld( c.worldId );
	return worst;
}

// Working loads: a chain of driven pistons carrying a heavy end must sit ON its
// stop, not past it. Before the limit was solved hard this measured 2.655 m at
// the mildest load here and ran away to 1434 m at the strongest, so the bound
// below is a real guard rather than a restatement of current behaviour.
static int TestPistonChainHoldsItsLimit( void )
{
	int links[] = { 1, 2, 4, 7 };
	float worst = 0.0f;

	for ( int li = 0; li < 4; ++li )
	{
		for ( float force = 0.0f; force <= 4000.0f; force += 2000.0f )
		{
			float t = RunChain( links[li], force, PISTON_FORCE, CONSTRAINT_HERTZ, SUB_STEPS );
			if ( t > worst ) worst = t;
		}
	}

	printf( "  piston chain: worst translation %.3f m against a %.3f m limit\n", worst, PISTON_LIMIT );
	ENSURE( worst < PISTON_LIMIT * 1.02f );
	return 0;
}

// The same chain under a load far past anything the game applies — 800 g on the
// end mass. Reported, not asserted: it is the headroom left over, and the number
// to watch if this ever needs to go further.
static int TestPistonChainUnderExtremeLoad( void )
{
	printf( "  extreme load headroom: %.3f m (7 links, 40 kN)\n",
			RunChain( 7, 40000.0f, PISTON_FORCE, CONSTRAINT_HERTZ, SUB_STEPS ) );
	return 0;
}

int PistonChainTest( void )
{
	RUN_SUBTEST( TestPistonChainHoldsItsLimit );
	RUN_SUBTEST( TestPistonChainUnderExtremeLoad );
	return 0;
}
