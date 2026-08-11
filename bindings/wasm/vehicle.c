/**
 * A cylinder-shapecast vehicle controller, ported from
 * `rapier/src/control/shape_cast_vehicle_controller.rs`.
 *
 * That controller is not upstream Bullet or upstream Rapier: it carries a dozen
 * commits of this game's own development — anti-squat, suspension reject angle,
 * a direct-solve bump stop, static longitudinal friction with a position
 * anchor, airborne droop, side friction shared across grounded wheels, and not
 * driving a sleeping chassis. Every one of those exists because a car did
 * something wrong on screen, so this is a translation that preserves them
 * line for line rather than a reimplementation of "a vehicle controller".
 *
 * It lives in C, next to the engine, because the alternative is crossing the
 * WASM boundary once per wheel per step for the shapecast alone — four to eight
 * crossings per vehicle per step, against a direct call here.
 *
 * **The one place the geometry is not identical, and it is deliberate.** Parry
 * casts a true cylinder. Box3D casts a point cloud with a radius, which can
 * express a capsule exactly but not a cylinder — a two-point proxy with radius
 * r rounds the tread and would change ride height wherever a wheel meets an
 * edge. So the wheel is cast as a polygonal prism: two rings of
 * `TREAD_SEGMENTS` points at the tyre's outer radius, zero proxy radius. The
 * error is r·(1 − cos(π/TREAD_SEGMENTS)), which at 8 segments is 1.9% of the
 * radius and under Box3D's 5 mm linear slop for any wheel the game builds.
 */

#include "box3d/box3d.h"
#include "box3d/math_functions.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

#define B3W_MAX_WHEELS 32
/** Points per cap of the cast prism. See the header note on tread geometry. */
#define TREAD_SEGMENTS 8

/** Below this chassis speed a wheel with no drive or brake grips longitudinally. */
static const float STATIC_FRICTION_MAX_SPEED = 0.5f;
/** Fraction of anchor drift undone per step — the usual Baumgarte trade. */
static const float STATIC_FRICTION_ERP = 0.2f;
/** Furthest a stuck contact may drift before the tyre is treated as having slipped. */
static const float MAX_STATIC_FRICTION_SLIP = 0.05f;

typedef struct
{
	b3Vec3 contactNormalWs;
	b3Vec3 groundNormalWs;
	b3Pos contactPointWs;
	float suspensionLength;
	b3Pos hardPointWs;
	bool isInContact;
	/** Body index of the ground for reporting, or -1. */
	int groundBody;
	/** The ground body itself, so the dynamic-ground branches never look one up. */
	b3BodyId groundBodyId;
	bool bottomedOut;
	float bottomOutOvershoot;
	float lastToi;
	float lastCastOffset;
	b3Pos lastCastStart;
	b3Vec3 lastAxleWs;
} ShapeCastInfo;

typedef struct
{
	ShapeCastInfo info;

	b3Pos center;
	b3Vec3 wheelDirectionWs;
	b3Vec3 wheelAxleWs;

	b3Vec3 chassisConnectionPointCs;
	b3Vec3 directionCs;
	b3Vec3 axleCs;

	float suspensionRestLength;
	float maxSuspensionTravel;
	float radius;
	float width;
	float suspensionStiffness;
	float dampingCompression;
	float dampingRelaxation;
	float frictionSlip;
	float sideFrictionStiffness;
	float maxSuspensionForce;

	float rotation;
	float deltaRotation;
	float rollInfluence;
	float antiSquat;
	float suspensionRejectAngle;
	float surfaceRejectAngle;

	float forwardImpulse;
	float sideImpulse;

	float steering;
	float engineForce;
	float brake;

	bool hasAnchor;
	b3Pos staticFrictionAnchor;

	float clippedInvContactDotSuspension;
	float suspensionRelativeVelocity;
	float wheelSuspensionForce;
	float skidInfo;
} Wheel;

typedef struct
{
	bool used;
	b3WorldId world;
	b3BodyId chassis;
	int wheelCount;
	Wheel wheels[B3W_MAX_WHEELS];
	b3Vec3 forwardWs[B3W_MAX_WHEELS];
	b3Vec3 axle[B3W_MAX_WHEELS];
	float currentVehicleSpeed;
	int indexUpAxis;
	int indexForwardAxis;

	/** Body indices the wheels may not stand on — own structure and the player. */
	int excludeLo;
	int excludeHi;
} Vehicle;

static b3Vec3 AxisVector( int index )
{
	b3Vec3 v = { 0.0f, 0.0f, 0.0f };
	if ( index == 0 )
		v.x = 1.0f;
	else if ( index == 1 )
		v.y = 1.0f;
	else
		v.z = 1.0f;
	return v;
}

static b3Vec3 SubPos( b3Pos a, b3Pos b )
{
	return B3_LITERAL( b3Vec3 ){ (float)( a.x - b.x ), (float)( a.y - b.y ), (float)( a.z - b.z ) };
}

static b3Pos AddPosVec( b3Pos p, b3Vec3 v )
{
	return B3_LITERAL( b3Pos ){ p.x + v.x, p.y + v.y, p.z + v.z };
}

static b3Vec3 Scale( b3Vec3 v, float s )
{
	return B3_LITERAL( b3Vec3 ){ v.x * s, v.y * s, v.z * s };
}

static b3Vec3 Add3( b3Vec3 a, b3Vec3 b )
{
	return B3_LITERAL( b3Vec3 ){ a.x + b.x, a.y + b.y, a.z + b.z };
}

static b3Vec3 Sub3( b3Vec3 a, b3Vec3 b )
{
	return B3_LITERAL( b3Vec3 ){ a.x - b.x, a.y - b.y, a.z - b.z };
}

/** Normalize, or return the input untouched when it is too short to trust. */
static b3Vec3 TryNormalize( b3Vec3 v, b3Vec3 fallback )
{
	float len = b3Length( v );
	if ( len < 1.0e-5f )
	{
		return fallback;
	}
	return Scale( v, 1.0f / len );
}

/**
 * World-space inverse inertia tensor.
 *
 * Box3D reports the local tensor about the centre of mass; the controller needs
 * it in world space for every impulse denominator, which is R·I⁻¹·Rᵀ.
 */
static b3Matrix3 WorldInvInertia( b3BodyId body )
{
	// b3Body_GetMassData reports the tensor about the body's centre of mass in
	// its LOCAL frame; every impulse denominator here needs it in world space,
	// which is R·I⁻¹·Rᵀ applied column by column.
	b3MassData md = b3Body_GetMassData( body );
	b3Matrix3 invLocal = b3InvertMatrix( md.inertia );
	b3Quat q = b3Body_GetRotation( body );

	b3Vec3 basis[3] = { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };
	b3Matrix3 out;
	b3Vec3 cols[3];
	for ( int i = 0; i < 3; i++ )
	{
		b3Vec3 local = b3InvRotateVector( q, basis[i] );
		cols[i] = b3RotateVector( q, b3MulMV( invLocal, local ) );
	}
	out.cx = cols[0];
	out.cy = cols[1];
	out.cz = cols[2];
	return out;
}

/**
 * World-space centre of mass.
 *
 * Box3D exposes only the LOCAL centre (`b3Body_GetLocalCenter`), so this is the
 * body transform applied to it. Rapier hands back the world point directly, and
 * assuming the same here would have put every impulse lever arm at the body
 * origin instead — silently wrong on any chassis whose mass is not centred.
 */
static b3Pos CenterOfMass( b3BodyId body )
{
	b3WorldTransform t = b3Body_GetTransform( body );
	b3Vec3 local = b3Body_GetLocalCenter( body );
	b3Vec3 world = b3RotateVector( t.q, local );
	return AddPosVec( t.p, world );
}

static float InvMass( b3BodyId body )
{
	float m = b3Body_GetMass( body );
	return m > 0.0f ? 1.0f / m : 0.0f;
}

/** `1/x`, with Rapier's convention that the inverse of zero is zero. */
static float SafeInv( float x )
{
	return x != 0.0f ? 1.0f / x : 0.0f;
}

// ---------------------------------------------------------------------------
// Shape cast
// ---------------------------------------------------------------------------

typedef struct
{
	/** The vehicle's own chassis, which can never be its own ground. */
	b3BodyId chassis;
	int excludeLo;
	int excludeHi;
	bool hit;
	float fraction;
	b3Vec3 normal;
	b3Pos point;
	int body;
	b3BodyId bodyId;
} WheelCast;

static int BodyIndexOfShapeV( b3ShapeId shapeId )
{
	intptr_t tagged = (intptr_t)b3Body_GetUserData( b3Shape_GetBody( shapeId ) );
	return tagged > 0 ? (int)( tagged - 1 ) : -1;
}

static float OnWheelHit( b3ShapeId shapeId, b3Pos point, b3Vec3 normal, float fraction, uint64_t userMaterialId,
						 int triangleIndex, int childIndex, void* context )
{
	(void)userMaterialId;
	(void)triangleIndex;
	(void)childIndex;

	WheelCast* ctx = (WheelCast*)context;
	int body = BodyIndexOfShapeV( shapeId );
	b3BodyId hitBody = b3Shape_GetBody( shapeId );

	// The cast starts one radius ABOVE the hard point, which is inside the
	// chassis, so the chassis is the first thing every wheel hits. Rapier drops
	// that with `stop_at_penetration: false`; Box3D has no such option, so the
	// chassis is skipped by identity — it can never be its own ground.
	if ( B3_ID_EQUALS( hitBody, ctx->chassis ) )
	{
		return -1.0f;
	}

	// The vehicle's own structure and the player are not ground either. Return
	// -1 so the cast keeps looking past them rather than stopping.
	if ( body >= 0 && body >= ctx->excludeLo && body <= ctx->excludeHi )
	{
		return -1.0f;
	}

	ctx->hit = true;
	ctx->fraction = fraction;
	ctx->normal = normal;
	ctx->point = point;
	ctx->body = body;
	ctx->bodyId = b3Shape_GetBody( shapeId );
	return fraction;
}

/**
 * Cast one wheel and fill its `ShapeCastInfo`.
 *
 * A line-for-line port of `shape_cast`, including both rejection filters and
 * the droop-toward-full-extension branch when nothing is hit.
 */
static void ShapeCastWheel( Vehicle* v, int wheelId, float dt )
{
	b3Vec3 worldUp = AxisVector( v->indexUpAxis );
	Wheel* w = &v->wheels[wheelId];
	b3Pos source = w->info.hardPointWs;

	b3Vec3 axle = TryNormalize( w->wheelAxleWs, B3_LITERAL( b3Vec3 ){ 0.0f, 1.0f, 0.0f } );
	b3Vec3 direction = TryNormalize( w->wheelDirectionWs, w->wheelDirectionWs );

	// Start one radius behind the hard point so the prism's near cap sits at
	// it, and grow the cast length by the same offset so detection range is
	// unchanged. suspension_length = toi − offset.
	float r = w->radius;
	float raylen = w->suspensionRestLength + r;
	float offset = r;

	b3Pos castStart = AddPosVec( source, Scale( direction, -offset ) );

	// Build the prism: two rings of points around the axle, at the tyre radius.
	b3Vec3 up = fabsf( axle.y ) < 0.9f ? B3_LITERAL( b3Vec3 ){ 0.0f, 1.0f, 0.0f } : B3_LITERAL( b3Vec3 ){ 1.0f, 0.0f, 0.0f };
	b3Vec3 t1 = b3Normalize( b3Cross( axle, up ) );
	b3Vec3 t2 = b3Cross( axle, t1 );

	b3Vec3 points[2 * TREAD_SEGMENTS];
	float half = w->width * 0.5f;
	for ( int i = 0; i < TREAD_SEGMENTS; i++ )
	{
		float a = ( 2.0f * 3.14159265358979f * (float)i ) / (float)TREAD_SEGMENTS;
		b3Vec3 radial = Add3( Scale( t1, r * cosf( a ) ), Scale( t2, r * sinf( a ) ) );
		points[i] = Add3( radial, Scale( axle, half ) );
		points[i + TREAD_SEGMENTS] = Add3( radial, Scale( axle, -half ) );
	}

	b3ShapeProxy proxy;
	proxy.points = points;
	proxy.count = 2 * TREAD_SEGMENTS;
	proxy.radius = 0.0f;

	WheelCast cast = { .chassis = v->chassis,
					   .excludeLo = v->excludeLo,
					   .excludeHi = v->excludeHi,
					   .hit = false,
					   .fraction = 1.0f,
					   .body = -1,
					   .bodyId = b3_nullBodyId };

	float maxToi = raylen + offset;
	b3World_CastShape( v->world, castStart, &proxy, Scale( direction, maxToi ), b3DefaultQueryFilter(), OnWheelHit,
					   &cast );

	// Filter 1: chassis pose. Drop the hit when the suspension does not point
	// broadly downward — sideways, nose-standing or inverted, where a
	// heightfield's always-outward normal would mislead a normal-only test.
	float suspensionDownward = -b3Dot( direction, worldUp );
	bool accepted = cast.hit && suspensionDownward > cosf( w->suspensionRejectAngle );

	// Filter 2: surface. Drivability is a world-space property of the surface,
	// so this compares against world up rather than the suspension direction —
	// otherwise chassis tilt would move the accept boundary around.
	if ( accepted )
	{
		float worldUpFacing = b3Dot( cast.normal, worldUp );
		accepted = worldUpFacing > cosf( w->surfaceRejectAngle );
	}

	w->info.groundBody = -1;
	w->info.groundBodyId = b3_nullBodyId;

	if ( accepted )
	{
		// Suspension and friction both use the real ground normal so the spring
		// stays aligned with the contact plane on slopes; fall back to the
		// suspension axis when the hit is more than ~60° off it, which is what
		// stops the spring shoving the chassis sideways off a vertical face.
		b3Vec3 axial = Scale( w->wheelDirectionWs, -1.0f );
		b3Vec3 groundNormal = b3Dot( cast.normal, axial ) > 0.5f ? cast.normal : axial;

		w->info.contactNormalWs = groundNormal;
		w->info.groundNormalWs = groundNormal;
		w->info.isInContact = true;
		w->info.groundBody = cast.body;
		w->info.groundBodyId = cast.bodyId;

		float toi = cast.fraction * maxToi;
		float rawLength = toi - offset;
		w->info.lastToi = toi;
		w->info.lastCastOffset = offset;
		w->info.lastCastStart = castStart;
		w->info.lastAxleWs = axle;

		// The wheel cannot travel past the hard point — the spring has a
		// mechanical end stop, and a negative raw length means the chassis is
		// overshooting into the ground and must be absorbed rigidly.
		float maxSuspensionLength = w->suspensionRestLength + w->maxSuspensionTravel;
		float targetLength = rawLength < 0.0f ? 0.0f : ( rawLength > maxSuspensionLength ? maxSuspensionLength : rawLength );
		w->info.bottomedOut = rawLength < 0.0f;
		w->info.bottomOutOvershoot = rawLength < 0.0f ? -rawLength : 0.0f;
		w->info.suspensionLength = targetLength;
		w->info.contactPointWs = cast.point;

		float denominator = b3Dot( w->info.contactNormalWs, w->wheelDirectionWs );
		b3Vec3 velAtContact = b3Body_GetWorldPointVelocity( v->chassis, w->info.contactPointWs );
		float projVel = b3Dot( w->info.contactNormalWs, velAtContact );

		if ( denominator >= -0.1f )
		{
			w->suspensionRelativeVelocity = 0.0f;
			w->clippedInvContactDotSuspension = 1.0f / 0.1f;
		}
		else
		{
			float inv = -1.0f / denominator;
			w->suspensionRelativeVelocity = projVel * inv;
			w->clippedInvContactDotSuspension = inv;
		}
	}
	else
	{
		// Airborne: droop toward the full extension stop rather than rest, so
		// there is little to catch up on when the ground returns. ~200 ms TC.
		float droopLength = w->suspensionRestLength + w->maxSuspensionTravel;
		float timeConstant = 0.2f;
		float factor = 1.0f - expf( -dt / timeConstant );
		w->info.suspensionLength += ( droopLength - w->info.suspensionLength ) * factor;
		w->info.bottomedOut = false;
		w->info.bottomOutOvershoot = 0.0f;
		w->suspensionRelativeVelocity = 0.0f;
		w->info.contactNormalWs = Scale( w->wheelDirectionWs, -1.0f );
		w->info.groundNormalWs = w->info.contactNormalWs;
		w->clippedInvContactDotSuspension = 1.0f;
	}
}

// ---------------------------------------------------------------------------
// Suspension
// ---------------------------------------------------------------------------

static void UpdateSuspension( Vehicle* v, float chassisMass, float dt )
{
	// Every grounded wheel damps the SAME chassis, so applied explicitly the
	// damping overshoots by roughly the wheel count and flips sign each step —
	// the vehicle rocks between its wheels forever. Folding the damper into an
	// implicit update is unconditionally stable and leaves the steady state
	// (and so the ride height) untouched.
	int contacts = 0;
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		if ( v->wheels[i].info.isInContact )
			contacts++;
	}
	if ( contacts < 1 )
		contacts = 1;

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];

		if ( !w->info.isInContact )
		{
			w->wheelSuspensionForce = 0.0f;
			continue;
		}

		float lengthDiff = w->suspensionRestLength - w->info.suspensionLength;
		float force = w->suspensionStiffness * lengthDiff * w->clippedInvContactDotSuspension;

		float projectedRelVel = w->suspensionRelativeVelocity;
		float suspDamping = projectedRelVel < 0.0f ? w->dampingCompression : w->dampingRelaxation;
		float implicit = 1.0f / ( 1.0f + dt * suspDamping * (float)contacts );
		force -= suspDamping * projectedRelVel * implicit;

		float result = force * chassisMass;
		w->wheelSuspensionForce = result > 0.0f ? result : 0.0f;
	}
}

// ---------------------------------------------------------------------------
// Friction
// ---------------------------------------------------------------------------

static float ImpulseDenominator( b3BodyId body, b3Pos pos, b3Vec3 n )
{
	b3Vec3 dpt = SubPos( pos, CenterOfMass( body ) );
	b3Vec3 gcross = b3Cross( dpt, n );
	b3Matrix3 invI = WorldInvInertia( body );
	b3Vec3 v = b3Cross( b3MulMV( invI, gcross ), dpt );
	return InvMass( body ) + b3Dot( n, v );
}

static float ResolveSingleUnilateral( b3BodyId body, b3Pos pt, b3Vec3 normal, float biasVelocity )
{
	b3Vec3 vel = b3Body_GetWorldPointVelocity( body, pt );
	b3Vec3 dpt = SubPos( pt, CenterOfMass( body ) );
	b3Vec3 aj = b3Cross( dpt, normal );
	b3Vec3 iaj = b3MulMV( WorldInvInertia( body ), aj );

	float jacDiagAb = InvMass( body ) + b3Dot( iaj, iaj );
	float jacDiagAbInv = SafeInv( jacDiagAb );
	float relVel = b3Dot( normal, vel );

	float contactDamping = 0.2f;
	return -contactDamping * ( relVel + biasVelocity ) * jacDiagAbInv;
}

static float ResolveSingleBilateral( b3BodyId body1, b3Pos pt1, b3BodyId body2, b3Pos pt2, b3Vec3 normal,
									 float biasVelocity )
{
	b3Vec3 vel1 = b3Body_GetWorldPointVelocity( body1, pt1 );
	b3Vec3 vel2 = b3Body_GetWorldPointVelocity( body2, pt2 );
	b3Vec3 dvel = Sub3( vel1, vel2 );

	b3Vec3 dpt1 = SubPos( pt1, CenterOfMass( body1 ) );
	b3Vec3 dpt2 = SubPos( pt2, CenterOfMass( body2 ) );
	b3Vec3 aj = b3Cross( dpt1, normal );
	b3Vec3 bj = b3Cross( dpt2, Scale( normal, -1.0f ) );
	b3Vec3 iaj = b3MulMV( WorldInvInertia( body1 ), aj );
	b3Vec3 ibj = b3MulMV( WorldInvInertia( body2 ), bj );

	float jacDiagAb = InvMass( body1 ) + InvMass( body2 ) + b3Dot( iaj, iaj ) + b3Dot( ibj, ibj );
	float jacDiagAbInv = SafeInv( jacDiagAb );
	float relVel = b3Dot( normal, dvel );

	float contactDamping = 0.2f;
	return -contactDamping * ( relVel + biasVelocity ) * jacDiagAbInv;
}

/**
 * Longitudinal friction demand, UNCLAMPED.
 *
 * Whether the tyre can actually deliver it is the caller's Coulomb test — a
 * value clamped here could not be told apart from one that merely happens to
 * sit at the limit, and that distinction is what lets a tyre break traction.
 */
static float CalcRollingFriction( b3BodyId chassis, b3BodyId ground, b3Pos contactPos, b3Vec3 dir,
								  int numWheelsOnGround, float biasVelocity )
{
	float denom0 = ImpulseDenominator( chassis, contactPos, dir );
	float denom1 = 0.0f;
	bool hasGround = B3_IS_NON_NULL( ground ) && b3Body_GetType( ground ) == b3_dynamicBody;

	if ( hasGround )
	{
		denom1 = ImpulseDenominator( ground, contactPos, dir );
	}

	float jacDiagAbInv = 1.0f / ( denom0 + denom1 );

	b3Vec3 vel1 = b3Body_GetWorldPointVelocity( chassis, contactPos );
	b3Vec3 vel2 = hasGround ? b3Body_GetWorldPointVelocity( ground, contactPos ) : B3_LITERAL( b3Vec3 ){ 0, 0, 0 };
	b3Vec3 vel = Sub3( vel1, vel2 );
	float vrel = b3Dot( dir, vel );

	return -( vrel + biasVelocity ) * jacDiagAbInv / (float)numWheelsOnGround;
}

/**
 * Longitudinal and lateral tyre friction.
 *
 * A port of `update_friction`, keeping its order: latch or release each wheel's
 * static-friction anchor, build the contact frame from the GROUND normal (not
 * the suspension axis, or drive force lifts a pitched chassis), solve lateral,
 * then longitudinal, then apply the friction circle, then apply the impulses.
 */
static void UpdateFriction( Vehicle* v, float dt )
{
	int numWheels = v->wheelCount;
	if ( numWheels == 0 )
	{
		return;
	}

	float vehicleSpeed = fabsf( v->currentVehicleSpeed );
	int numWheelsOnGround = 0;

	for ( int i = 0; i < numWheels; i++ )
	{
		Wheel* w = &v->wheels[i];
		bool hasGround = w->info.groundBody >= 0;

		if ( hasGround )
		{
			numWheelsOnGround++;
		}

		// Decided once, because BOTH friction axes need it: they are the same
		// velocity-only constraint and drift the same way. A driven wheel is
		// meant to travel, and a world anchor on a moving body is meaningless,
		// so neither gets one.
		bool groundIsDynamic = hasGround && B3_IS_NON_NULL( w->info.groundBodyId ) &&
							   b3Body_GetType( w->info.groundBodyId ) == b3_dynamicBody;
		bool canStick = hasGround && w->engineForce == 0.0f && vehicleSpeed < STATIC_FRICTION_MAX_SPEED &&
						!groundIsDynamic;

		if ( canStick )
		{
			if ( !w->hasAnchor )
			{
				w->hasAnchor = true;
				w->staticFrictionAnchor = w->info.contactPointWs;
			}

			// A tyre sticks only up to a finite slip distance; past that it has
			// broken traction and grips somewhere new. Without this the anchor
			// hauls a slow vehicle back to where it stood seconds ago.
			b3Vec3 drift = SubPos( w->info.contactPointWs, w->staticFrictionAnchor );
			if ( b3Length( drift ) > MAX_STATIC_FRICTION_SLIP )
			{
				w->staticFrictionAnchor = w->info.contactPointWs;
			}
		}
		else
		{
			w->hasAnchor = false;
		}

		w->sideImpulse = 0.0f;
		w->forwardImpulse = 0.0f;
	}

	for ( int i = 0; i < numWheels; i++ )
	{
		Wheel* w = &v->wheels[i];
		if ( w->info.groundBody < 0 )
		{
			continue;
		}

		// Build the contact frame from the real surface normal, so drive force
		// stays in the contact plane instead of tilting with the chassis.
		b3Vec3 surfNormalWs = w->info.groundNormalWs;
		b3Vec3 axle = w->wheelAxleWs;
		axle = Sub3( axle, Scale( surfNormalWs, b3Dot( axle, surfNormalWs ) ) );
		axle = TryNormalize( axle, B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.0f } );
		v->axle[i] = axle;
		v->forwardWs[i] = TryNormalize( b3Cross( surfNormalWs, axle ), B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.0f } );

		// How far the anchored contact has slid sideways, as the velocity
		// needed to undo it this step.
		float lateralBias = 0.0f;
		if ( w->hasAnchor )
		{
			b3Vec3 slid = SubPos( w->info.contactPointWs, w->staticFrictionAnchor );
			lateralBias = b3Dot( slid, v->axle[i] ) * STATIC_FRICTION_ERP / dt;
		}

		bool groundIsDynamic = B3_IS_NON_NULL( w->info.groundBodyId ) &&
							   b3Body_GetType( w->info.groundBodyId ) == b3_dynamicBody;

		if ( groundIsDynamic )
		{
			w->sideImpulse = ResolveSingleBilateral( v->chassis, w->info.contactPointWs, w->info.groundBodyId,
													w->info.contactPointWs, v->axle[i], 0.0f );
		}
		else
		{
			w->sideImpulse =
				ResolveSingleUnilateral( v->chassis, w->info.contactPointWs, v->axle[i], lateralBias );
		}

		w->sideImpulse *= w->sideFrictionStiffness;

		// Side friction is solved Jacobi-style — every grounded wheel computes
		// against the same pre-impulse velocity — so N wheels over-correct the
		// shared lateral/roll DOF by ~N×. Share the cancel across wheels, as the
		// longitudinal axis already does, so the summed impulse stays critical
		// for any wheel count without changing peak grip.
		w->sideImpulse /= (float)numWheelsOnGround;
	}

	float sideFactor = 1.0f;
	float fwdFactor = 0.5f;
	bool sliding = false;

	for ( int i = 0; i < numWheels; i++ )
	{
		Wheel* w = &v->wheels[i];
		float rollingFriction = 0.0f;
		bool hasGround = w->info.groundBody >= 0;

		if ( hasGround )
		{
			if ( w->engineForce != 0.0f )
			{
				rollingFriction = w->engineForce * dt;
			}
			else
			{
				// What the wheel resists with is what the brake holds. An
				// undriven, unbraked wheel does not resist along its travel
				// direction — raising this to the tyre's full static grip makes
				// a free wheel an immovable anchor, which pinned thruster-driven
				// craft in place. Creep is handled by the position anchor, whose
				// demand while parked is a tiny fraction of even a token brake.
				float maxImpulse = w->brake;

				float biasVelocity = 0.0f;
				if ( w->hasAnchor )
				{
					b3Vec3 slid = SubPos( w->info.contactPointWs, w->staticFrictionAnchor );
					biasVelocity = b3Dot( slid, v->forwardWs[i] ) * STATIC_FRICTION_ERP / dt;
				}

				float demand = CalcRollingFriction( v->chassis, w->info.groundBodyId, w->info.contactPointWs,
													v->forwardWs[i], numWheelsOnGround > 0 ? numWheelsOnGround : 1,
													biasVelocity );

				// Coulomb: the tyre sticks only while what it is asked to
				// deliver stays inside the friction circle. Past that it BREAKS
				// — it slides, and only the brake still resists. Clamping here
				// instead would make the wheel a permanent handbrake.
				if ( fabsf( demand ) <= maxImpulse )
				{
					rollingFriction = demand;
				}
				else
				{
					rollingFriction = ( demand < 0.0f ? -1.0f : 1.0f ) * w->brake;
					w->hasAnchor = false;
				}
			}
		}

		w->forwardImpulse = 0.0f;
		w->skidInfo = 1.0f;

		if ( hasGround )
		{
			float maxImp = w->wheelSuspensionForce * dt * w->frictionSlip;
			float maxImpSquared = maxImp * maxImp;

			w->forwardImpulse = rollingFriction;

			float x = w->forwardImpulse * fwdFactor;
			float y = w->sideImpulse * sideFactor;
			float impulseSquared = x * x + y * y;

			if ( impulseSquared > maxImpSquared )
			{
				sliding = true;
				float factor = maxImp * SafeInv( sqrtf( impulseSquared ) );
				w->skidInfo *= factor;
			}
		}
	}

	if ( sliding )
	{
		for ( int i = 0; i < numWheels; i++ )
		{
			Wheel* w = &v->wheels[i];
			if ( w->sideImpulse != 0.0f && w->skidInfo < 1.0f )
			{
				// The friction circle just said this tyre is past what it can
				// hold, so it is no longer stuck to the point it latched.
				// Without releasing here, a craft pushed sideways is pinned by a
				// hold that can never yield.
				w->hasAnchor = false;
				w->forwardImpulse *= w->skidInfo;
				w->sideImpulse *= w->skidInfo;
			}
		}
	}

	// Apply.
	b3Quat chassisRot = b3Body_GetRotation( v->chassis );
	b3Pos com = CenterOfMass( v->chassis );
	b3Vec3 chassisUp = b3RotateVector( chassisRot, AxisVector( v->indexUpAxis ) );

	for ( int i = 0; i < numWheels; i++ )
	{
		Wheel* w = &v->wheels[i];
		b3Pos contactPoint = w->info.contactPointWs;
		float toContactUp = b3Dot( chassisUp, SubPos( contactPoint, com ) );

		// Anti-squat and anti-roll both slide the application point UP toward
		// COM height, killing the pitch/roll moment of a force applied at the
		// bottom of the wheel.
		if ( w->forwardImpulse != 0.0f )
		{
			b3Pos forwardPoint = AddPosVec( contactPoint, Scale( chassisUp, -toContactUp * w->antiSquat ) );
			b3Body_ApplyLinearImpulse( v->chassis, Scale( v->forwardWs[i], w->forwardImpulse ), forwardPoint, false );
		}
		if ( w->sideImpulse != 0.0f )
		{
			b3Vec3 sideImpulse = Scale( v->axle[i], w->sideImpulse );
			b3Pos sidePoint =
				AddPosVec( contactPoint, Scale( chassisUp, -toContactUp * ( 1.0f - w->rollInfluence ) ) );
			b3Body_ApplyLinearImpulse( v->chassis, sideImpulse, sidePoint, false );
		}
	}
}

/**
 * Solve `A·λ = b` in place for small dense systems, returning false if singular.
 * Gaussian elimination with partial pivoting; n is at most the wheel count.
 */
static bool SolveDense( float* a, float* b, int n, float* out )
{
	for ( int col = 0; col < n; col++ )
	{
		int pivot = col;
		float best = fabsf( a[col * n + col] );
		for ( int row = col + 1; row < n; row++ )
		{
			float mag = fabsf( a[row * n + col] );
			if ( mag > best )
			{
				best = mag;
				pivot = row;
			}
		}
		if ( best < 1.0e-9f )
		{
			return false;
		}
		if ( pivot != col )
		{
			for ( int k = 0; k < n; k++ )
			{
				float tmp = a[col * n + k];
				a[col * n + k] = a[pivot * n + k];
				a[pivot * n + k] = tmp;
			}
			float tmp = b[col];
			b[col] = b[pivot];
			b[pivot] = tmp;
		}
		for ( int row = col + 1; row < n; row++ )
		{
			float f = a[row * n + col] / a[col * n + col];
			if ( f == 0.0f )
			{
				continue;
			}
			for ( int k = col; k < n; k++ )
			{
				a[row * n + k] -= f * a[col * n + k];
			}
			b[row] -= f * b[col];
		}
	}

	for ( int row = n - 1; row >= 0; row-- )
	{
		float sum = b[row];
		for ( int k = row + 1; k < n; k++ )
		{
			sum -= a[row * n + k] * out[k];
		}
		out[row] = sum / a[row * n + row];
	}
	return true;
}

static void UpdateWheelTransforms( Vehicle* v, int wheelId )
{
	Wheel* w = &v->wheels[wheelId];
	w->info.isInContact = false;

	b3WorldTransform t = b3Body_GetTransform( v->chassis );

	w->info.hardPointWs = AddPosVec( t.p, b3RotateVector( t.q, w->chassisConnectionPointCs ) );
	w->wheelDirectionWs = b3RotateVector( t.q, w->directionCs );
	w->wheelAxleWs = b3RotateVector( t.q, w->axleCs );

	// Steering rotates the axle about the suspension direction.
	b3Vec3 steerAxis = Scale( w->wheelDirectionWs, -1.0f );
	b3Quat steerRot = b3MakeQuatFromAxisAngle( TryNormalize( steerAxis, steerAxis ), w->steering );
	w->wheelAxleWs = b3RotateVector( steerRot, b3RotateVector( t.q, w->axleCs ) );

	w->center = AddPosVec( w->info.hardPointWs, Scale( w->wheelDirectionWs, w->info.suspensionLength ) );
}

void b3w_vehicle_update( Vehicle* v, float dt )
{
	// A sleeping chassis is not simulated, but applying an impulse still mutates
	// its stored velocity — a parked vehicle silently wound its suspension
	// impulses into a velocity that was never integrated away, then released it
	// as an upward kick the instant anything woke it.
	if ( !b3Body_IsAwake( v->chassis ) )
	{
		return;
	}

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		UpdateWheelTransforms( v, i );
	}

	b3Vec3 linvel = b3Body_GetLinearVelocity( v->chassis );
	v->currentVehicleSpeed = b3Length( linvel );

	b3Quat chassisRot = b3Body_GetRotation( v->chassis );
	b3Vec3 forwardW = b3RotateVector( chassisRot, AxisVector( v->indexForwardAxis ) );
	if ( b3Dot( forwardW, linvel ) < 0.0f )
	{
		v->currentVehicleSpeed *= -1.0f;
	}

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		ShapeCastWheel( v, i, dt );
	}

	float chassisMass = b3Body_GetMass( v->chassis );
	UpdateSuspension( v, chassisMass, dt );

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		if ( w->engineForce > 0.0f )
		{
			b3Body_SetAwake( v->chassis, true );
		}

		float suspensionForce = w->wheelSuspensionForce;
		if ( suspensionForce > w->maxSuspensionForce )
		{
			suspensionForce = w->maxSuspensionForce;
		}

		b3Vec3 impulse = Scale( w->info.contactNormalWs, suspensionForce * dt );
		b3Body_ApplyLinearImpulse( v->chassis, impulse, w->info.contactPointWs, false );
	}

	// Bump-stop velocity constraint — one simultaneous solve across every
	// bottomed wheel rather than iterative PGS, because they share a body.
	// Enforce v_at(p_i)·n_i >= 0 with lambda >= 0; drop any wheel whose lambda
	// goes negative (it would have to pull the chassis down) and resolve.
	{
		int idx[B3W_MAX_WHEELS];
		b3Pos contacts[B3W_MAX_WHEELS];
		b3Vec3 normals[B3W_MAX_WHEELS];
		float relVels[B3W_MAX_WHEELS];
		int n = 0;

		for ( int i = 0; i < v->wheelCount; i++ )
		{
			Wheel* w = &v->wheels[i];
			if ( !w->info.bottomedOut )
			{
				continue;
			}
			b3Vec3 vel = b3Body_GetWorldPointVelocity( v->chassis, w->info.contactPointWs );
			float relVel = b3Dot( w->info.contactNormalWs, vel );
			if ( relVel >= 0.0f )
			{
				continue;
			}
			idx[n] = i;
			contacts[n] = w->info.contactPointWs;
			normals[n] = w->info.contactNormalWs;
			relVels[n] = relVel;
			n++;
		}

		if ( n > 0 )
		{
			b3Pos com = CenterOfMass( v->chassis );
			float invMass = InvMass( v->chassis );
			b3Matrix3 invInertia = WorldInvInertia( v->chassis );

			while ( n > 0 )
			{
				float a[B3W_MAX_WHEELS * B3W_MAX_WHEELS];
				float b[B3W_MAX_WHEELS];
				float lambda[B3W_MAX_WHEELS];

				for ( int i = 0; i < n; i++ )
				{
					b3Vec3 ri = SubPos( contacts[i], com );
					b3Vec3 rxni = b3Cross( ri, normals[i] );
					b3Vec3 irxni = b3MulMV( invInertia, rxni );
					b[i] = -relVels[i];

					for ( int j = 0; j < n; j++ )
					{
						b3Vec3 rj = SubPos( contacts[j], com );
						b3Vec3 rxnj = b3Cross( rj, normals[j] );
						a[i * n + j] = invMass * b3Dot( normals[i], normals[j] ) + b3Dot( irxni, rxnj );
					}
				}

				if ( !SolveDense( a, b, n, lambda ) )
				{
					// Singular (duplicate contacts). Fall back to per-wheel
					// iteration so something sensible still happens.
					for ( int i = 0; i < n; i++ )
					{
						b3Vec3 ri = SubPos( contacts[i], com );
						b3Vec3 rxni = b3Cross( ri, normals[i] );
						float denom = invMass * b3Dot( normals[i], normals[i] ) +
									  b3Dot( b3MulMV( invInertia, rxni ), rxni );
						if ( denom < 1.0e-6f )
						{
							denom = 1.0e-6f;
						}
						float li = -relVels[i] / denom;
						if ( li > 0.0f )
						{
							b3Body_ApplyLinearImpulse( v->chassis, Scale( normals[i], li ), contacts[i], false );
						}
					}
					break;
				}

				int worst = -1;
				float worstVal = 0.0f;
				for ( int i = 0; i < n; i++ )
				{
					if ( lambda[i] < 0.0f && ( worst < 0 || lambda[i] < worstVal ) )
					{
						worst = i;
						worstVal = lambda[i];
					}
				}

				if ( worst >= 0 )
				{
					for ( int k = worst; k < n - 1; k++ )
					{
						idx[k] = idx[k + 1];
						contacts[k] = contacts[k + 1];
						normals[k] = normals[k + 1];
						relVels[k] = relVels[k + 1];
					}
					n--;
					continue;
				}

				for ( int i = 0; i < n; i++ )
				{
					b3Body_ApplyLinearImpulse( v->chassis, Scale( normals[i], lambda[i] ), contacts[i], false );
				}
				break;
			}
		}
	}

	// Bump-stop position correction — ONE correction on the chassis per step,
	// resolving the worst overshoot. Aggregating is what makes a full-strength
	// correction safe; per-wheel at that factor stacks and launches the chassis.
	{
		float deepest = 0.0f;
		b3Vec3 correctionNormal = { 0.0f, 0.0f, 0.0f };
		float cap = 0.0f;

		for ( int i = 0; i < v->wheelCount; i++ )
		{
			Wheel* w = &v->wheels[i];
			if ( !w->info.bottomedOut )
			{
				continue;
			}
			if ( w->maxSuspensionTravel > cap )
			{
				cap = w->maxSuspensionTravel;
			}
			if ( w->info.bottomOutOvershoot > deepest )
			{
				deepest = w->info.bottomOutOvershoot;
				correctionNormal = w->info.contactNormalWs;
			}
		}

		if ( deepest > 0.0f )
		{
			// Capped at the wheel's max suspension travel: a real bottom-out is
			// corrected over a few frames (visually identical, gentler on
			// velocity), and a spurious one — a cast clipping a seam between
			// adjacent static colliders — only nudges the chassis a few cm
			// before the next frame re-evaluates.
			float applied = deepest < cap ? deepest : cap;
			b3WorldTransform t = b3Body_GetTransform( v->chassis );
			b3Pos newPos = AddPosVec( t.p, Scale( correctionNormal, applied ) );
			b3Body_SetTransform( v->chassis, newPos, t.q );
		}
	}

	UpdateFriction( v, dt );

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		b3Vec3 vel = b3Body_GetWorldPointVelocity( v->chassis, w->info.hardPointWs );

		if ( w->info.isInContact )
		{
			b3Quat rot = b3Body_GetRotation( v->chassis );
			b3Vec3 fwd = b3RotateVector( rot, AxisVector( v->indexForwardAxis ) );
			float proj = b3Dot( fwd, w->info.contactNormalWs );
			fwd = Sub3( fwd, Scale( w->info.contactNormalWs, proj ) );

			float proj2 = b3Dot( fwd, vel );
			w->deltaRotation = ( proj2 * dt ) / w->radius;
			w->rotation += w->deltaRotation;
		}
		else
		{
			w->rotation += w->deltaRotation;
		}

		w->deltaRotation *= 0.99f; // damping of rotation when not in contact
	}
}

// ---------------------------------------------------------------------------
// Registry and the C API the wasm shim wraps
//
// Vehicles live here rather than in the shim so the shim stays a translation
// layer: it turns JavaScript's integers into ids and calls straight through.
// ---------------------------------------------------------------------------

#define B3W_MAX_VEHICLES 64

static Vehicle g_vehicles[B3W_MAX_VEHICLES];

Vehicle* b3w_vehicle_get( int handle )
{
	if ( handle < 0 || handle >= B3W_MAX_VEHICLES || !g_vehicles[handle].used )
	{
		return NULL;
	}
	return &g_vehicles[handle];
}

int b3w_vehicle_create( b3WorldId world, b3BodyId chassis )
{
	for ( int i = 0; i < B3W_MAX_VEHICLES; i++ )
	{
		if ( g_vehicles[i].used )
		{
			continue;
		}

		Vehicle* v = &g_vehicles[i];
		memset( v, 0, sizeof( Vehicle ) );
		v->used = true;
		v->world = world;
		v->chassis = chassis;
		v->indexUpAxis = 1;
		v->indexForwardAxis = 0;
		v->excludeLo = -1;
		v->excludeHi = -2;
		return i;
	}
	return -1;
}

void b3w_vehicle_destroy( int handle )
{
	Vehicle* v = b3w_vehicle_get( handle );
	if ( v != NULL )
	{
		v->used = false;
	}
}

void b3w_vehicle_set_exclusion( int handle, int lo, int hi )
{
	Vehicle* v = b3w_vehicle_get( handle );
	if ( v != NULL )
	{
		v->excludeLo = lo;
		v->excludeHi = hi;
	}
}

void b3w_vehicle_set_up_axis( int handle, int axis )
{
	Vehicle* v = b3w_vehicle_get( handle );
	if ( v != NULL )
	{
		v->indexUpAxis = axis;
	}
}

int b3w_vehicle_up_axis( int handle )
{
	Vehicle* v = b3w_vehicle_get( handle );
	return v != NULL ? v->indexUpAxis : 1;
}

int b3w_vehicle_wheel_count( int handle )
{
	Vehicle* v = b3w_vehicle_get( handle );
	return v != NULL ? v->wheelCount : 0;
}

/**
 * Add a wheel, with the same defaults Rapier's `WheelTuning` carries.
 *
 * The tuning values are overwritten immediately by the game's per-wheel setters,
 * but they are the documented defaults and a wheel that is added and never
 * configured should behave the same on both engines rather than collapse.
 */
int b3w_vehicle_add_wheel( int handle, float cx, float cy, float cz, float dx, float dy, float dz, float ax, float ay,
						   float az, float restLength, float radius, float width )
{
	Vehicle* v = b3w_vehicle_get( handle );
	if ( v == NULL || v->wheelCount >= B3W_MAX_WHEELS )
	{
		return -1;
	}

	int i = v->wheelCount++;
	Wheel* w = &v->wheels[i];
	memset( w, 0, sizeof( Wheel ) );

	w->chassisConnectionPointCs = B3_LITERAL( b3Vec3 ){ cx, cy, cz };
	w->directionCs = B3_LITERAL( b3Vec3 ){ dx, dy, dz };
	w->axleCs = B3_LITERAL( b3Vec3 ){ ax, ay, az };
	w->wheelDirectionWs = w->directionCs;
	w->wheelAxleWs = w->axleCs;
	w->suspensionRestLength = restLength;
	w->radius = radius;
	w->width = width;

	w->suspensionStiffness = 5.88f;
	w->dampingCompression = 0.83f;
	w->dampingRelaxation = 0.88f;
	w->maxSuspensionTravel = 5.0f;
	w->sideFrictionStiffness = 1.0f;
	w->frictionSlip = 10.5f;
	w->maxSuspensionForce = 6000.0f;

	w->rollInfluence = 0.1f;
	w->antiSquat = 0.5f;
	// 80° from upright, both filters. Accepts tilt well past a drivable slope,
	// rejects sideways, nose-standing and inverted poses. A tighter π/2 boundary
	// is float-sensitive — near-90° tilts slipped through.
	w->suspensionRejectAngle = 80.0f * 3.14159265358979f / 180.0f;
	w->surfaceRejectAngle = 80.0f * 3.14159265358979f / 180.0f;
	w->skidInfo = 1.0f;

	return i;
}

#define WHEEL_SETTER( name, field )                                                                                    \
	void name( int handle, int index, float value )                                                                    \
	{                                                                                                                  \
		Vehicle* v = b3w_vehicle_get( handle );                                                                        \
		if ( v != NULL && index >= 0 && index < v->wheelCount )                                                         \
		{                                                                                                              \
			v->wheels[index].field = value;                                                                            \
		}                                                                                                              \
	}

WHEEL_SETTER( b3w_vehicle_set_engine_force, engineForce )
WHEEL_SETTER( b3w_vehicle_set_brake, brake )
WHEEL_SETTER( b3w_vehicle_set_steering, steering )
WHEEL_SETTER( b3w_vehicle_set_side_friction, sideFrictionStiffness )
WHEEL_SETTER( b3w_vehicle_set_friction_slip, frictionSlip )
WHEEL_SETTER( b3w_vehicle_set_suspension_stiffness, suspensionStiffness )
WHEEL_SETTER( b3w_vehicle_set_suspension_compression, dampingCompression )
WHEEL_SETTER( b3w_vehicle_set_suspension_relaxation, dampingRelaxation )
WHEEL_SETTER( b3w_vehicle_set_suspension_rest_length, suspensionRestLength )
WHEEL_SETTER( b3w_vehicle_set_max_suspension_travel, maxSuspensionTravel )
WHEEL_SETTER( b3w_vehicle_set_max_suspension_force, maxSuspensionForce )
WHEEL_SETTER( b3w_vehicle_set_anti_squat, antiSquat )

/**
 * Every per-wheel readout in one crossing.
 *
 * The game reads a dozen values per wheel per frame for its own logic and its
 * debug overlay. One call filling a buffer beats a dozen boundary crossings,
 * and it is the same bulk-readback shape the body transforms already use.
 *
 * Layout, 20 floats per wheel:
 *   0  isInContact (0/1)      10 contactNormal x
 *   1  suspensionLength       11 contactNormal y
 *   2  suspensionForce        12 contactNormal z
 *   3  forwardImpulse         13 lastCastStart x
 *   4  sideImpulse            14 lastCastStart y
 *   5  lastToi                15 lastCastStart z
 *   6  steering               16 lastAxleWs x
 *   7  radius                 17 lastAxleWs y
 *   8  width                  18 lastAxleWs z
 *   9  groundBody (-1 none)   19 rotation
 */
void b3w_vehicle_read_wheels( int handle, float* out )
{
	Vehicle* v = b3w_vehicle_get( handle );
	if ( v == NULL )
	{
		return;
	}

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		float* o = out + i * 20;

		o[0] = w->info.isInContact ? 1.0f : 0.0f;
		o[1] = w->info.suspensionLength;
		o[2] = w->wheelSuspensionForce;
		o[3] = w->forwardImpulse;
		o[4] = w->sideImpulse;
		o[5] = w->info.lastToi;
		o[6] = w->steering;
		o[7] = w->radius;
		o[8] = w->width;
		o[9] = (float)w->info.groundBody;
		o[10] = w->info.contactNormalWs.x;
		o[11] = w->info.contactNormalWs.y;
		o[12] = w->info.contactNormalWs.z;
		o[13] = (float)w->info.lastCastStart.x;
		o[14] = (float)w->info.lastCastStart.y;
		o[15] = (float)w->info.lastCastStart.z;
		o[16] = w->info.lastAxleWs.x;
		o[17] = w->info.lastAxleWs.y;
		o[18] = w->info.lastAxleWs.z;
		o[19] = w->rotation;
	}
}

/** The wheel's suspension direction in chassis space, for drawing the cast. */
void b3w_vehicle_wheel_direction( int handle, int index, float* out )
{
	Vehicle* v = b3w_vehicle_get( handle );
	if ( v == NULL || index < 0 || index >= v->wheelCount )
	{
		return;
	}
	out[0] = v->wheels[index].directionCs.x;
	out[1] = v->wheels[index].directionCs.y;
	out[2] = v->wheels[index].directionCs.z;
}
