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
/** Bodies a vehicle may be told to ignore: its own, plus the player. */
/** Owner id reserved for bodies that are never ground for any vehicle. */
#define B3W_OWNER_NEVER_GROUND 0xFFFFFFFFu
/** Points per cap of the cast prism. See the header note on tread geometry. */
#define TREAD_SEGMENTS 8
/** Contact points a single wheel may rest on. Uncharted 4's number, and for
 *  their reason: one is a wheel that ping-pongs between surfaces. */
#define B3W_MAX_CONTACTS 3
/** Rows the bump-stop system will solve simultaneously. Bounds the dense
 *  matrix on the stack; beyond it the extra bottomed contacts simply wait for
 *  the next substep. */
#define B3W_MAX_BUMP_ROWS 32

/**
 * Two hits closer together than this describe the same patch of surface.
 *
 * Without it, a flat floor gives a wheel three contacts a couple of millimetres
 * apart and a genuinely different surface — the second plank, the kerb edge —
 * never gets a slot.
 */
static const float CONTACT_SPREAD = 0.04f;

/** Below this chassis speed a wheel with no drive or brake grips longitudinally. */
static const float STATIC_FRICTION_MAX_SPEED = 0.5f;
/** Fraction of anchor drift undone per step — the usual Baumgarte trade. */
static const float STATIC_FRICTION_ERP = 0.2f;

/** Fraction of a bump-stop overshoot removed per second. Firm enough to look
 *  rigid, soft enough not to launch a chassis that is briefly deep. */
static const float BUMP_STOP_ERP = 0.6f;
/** Furthest a stuck contact may drift before the tyre is treated as having slipped. */
static const float MAX_STATIC_FRICTION_SLIP = 0.05f;

/**
 * One place a wheel touches the world.
 *
 * A wheel gets up to `B3W_MAX_CONTACTS` of these. The nearest-only version is
 * Uncharted 4's first attempt, and it fails the same way theirs did: a tyre
 * spanning two planks, or crossing a seam, has its single contact snap between
 * surfaces and the wheel ping-pongs. Holding the three nearest instead lets the
 * suspension rest on both at once.
 */
typedef struct
{
	b3Vec3 normalWs;
	b3Pos pointWs;
	float suspensionLength;
	float clippedInvContactDotSuspension;
	bool bottomedOut;
	float bottomOutOvershoot;
	/**
	 * Where the wheel centre sits relative to the contact plane, along the
	 * contact normal, at the length the cast measured.
	 *
	 * This is the ONE number that lets the spring re-measure its own compression
	 * every substep without casting again. The cast is expensive and its
	 * geometry — contact point, normal, the inverse dot — is good for the whole
	 * tick because the chassis moves millimetres. The chassis POSE is not: it
	 * rises and falls under the spring 32 times per tick, and a spring reading a
	 * tick-old compression is the position twin of the stale-velocity damper. It
	 * overshoots, corrects late, and settles into a limit cycle instead of
	 * settling. Holding the plane offset turns the cast's answer into a plane
	 * the live hard point can be measured against.
	 */
	float planeOffset;
	/** Share of the wheel's suspension force this contact carries. */
	float force;
} WheelContact;

typedef struct
{
	b3Vec3 contactNormalWs;
	b3Vec3 groundNormalWs;
	b3Pos contactPointWs;
	float suspensionLength;
	b3Pos hardPointWs;
	bool isInContact;
	/** The kept contacts, nearest first. Index 0 mirrors the singular fields
	 *  above, which friction, reporting and the wheel transform still read. */
	WheelContact contacts[B3W_MAX_CONTACTS];
	int contactCount;
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
	/**
	 * Friction impulse already spent this TICK, across its substeps.
	 *
	 * The friction circle is a budget, and a budget is per tick — the tyre can
	 * deliver so much before it breaks traction, not so much per substep. Without
	 * carrying it, the per-substep budget (suspensionForce * h * slip) was
	 * compared against a lateral impulse that cancels velocity and therefore has
	 * no time scale at all: the circle broke every substep, skidInfo pinned to
	 * ~1/32, and a car at full throttle travelled 0.9mm. This is warm starting,
	 * and it is what makes solving 32 times equivalent to solving once.
	 */
	float accumForward;
	float accumSide;

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
	/**
	 * Whether this vehicle's cast state is fresh enough to solve against.
	 *
	 * `prepare` does the shape casts once per tick and `solve` runs per substep
	 * off what they found. A world step that had no prepare therefore solves
	 * against LAST tick's contacts — or, for a vehicle that has never been
	 * prepared, against a zeroed contact point at the world origin, which sends
	 * the chassis to NaN in one step. `createGround` ends with exactly such a
	 * step (a gravity-free settle), so building a machine and then loading a
	 * world destroyed it.
	 *
	 * Set by prepare, cleared by the world step that consumes it: cast state is
	 * good for one step and no more.
	 */
	bool armed;
	b3WorldId world;
	/** The shim's world handle, for the body-owner table. */
	int worldHandle;
	b3BodyId chassis;
	int wheelCount;
	Wheel wheels[B3W_MAX_WHEELS];
	b3Vec3 forwardWs[B3W_MAX_WHEELS];
	b3Vec3 axle[B3W_MAX_WHEELS];
	float currentVehicleSpeed;
	/** The whole tick, recorded at prepare. Error-correction rates use this;
	 *  impulses use the substep. See UpdateFriction. */
	float stepDt;
	int indexUpAxis;
	int indexForwardAxis;

	/**
	 * Body indices the wheels may not stand on — the vehicle's own bodies and
	 * the player.
	 *
	 * An explicit LIST, not a range. A range looks like a cheap way to cross the
	 * boundary until you notice what it actually excludes: the player is body 0
	 * and the terrain is body 1, so a vehicle whose own bodies are 40-42 asked to
	 * skip [0, 42] and skipped the ground with them. Every wheel cast then found
	 * nothing, the suspension never engaged, and the wheels went through the
	 * floor. The set is a handful of entries; scanning it is nothing next to the
	 * cast it guards.
	 */
	/**
	 * Who this vehicle belongs to. A wheel may not stand on any body carrying
	 * the same owner, nor on a body the host has marked non-drivable.
	 *
	 * This replaces a copied list of body indices with a capped size. The list
	 * was uploaded every tick and silently truncated past 64 bodies, so a large
	 * enough structure quietly started treating its OWN bodies as ground —
	 * exactly the failure a cap without a report produces. Ownership is a fact
	 * about a body, so it lives on the body (`b3Body_SetUserData`) and is
	 * compared in O(1) here.
	 */
	uint32_t owner;
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
	float fraction;
	b3Vec3 normal;
	b3Pos point;
	int body;
	b3BodyId bodyId;
} WheelHit;

typedef struct
{
	/** The vehicle's own chassis, which can never be its own ground. */
	b3BodyId chassis;
	uint32_t owner;
	int worldHandle;
	/** The surface filter, applied here rather than after the cast: a hit on a
	 *  wall must not occupy a slot a drivable surface behind it could use. */
	b3Vec3 worldUp;
	float minSurfaceUpFacing;
	int count;
	WheelHit hits[B3W_MAX_CONTACTS];
} WheelCast;

static int BodyIndexOfShapeV( b3ShapeId shapeId )
{
	intptr_t tagged = (intptr_t)b3Body_GetUserData( b3Shape_GetBody( shapeId ) );
	return tagged > 0 ? (int)( tagged - 1 ) : -1;
}

/**
 * Merge a hit into the nearest-first set, spread rule and all.
 *
 * A candidate sitting on top of one already kept is the same surface seen
 * twice — it replaces that entry only if it is nearer, and never takes a second
 * slot. Anything else is inserted by fraction and the furthest falls off the
 * end.
 */
static void KeepWheelHit( WheelCast* ctx, WheelHit hit )
{
	for ( int i = 0; i < ctx->count; i++ )
	{
		if ( b3Length( SubPos( hit.point, ctx->hits[i].point ) ) >= CONTACT_SPREAD )
		{
			continue;
		}
		if ( hit.fraction < ctx->hits[i].fraction )
		{
			// Nearer duplicate: take the slot, then bubble it into order.
			ctx->hits[i] = hit;
			for ( int k = i; k > 0 && ctx->hits[k].fraction < ctx->hits[k - 1].fraction; k-- )
			{
				WheelHit tmp = ctx->hits[k];
				ctx->hits[k] = ctx->hits[k - 1];
				ctx->hits[k - 1] = tmp;
			}
		}
		return;
	}

	int slot = ctx->count < B3W_MAX_CONTACTS ? ctx->count : B3W_MAX_CONTACTS - 1;
	if ( ctx->count == B3W_MAX_CONTACTS && hit.fraction >= ctx->hits[slot].fraction )
	{
		return;
	}

	int at = slot;
	while ( at > 0 && hit.fraction < ctx->hits[at - 1].fraction )
	{
		ctx->hits[at] = ctx->hits[at - 1];
		at--;
	}
	ctx->hits[at] = hit;
	if ( ctx->count < B3W_MAX_CONTACTS )
	{
		ctx->count++;
	}
}

/** Defined in shim.c — the body-owner table lives with the body table. */
uint32_t b3w_owner_of_body( int handle, int body );

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
	//
	// Owner 0 means "unowned" — terrain and loose world geometry — and is always
	// drivable. B3W_OWNER_NEVER_GROUND is the host's way of saying a body is
	// never ground for anyone, which is what the player is.
	if ( body >= 0 )
	{
		uint32_t owner = b3w_owner_of_body( ctx->worldHandle, body );
		if ( owner != 0 && ( owner == ctx->owner || owner == B3W_OWNER_NEVER_GROUND ) )
		{
			return -1.0f;
		}
	}

	// Drivability is a world-space property of the surface, so it compares
	// against world up rather than the suspension direction — otherwise chassis
	// tilt would move the accept boundary around.
	if ( b3Dot( normal, ctx->worldUp ) <= ctx->minSurfaceUpFacing )
	{
		return -1.0f;
	}

	WheelHit hit = { .fraction = fraction, .normal = normal, .point = point, .body = body, .bodyId = hitBody };
	KeepWheelHit( ctx, hit );

	// Keep searching the whole sweep until three slots are full; after that
	// nothing beyond the furthest kept hit can improve the set.
	return ctx->count < B3W_MAX_CONTACTS ? 1.0f : ctx->hits[ctx->count - 1].fraction;
}

/**
 * Turn a raw travel distance into the contact's stored suspension state.
 *
 * The cast measures this once per tick and the substep solve re-measures it
 * from the live chassis pose; both must clamp identically or the end stop and
 * the bump stop would mean different things depending on who asked.
 *
 * The wheel cannot travel past the hard point — the spring has a mechanical end
 * stop, and a negative raw length means the chassis is overshooting into the
 * ground and must be absorbed rigidly.
 */
static void StoreSuspensionLength( WheelContact* contact, float rawLength, float maxSuspensionLength )
{
	contact->suspensionLength =
		rawLength < 0.0f ? 0.0f : ( rawLength > maxSuspensionLength ? maxSuspensionLength : rawLength );
	contact->bottomedOut = rawLength < 0.0f;
	contact->bottomOutOvershoot = rawLength < 0.0f ? -rawLength : 0.0f;
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
					   .owner = v->owner,
					   .worldHandle = v->worldHandle,
					   .worldUp = worldUp,
					   .minSurfaceUpFacing = cosf( w->surfaceRejectAngle ),
					   .count = 0 };

	float maxToi = raylen + offset;
	b3World_CastShape( v->world, castStart, &proxy, Scale( direction, maxToi ), b3DefaultQueryFilter(), OnWheelHit,
					   &cast );

	// Filter: chassis pose. Drop every hit when the suspension does not point
	// broadly downward — sideways, nose-standing or inverted, where a
	// heightfield's always-outward normal would mislead a normal-only test. The
	// surface filter is applied per hit, inside the cast callback.
	float suspensionDownward = -b3Dot( direction, worldUp );
	bool accepted = cast.count > 0 && suspensionDownward > cosf( w->suspensionRejectAngle );

	w->info.groundBody = -1;
	w->info.groundBodyId = b3_nullBodyId;
	w->info.contactCount = 0;

	if ( accepted )
	{
		float maxSuspensionLength = w->suspensionRestLength + w->maxSuspensionTravel;
		b3Vec3 axial = Scale( w->wheelDirectionWs, -1.0f );

		for ( int c = 0; c < cast.count; c++ )
		{
			const WheelHit* hit = &cast.hits[c];
			WheelContact* contact = &w->info.contacts[c];

			// Suspension and friction both use the real ground normal so the
			// spring stays aligned with the contact plane on slopes; fall back to
			// the suspension axis when the hit is more than ~60° off it, which is
			// what stops the spring shoving the chassis sideways off a vertical
			// face.
			contact->normalWs = b3Dot( hit->normal, axial ) > 0.5f ? hit->normal : axial;

			float rawLength = hit->fraction * maxToi - offset;
			StoreSuspensionLength( contact, rawLength, maxSuspensionLength );
			contact->force = 0.0f;

			// The wheel centre at the length the cast found, measured against the
			// contact plane. Kept UNCLAMPED so a bottomed wheel re-derives the same
			// negative raw length the cast saw, and so the end stop keeps working.
			b3Pos centerAtCast = AddPosVec( source, Scale( w->wheelDirectionWs, rawLength ) );

			// Slide the witness point across the contact plane until it sits
			// under the wheel's centre.
			//
			// A prism resting flat on flat ground is a FACE-face contact, so the
			// witness GJK returns is any point of that face — and it snaps
			// between them on sub-micron input changes. Measured on a parked car
			// it teleported the full wheel width, 0.242 m, every single tick.
			// That point is where the suspension force is applied and where the
			// damper reads its velocity, so an alternating lever arm turned the
			// damper into a pump: the chassis rocked in a 4 Hz roll limit cycle
			// forever and the reported suspension length swung 9.8 mm on a
			// chassis whose centre held to 0.17 mm.
			//
			// Under the wheel centre is where a symmetric tyre's patch actually
			// is, and it is the only choice that leaves no ambiguity to snap
			// between. Moving the point WITHIN its plane is exactly neutral for
			// everything measured along the normal, so the suspension length,
			// the end stop and the bump stop are untouched by construction —
			// only the lever arm changes, from arbitrary to stable.
			b3Vec3 toCenter = SubPos( centerAtCast, hit->point );
			b3Vec3 alongPlane = Sub3( toCenter, Scale( contact->normalWs, b3Dot( contact->normalWs, toCenter ) ) );
			contact->pointWs = AddPosVec( hit->point, alongPlane );

			contact->planeOffset = b3Dot( contact->normalWs, SubPos( centerAtCast, contact->pointWs ) );

			float denominator = b3Dot( contact->normalWs, w->wheelDirectionWs );
			contact->clippedInvContactDotSuspension = denominator >= -0.1f ? 1.0f / 0.1f : -1.0f / denominator;
		}
		w->info.contactCount = cast.count;

		// The nearest contact is the wheel's: it is the surface the tyre rests
		// on, and what friction, the wheel transform and the readback all mean by
		// "the" contact.
		const WheelContact* primary = &w->info.contacts[0];
		w->info.contactNormalWs = primary->normalWs;
		w->info.groundNormalWs = primary->normalWs;
		w->info.contactPointWs = primary->pointWs;
		w->info.suspensionLength = primary->suspensionLength;
		w->info.bottomedOut = primary->bottomedOut;
		w->info.bottomOutOvershoot = primary->bottomOutOvershoot;
		w->clippedInvContactDotSuspension = primary->clippedInvContactDotSuspension;
		w->info.isInContact = true;
		w->info.groundBody = cast.hits[0].body;
		w->info.groundBodyId = cast.hits[0].bodyId;

		w->info.lastToi = cast.hits[0].fraction * maxToi;
		w->info.lastCastOffset = offset;
		w->info.lastCastStart = castStart;
		w->info.lastAxleWs = axle;

		b3Vec3 velAtContact = b3Body_GetWorldPointVelocity( v->chassis, w->info.contactPointWs );
		float projVel = b3Dot( w->info.contactNormalWs, velAtContact );
		float denominator = b3Dot( w->info.contactNormalWs, w->wheelDirectionWs );
		w->suspensionRelativeVelocity = denominator >= -0.1f ? 0.0f : projVel * w->clippedInvContactDotSuspension;
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

		// One spring row PER CONTACT, each carrying an equal share of the wheel's
		// load. The share is what keeps this honest: a wheel is one spring no
		// matter how many surfaces it happens to touch, so the TOTAL it delivers
		// must not grow with the contact count. Letting it would make a tyre
		// crossing a plank seam kick — the same Jacobi over-correction the side
		// friction already divides by `numWheelsOnGround`, one level down.
		int n = w->info.contactCount > 0 ? w->info.contactCount : 1;
		float share = 1.0f / (float)n;
		float total = 0.0f;

		// Re-measure compression against the LIVE chassis, the same way and for
		// the same reason the damper re-reads velocity below. No new cast: the
		// contact plane the tick found still holds, only the hard point has
		// moved — which is exactly the spring's input.
		b3WorldTransform t = b3Body_GetTransform( v->chassis );
		b3Pos hardPointNow = AddPosVec( t.p, b3RotateVector( t.q, w->chassisConnectionPointCs ) );
		float maxSuspensionLength = w->suspensionRestLength + w->maxSuspensionTravel;

		for ( int c = 0; c < w->info.contactCount; c++ )
		{
			WheelContact* contact = &w->info.contacts[c];

			// Solve dot(n, hardPoint + dir*length - contactPoint) == planeOffset
			// for length. The stored inverse dot is already negated — dir points
			// into the ground and the normal out of it — so dividing by
			// dot(n, dir) is a multiply by it here.
			float gap = b3Dot( contact->normalWs, SubPos( hardPointNow, contact->pointWs ) ) - contact->planeOffset;
			StoreSuspensionLength( contact, gap * contact->clippedInvContactDotSuspension, maxSuspensionLength );

			float lengthDiff = w->suspensionRestLength - contact->suspensionLength;
			float force = w->suspensionStiffness * lengthDiff * contact->clippedInvContactDotSuspension;

			// Re-read the closing speed from the LIVE chassis every substep.
			//
			// The cast measures it once per tick, and a damper fed a stale
			// velocity is an energy pump: the chassis rises, the damper still
			// believes it is compressing, and keeps pushing up. Applied once per
			// tick that was a wobble; applied 32 times it launched a car to 575
			// metres. The geometry (contact point, normal, the clipped inverse
			// dot) is fine to reuse — the chassis moves millimetres in a tick —
			// but the VELOCITY is exactly what changes while the substeps run,
			// and it is the only input the damper has.
			b3Vec3 velNow = b3Body_GetWorldPointVelocity( v->chassis, contact->pointWs );
			float projectedRelVel = b3Dot( contact->normalWs, velNow ) * contact->clippedInvContactDotSuspension;
			if ( c == 0 )
			{
				w->suspensionRelativeVelocity = projectedRelVel;
			}

			float suspDamping = projectedRelVel < 0.0f ? w->dampingCompression : w->dampingRelaxation;
			float implicit = 1.0f / ( 1.0f + dt * suspDamping * (float)contacts );
			force -= suspDamping * projectedRelVel * implicit;

			float result = force * chassisMass * share;
			contact->force = result > 0.0f ? result : 0.0f;
			total += contact->force;
		}

		w->wheelSuspensionForce = total;

		// Index 0 mirrors the singular fields, which reporting and the wheel
		// transform read; they should show the length the spring just used, not
		// the one the cast measured a substep-count ago.
		if ( w->info.contactCount > 0 )
		{
			const WheelContact* primary = &w->info.contacts[0];
			w->info.suspensionLength = primary->suspensionLength;
			w->info.bottomedOut = primary->bottomedOut;
			w->info.bottomOutOvershoot = primary->bottomOutOvershoot;
		}
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

/**
 * The impulse that cancels a point's velocity along `normal`.
 *
 * The denominator is how much that point moves along `normal` per unit impulse:
 * `1/m + n . ((I^-1 (r x n)) x r)`. It used to read `1/m + |I^-1 (r x n)|^2`,
 * which is not that quantity — it is the squared magnitude of an angular
 * velocity, and it sized every lateral tyre impulse wrong. Where the wheels sit
 * far from the centre of mass the error is large and the cancel OVERSHOOTS, so
 * each wheel reverses the velocity it was asked to remove; on four wheels
 * sharing one body the surplus lands in the least damped axis. The built-in Car
 * template, spawned and left alone, span at 0.6 rad/s for ever with all four
 * wheels gripping, and its wheels turned with it because a rolling wheel is what
 * the renderer spins. `ImpulseDenominator` above already had the right form.
 */
static float ResolveSingleUnilateral( b3BodyId body, b3Pos pt, b3Vec3 normal, float biasVelocity )
{
	b3Vec3 vel = b3Body_GetWorldPointVelocity( body, pt );
	b3Vec3 dpt = SubPos( pt, CenterOfMass( body ) );
	b3Vec3 aj = b3Cross( dpt, normal );
	b3Vec3 iaj = b3MulMV( WorldInvInertia( body ), aj );

	float jacDiagAb = InvMass( body ) + b3Dot( normal, b3Cross( iaj, dpt ) );
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

	// Same denominator as the unilateral case above, for two bodies. The second
	// body's constraint normal is the negated one — `bj` is built from it — so its
	// term is measured along -normal, or the two would subtract.
	float jacDiagAb = InvMass( body1 ) + InvMass( body2 ) + b3Dot( normal, b3Cross( iaj, dpt1 ) ) -
					  b3Dot( normal, b3Cross( ibj, dpt2 ) );
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
/**
 * `dt` is the SUBSTEP here; `v->stepDt` is the whole tick.
 *
 * Impulses scale with the substep — a force applied 32 times must each time be
 * worth a 32nd. Position-error correction does NOT: it is a RATE, "remove this
 * fraction of the accumulated slide per second", so dividing it by the substep
 * makes it 32x stronger and then applies it 32 times. That is 1024x, and it
 * pinned the vehicle to the ground so hard that full throttle moved it 0.9mm.
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
			lateralBias = b3Dot( slid, v->axle[i] ) * STATIC_FRICTION_ERP / v->stepDt;
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

		/*
		 * Share the lateral cancel across wheels ONLY at rest.
		 *
		 * Side friction is solved Jacobi-style — every grounded wheel computes
		 * against the same pre-impulse velocity — so N wheels over-correct the
		 * shared lateral DOF by ~N. Dividing always is the safe answer and it is
		 * what shipped, but it also makes the tyre STIFFNESS-limited: the summed
		 * impulse is small enough that the friction circle rarely binds, and a
		 * long machine feels like it is on ice. Nick reported exactly that, on
		 * both the car and the tank.
		 *
		 * Left undivided while moving, the impulses saturate against the circle
		 * instead, so the tyre is GRIP-limited — it holds everything mu*N allows
		 * and slides past it. That is the behaviour he tested and kept, and it
		 * leaves a tank's differential steering intact: more engine speed turns
		 * faster, and lifting the throttle pivots it.
		 *
		 * At rest there is no circle to clamp against, and the over-correction
		 * shows up as creep — a parked tank walked 7.3 cm against a 5 cm limit.
		 * So the share is applied below the same speed the static-friction
		 * anchor already uses to decide a wheel is standing still, rather than
		 * introducing a second idea of "stopped".
		 */
		/*
		 * Side friction is solved Jacobi-style — every grounded wheel computes
		 * against the same pre-impulse velocity — so N wheels over-correct the
		 * shared lateral DOF by ~N. Sharing the cancel is what keeps the summed
		 * impulse right for any wheel count.
		 *
		 * Tried and REVERTED: leaving them undivided while moving, on the theory
		 * that saturating the friction circle would feel grippier. It does, and
		 * it also inflates the LATERAL term of that same circle by ~N — so a
		 * sustained turn spends the entire grip budget sideways and starves the
		 * drive. Holding W and D, a car settled to 1.1 m/s and stayed there;
		 * with the share restored it circles at 16.1. Nick reported it as tight
		 * loops slowly coming to a halt.
		 *
		 * Grip belongs in the tuning constants, not in an over-correction:
		 * VEHICLE_SIDE_FRICTION_STIFFNESS_CAR / _TANK are the knobs. This also
		 * matches what the coupled dense solve produced when it was tried — the
		 * share is a good approximation of solving the wheels together, not a
		 * fudge around it.
		 */
		w->sideImpulse /= (float)numWheelsOnGround;
	}

	float sideFactor = 1.0f;
	/*
	 * 0.5, as Bullet and Rapier both have it, and REVERTED here after trying 1.0.
	 *
	 * 1.0 is the physically honest number — the circle bounds the tyre's total
	 * force at mu*N whichever way it points. But the game's drive force is
	 * larger than its own grip (4000 N against mu*N of about 3015 on the Car),
	 * so an honest circle leaves the tyre permanently SATURATED, and a saturated
	 * tyre scales `sideImpulse` down by skidInfo along with the forward one.
	 * Accelerating then eats the lateral grip: Nick reported the car sliding
	 * "like I'm on ice" and unable to hold a straight line, within minutes.
	 *
	 * The honest fix is not this constant — it is that full throttle should not
	 * ask for more than the tyres can deliver, which is a tuning decision about
	 * VEHICLE_ENGINE_FORCE and not mine to take. Until then 0.5 keeps the circle
	 * off the everyday case, and the top-speed taper is what stops the rollover.
	 */
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
				// A brake resists ROTATION; the ground resists SLIDING. They are
				// different limits and conflating them is why a parked car ran
				// away downhill: an idle vehicle carries a token brake
				// (VEHICLE_BRAKE_FORCE, 0.5), and capping the hold at that meant
				// a stationary tyre could resist half a newton-second while
				// gravity pulled with a hundred. Measured on a 20 degree slope,
				// a parked car slid 342 m in 20 s; on 10 degrees, half a metre.
				//
				// So a STATIONARY tyre — anchored, undriven, on static ground —
				// is held by its friction circle, the same limit the combined
				// impulse is tested against below. A ROLLING one still decays
				// only as fast as the brake allows, which is what keeps coasting
				// feeling like coasting.
				//
				// Still gated on a brake being asked for at all: the game zeroes
				// the brake for thruster-driven craft (`hasActiveThruster`), and
				// that zero is what keeps a free wheel from becoming an immovable
				// anchor and pinning them in place.
				float maxImpulse = w->brake;
				if ( w->hasAnchor && w->brake > 0.0f )
				{
					float grip = w->wheelSuspensionForce * v->stepDt * w->frictionSlip;
					maxImpulse = maxImpulse > grip ? maxImpulse : grip;
				}

				float biasVelocity = 0.0f;
				if ( w->hasAnchor )
				{
					b3Vec3 slid = SubPos( w->info.contactPointWs, w->staticFrictionAnchor );
					biasVelocity = b3Dot( slid, v->forwardWs[i] ) * STATIC_FRICTION_ERP / v->stepDt;
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
			// The budget is the whole tick's, because what it is being compared
			// against is: the lateral solve cancels velocity outright and the
			// longitudinal impulses accumulate over the substeps to engineForce
			// times the tick. Both sides of this test are now tick-scale.
			float maxImp = w->wheelSuspensionForce * v->stepDt * w->frictionSlip;
			float maxImpSquared = maxImp * maxImp;

			w->forwardImpulse = rollingFriction;

			float x = ( w->accumForward + w->forwardImpulse ) * fwdFactor;
			// Lateral IS accumulated, and testing the alternative is how I know:
			// charging only the current substep's lateral impulse pinned the car
			// again (0.5mm at full throttle). Both terms describe grip the tick
			// has committed, so both belong in the same running total.
			float y = ( w->accumSide + w->sideImpulse ) * sideFactor;
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
			// Also diverges: upstream additionally requires `sideImpulse != 0`,
			// which skips the clamp for a wheel driving dead straight — the one
			// case where the whole friction budget is being spent forwards.
			if ( w->skidInfo < 1.0f )
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

		// Spent, so the next substep sees what this tick has already used.
		w->accumForward += w->forwardImpulse;
		w->accumSide += w->sideImpulse;
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

/**
 * Everything that must happen ONCE per full step: where the wheels are, how
 * fast the chassis is going, and the shape casts that find the ground.
 *
 * Casting is the expensive half and its answer is good for the whole step —
 * the chassis moves millimetres in one tick. Splitting it from the solve is
 * what makes solving 32 times affordable.
 */
void b3w_vehicle_prepare( Vehicle* v, float dt )
{
	v->stepDt = dt > 0.0f ? dt : 1.0f / 60.0f;
	v->armed = true;

	// A fresh friction budget for the tick the substeps are about to share.
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		v->wheels[i].accumForward = 0.0f;
		v->wheels[i].accumSide = 0.0f;
	}

	// A sleeping DYNAMIC chassis is not simulated, but applying an impulse still
	// mutates its stored velocity — a parked vehicle silently wound its
	// suspension impulses into a velocity that was never integrated away, then
	// released it as an upward kick the instant anything woke it.
	//
	// The body type test is not decoration. Box3D reports a STATIC body as never
	// awake, so guarding on awakeness alone skipped the whole controller for
	// every static vehicle — their wheels kept the suspension length they were
	// initialised with, zero, which draws a wheel retracted into its mount and
	// makes the car look sunk into the ground with its wheels buried. Rapier's
	// `is_sleeping` only ever describes a dynamic body, which is why the same
	// guard behaved correctly there.
	//
	// A COMMAND TO DRIVE WAKES THE VEHICLE, and it has to happen here, above the
	// guard. It used to sit inside `solve`, below solve's identical sleep
	// early-out, so it could only ever run on a chassis that was already awake —
	// dead code for the one case it was written for. Measured: a real car driven
	// fresh travels 31.76 m in 5 s, and the same car after 30 s parked travels
	// 0.00 m. Park, wait, press the throttle, nothing happens.
	//
	// Drive only, never the brake: an idle vehicle carries a standing brake, so
	// waking on that would hold every parked machine in the world awake forever
	// and sleep is the tick budget's biggest lever.
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		if ( v->wheels[i].engineForce != 0.0f )
		{
			b3Body_SetAwake( v->chassis, true );
			break;
		}
	}

	if ( b3Body_GetType( v->chassis ) == b3_dynamicBody && !b3Body_IsAwake( v->chassis ) )
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

	// Wheel spin, once per TICK. It is odometry — how far the tyre rolled — not a
	// force, so it advances with the tick, not with each substep.
	//
	// Per substep the AIRBORNE branch was catastrophic: it re-adds the last
	// grounded `deltaRotation` every time, so summed over 32 substeps with the
	// 0.99 decay that is ~27x one tick's rotation. A wheel off the ground span
	// far too fast and coasted almost forever — "the wheels keep rotating". The
	// 0.99 decay is per tick for the same reason: applied 32 times it is 0.72,
	// which stops a free wheel ~30x too quickly.

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

/**
 * The force model, once per SUBSTEP, against `h` rather than the whole tick.
 *
 * Applied once per tick this was the least accurate thing in the world and it
 * was attached to the thing players look at hardest: the suspension integrated
 * explicitly at the tick rate while every constraint around it integrated 32
 * times finer. The damper suffered worst — its implicit attenuation is
 * 1/(1 + h*c*contacts), so a tick-sized h delivered a fraction of the damping
 * the springs were tuned for, and the chassis wobbled until the sleep
 * threshold caught it.
 */
void b3w_vehicle_solve( Vehicle* v, float h )
{
	if ( b3Body_GetType( v->chassis ) == b3_dynamicBody && !b3Body_IsAwake( v->chassis ) )
	{
		return;
	}

	float chassisMass = b3Body_GetMass( v->chassis );
	UpdateSuspension( v, chassisMass, h );

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];

		// The cap is the WHEEL's, so an over-force wheel scales all of its
		// contacts down together rather than capping each one and delivering
		// N times the limit.
		float scale = 1.0f;
		if ( w->wheelSuspensionForce > w->maxSuspensionForce )
		{
			scale = w->maxSuspensionForce / w->wheelSuspensionForce;
		}

		for ( int c = 0; c < w->info.contactCount; c++ )
		{
			const WheelContact* contact = &w->info.contacts[c];
			b3Vec3 impulse = Scale( contact->normalWs, contact->force * scale * h );
			b3Body_ApplyLinearImpulse( v->chassis, impulse, contact->pointWs, false );
		}
	}

	// Bump-stop velocity constraint — one simultaneous solve across every
	// bottomed wheel rather than iterative PGS, because they share a body.
	// Enforce v_at(p_i)·n_i >= 0 with lambda >= 0; drop any wheel whose lambda
	// goes negative (it would have to pull the chassis down) and resolve.
	{
		int idx[B3W_MAX_BUMP_ROWS];
		b3Pos contacts[B3W_MAX_BUMP_ROWS];
		b3Vec3 normals[B3W_MAX_BUMP_ROWS];
		float relVels[B3W_MAX_BUMP_ROWS];
		int n = 0;

		for ( int i = 0; i < v->wheelCount && n < B3W_MAX_BUMP_ROWS; i++ )
		{
			Wheel* w = &v->wheels[i];

			for ( int c = 0; c < w->info.contactCount && n < B3W_MAX_BUMP_ROWS; c++ )
			{
				const WheelContact* contact = &w->info.contacts[c];
				if ( !contact->bottomedOut )
				{
					continue;
				}

				b3Vec3 vel = b3Body_GetWorldPointVelocity( v->chassis, contact->pointWs );
				float relVel = b3Dot( contact->normalWs, vel );

				// How fast the chassis must rise to undo the overshoot, as a
				// RATE: a fraction of the error per second, so it is independent
				// of both the tick and the substep. This replaces a position
				// teleport — see below — and it is the whole reason the bump stop
				// stops chattering.
				float bias = 0.0f;
				if ( contact->bottomOutOvershoot > 0.0f )
				{
					float capped = contact->bottomOutOvershoot < w->maxSuspensionTravel ? contact->bottomOutOvershoot
																						: w->maxSuspensionTravel;
					bias = capped * BUMP_STOP_ERP / v->stepDt;
				}

				if ( relVel >= bias )
				{
					continue;
				}
				idx[n] = i;
				contacts[n] = contact->pointWs;
				normals[n] = contact->normalWs;
				// Solved toward the bias rather than toward zero, so the
				// constraint both stops the chassis sinking further AND lifts it
				// out.
				relVels[n] = relVel - bias;
				n++;
			}
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

	// The bump-stop position correction that used to live here is gone. It moved
	// the chassis with b3Body_SetTransform — a teleport on a dynamic body, which
	// the solver knows nothing about: it fought the solver's own position
	// integration, left velocity untouched so the wheel was bottomed again next
	// step, and corrected again. That loop is a rapid chatter you can see, and it
	// was there at every tick rate. The overshoot is now a bias inside the
	// velocity constraint above, which is where a penetration error belongs.

	UpdateFriction( v, h );
}

// ---------------------------------------------------------------------------
// Registry and the C API the wasm shim wraps
//
// Vehicles live here rather than in the shim so the shim stays a translation
// layer: it turns JavaScript's integers into ids and calls straight through.
// ---------------------------------------------------------------------------

/** Fraction of a bump-stop overshoot removed per second. Matches the static
 *  friction anchor's role: firm enough to look rigid, soft enough not to launch
 *  a chassis that is briefly deep. */
#define BUMP_STOP_ERP 0.6f

#define B3W_MAX_VEHICLES 64

static Vehicle g_vehicles[B3W_MAX_VEHICLES];

/**
 * Solve the STEPPING world's vehicles for one substep. Registered with the world
 * as its substep callback, so the force model keeps step with the solver instead
 * of being applied once per tick and left to be 32 substeps stale.
 *
 * `context` is the stepping world, stored by `b3StoreWorldId`. The registry is
 * global while worlds are not, and iterating all of it was wrong twice over: a
 * second world solved the first world's vehicles, and — because body ids are
 * reused as worlds are destroyed and created — a vehicle whose world had gone
 * would find a LIVE body at its old chassis id and drive it. That reads as an
 * unrelated car creeping or twitching, with nothing in its own world to blame.
 */
void b3w_vehicle_substep( float h, void* context )
{
	uint32_t stepping = (uint32_t)(uintptr_t)context;

	for ( int i = 0; i < B3W_MAX_VEHICLES; i++ )
	{
		if ( g_vehicles[i].used && g_vehicles[i].armed && b3StoreWorldId( g_vehicles[i].world ) == stepping )
		{
			b3w_vehicle_solve( &g_vehicles[i], h );
		}
	}
}

/**
 * Spend the cast state of a world's vehicles, called as its step completes.
 *
 * The pairing with `prepare` is the whole invariant: one prepare arms one step.
 * Anything that steps the world without preparing — a settle step, a tool, a
 * test — then simulates the vehicles as inert bodies rather than driving them
 * from stale contacts.
 */
void b3w_vehicle_disarm_world( b3WorldId world )
{
	uint32_t stepped = b3StoreWorldId( world );

	for ( int i = 0; i < B3W_MAX_VEHICLES; i++ )
	{
		if ( g_vehicles[i].used && b3StoreWorldId( g_vehicles[i].world ) == stepped )
		{
			g_vehicles[i].armed = false;
		}
	}
}

/**
 * Release every vehicle belonging to a world, called as that world is destroyed.
 *
 * A vehicle holds a chassis id and nothing owns it but this registry, so without
 * this a destroyed world leaks its slots permanently — 64 of them — and leaves
 * entries pointing at freed bodies.
 */
void b3w_vehicle_destroy_world( b3WorldId world )
{
	uint32_t dying = b3StoreWorldId( world );

	for ( int i = 0; i < B3W_MAX_VEHICLES; i++ )
	{
		if ( g_vehicles[i].used && b3StoreWorldId( g_vehicles[i].world ) == dying )
		{
			g_vehicles[i].used = false;
		}
	}
}

Vehicle* b3w_vehicle_get( int handle )
{
	if ( handle < 0 || handle >= B3W_MAX_VEHICLES || !g_vehicles[handle].used )
	{
		return NULL;
	}
	return &g_vehicles[handle];
}

int b3w_vehicle_create( b3WorldId world, int worldHandle, b3BodyId chassis )
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
		v->worldHandle = worldHandle;
		v->chassis = chassis;
		v->indexUpAxis = 1;
		v->indexForwardAxis = 0;
		v->owner = 0;
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

void b3w_vehicle_set_owner( int handle, unsigned int owner )
{
	Vehicle* v = b3w_vehicle_get( handle );
	if ( v == NULL )
	{
		return;
	}

	v->owner = (uint32_t)owner;
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

	// Park the wheel at its rest length rather than at zero. Zero means "wheel
	// retracted into its mount", which is what a renderer draws for a vehicle
	// whose controller has not run yet — and a chassis that spawns already
	// asleep never runs it, so the car sits with its wheels buried in the
	// ground. A wheel that has never been cast is at rest, not collapsed.
	w->info.suspensionLength = restLength;

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
