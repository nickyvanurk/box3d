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
 * error is r·(1 − cos(π/TREAD_SEGMENTS)), which at 16 segments is 0.5% of the
 * radius and under Box3D's 5 mm linear slop for any wheel the game builds.
 */

#include "box3d/box3d.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/** Bodies a vehicle may be told to ignore: its own, plus the player. */
/** Owner id reserved for bodies that are never ground for any vehicle. */
#define B3W_OWNER_NEVER_GROUND 0xFFFFFFFFu
/** Points per cap of the cast prism. See the header note on tread geometry. */
#define TREAD_SEGMENTS 16
/** Contact points a single wheel may rest on. Uncharted 4's number, and for
 *  their reason: one is a wheel that ping-pongs between surfaces. */
#define B3W_MAX_CONTACTS 3
/** Floats per wheel in `b3w_vehicle_read_wheels`. */
#define WHEEL_READBACK_STRIDE 25
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
/*
 * Error-correction RATES, in 1/s. Not fractions per step.
 *
 * These were written as per-step fractions and then divided by the tick length
 * at every use, which makes the correction stronger the shorter the tick — the
 * exact opposite of the tick-independence two of the comments claimed. Measured
 * across tick rates at matched solver cost, the controller degraded
 * monotonically as the rate rose: flat-ground vertical acceleration 8.66 m/s^2
 * at 24 Hz, 14.37 at 48, 17.82 at 60, 21.50 at 96, and paying double the
 * substeps did not rescue it. Every vehicle constant in the game was therefore
 * silently tuned to one tick rate.
 *
 * The values are the old fractions multiplied by 24, the rate they were tuned
 * at, so behaviour at 24 Hz is unchanged by construction and only OTHER rates
 * move — which is the whole point.
 */
static const float STATIC_FRICTION_RATE = 4.8f;
static const float BUMP_STOP_RATE = 14.4f;

/**
 * Damping ratio and push-out cap for the wheel's point constraint.
 *
 * Box3D's own contact defaults, deliberately: a tyre resting on a surface is the
 * problem the contact solver already solves well, and a second set of numbers
 * for it would be a second thing to tune. 10 is the world's
 * `contactDampingRatio`, 3 m/s its `contactSpeed`.
 */
/**
 * Contact frequency for a tyre on the ground.
 *
 * Measured on the isolation rig: 30 leaves a resting tyre in the surface, 60
 * costs 14 mm more depth on a hard landing than 120, and 120 and 190 are
 * identical because past that the limit is the suspension bottoming rather than
 * the contact yielding.
 */
static const float CONTACT_HERTZ = 120.0f;
static const float CONTACT_DAMPING_RATIO = 10.0f;

/** The steering rack: stiff enough that the wheel goes where it is told within a
 *  tick, damped enough not to ring. */
static const float STEERING_HERTZ = 30.0f;
static const float STEERING_DAMPING_RATIO = 5.0f;

/** The hard-point force's ceiling, as a multiple of the wheel's own maximum
 *  suspension force. Measured: 176 mm of landing depth at 1x, 124 at 3x, 117 at
 *  10x — it saturates, because past that the limit is the chassis's momentum
 *  against the travel available. */
static const float HARD_POINT_FORCE_SCALE = 3.0f;

/** Penetration a tyre is left to the constraint to resolve rather than being
 *  placed out of. Box3D's own linear slop. */
static const float PLACEMENT_DEADBAND = 0.005f;
/**
 * How fast a tyre may be pushed out of something it is already inside.
 *
 * Box3D's own `contactSpeed` — `types.c:19`, `3.0f * lengthUnits` — because a
 * tyre standing on the ground is a contact and a position error is a position
 * error. This was 30, ten times every other contact in the world, and no
 * measurement was recorded for it. A wheel bolted onto a hull that is already
 * resting starts a tyre deep in the surface, and at 30 m/s the point constraint
 * ejected it in a single tick and handed that to the chassis through the
 * suspension's hard limit: a 3x3 plate went 2.8 m into the air at 20 m/s and
 * tumbled. At the world's own speed it stands up on the wheel in five ticks and
 * settles at the ride height it would have had if the wheel had always been
 * there.
 *
 * Landings are unaffected: what answers a fast impact is the closing-velocity
 * term, not this cap, which only governs how quickly a STATIC overlap is
 * undone. Every drive and mechanical scenario is unchanged.
 */
static const float MAX_RECOVERY_SPEED = 3.0f;

/** How close the tyre's lowest corner must be to a surface to count as standing
 *  on it. A tyre that hangs above the ground carries no load, and the spring it
 *  hangs from has nothing to push off: applying the reaction to the ground
 *  anyway levitated a parked machine 235 mm into the air. */
static const float TYRE_TOUCH_CLEARANCE = 0.01f;


/** How far under the tyre's radius the bump stop sits. Measured: at 5 mm it
 *  carries a parked car, at 20 mm it levers a machine over on a climb, at 50 mm
 *  most of its value is gone. */
static const float BUMP_STOP_CLEARANCE = 0.035f;

/** How far inside the tread the wheel body's extent shape sits. */
static const float WHEEL_SHAPE_CLEARANCE = 0.06f;


/** Furthest a stuck contact may drift before the tyre is treated as having slipped. */
static const float MAX_STATIC_FRICTION_SLIP = 0.05f;

/** What is left of a tyre's sideways grip under a brake that has asked for
 *  everything the tyre can give. `ArcadeCarPhysics` uses 0.01 for its handbrake;
 *  this is the same idea, reached by how hard the brake is pulled. */
static const float BRAKE_SIDE_GRIP = 0.5f;

/** The other two transient states, from the same source: a stab of brake, and
 *  the moment after a landing. The arcade numbers are 0.5 and 0.02. */
static const float BRAKE_PRESS_SIDE_GRIP = 0.5f;
static const float BRAKE_PRESS_SLIPPERY_SECONDS = 0.4f;
static const float LANDING_SIDE_GRIP = 0.1f;
static const float LANDING_SLIPPERY_SECONDS = 0.5f;

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
	/**
	 * The surface's OWN normal, kept because `normalWs` is deliberately a lie.
	 *
	 * Forcing every contact normal to the suspension direction is what stops
	 * bumps stealing momentum, but the talk is equally clear about its cost:
	 * "the wheels no longer respond perfectly to collision, so we reimplement
	 * wheel collision response ourselves". Doing that needs the truth back.
	 */
	b3Vec3 surfaceNormalWs;
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
	/**
	 * The point constraint's impulse, accumulated across this TICK's substeps.
	 *
	 * Warm starting, and the term the soft constraint's `impulseScale` decays —
	 * without it the solve has nothing to take the bias energy back out of.
	 * Reset by `prepare`, for the same reason the friction budget is: the casts
	 * about to run replace the contacts these impulses were pushing against.
	 */
	float normalImpulse;
} WheelContact;

typedef struct
{
	b3Vec3 contactNormalWs;
	b3Vec3 groundNormalWs;
	b3Pos contactPointWs;
	float suspensionLength;
	/**
	 * Where the wheel is DRAWN, which is not where the spring thinks it is.
	 *
	 * The spring measures along the suspension against a plane whose normal is
	 * forced to the suspension axis. On a slope the real ground rises across the
	 * tyre, so its lower edge ends up under the surface — wheels visibly sunk
	 * into the floor, and worse the faster you go. This is the talk's own
	 * remedy: find the deepest point of the tyre against the REAL surface and
	 * lift the wheel along the suspension until it tangents the ground, never
	 * past the hard point. Placement only — the spring keeps `suspensionLength`.
	 */
	b3Pos hardPointWs;
	bool isInContact;
	/** The kept contacts, nearest first. Index 0 mirrors the singular fields
	 *  above, which friction, reporting and the wheel transform still read. */
	WheelContact contacts[B3W_MAX_CONTACTS];
	int contactCount;
	/** Hits the cast offered this tick, before dedup and the three-slot cap. */
	int considered;
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

	/**
	 * The wheel's own rigid body.
	 *
	 * The Uncharted 4 talk's first architectural fact: "a rigid body for each
	 * wheel... we turned off collision on the wheels in favour of Havok's point
	 * constraint system". It carries NO SHAPES and never will — a body with no
	 * shapes cannot collide, which is how "collision off" is expressed here, and
	 * the shapecast above stays the only way this wheel finds the ground.
	 *
	 * Kinematic for now, driven from the suspension length the spring computes,
	 * so it exists and tracks the wheel without changing a single force. It
	 * becomes dynamic when the point constraint takes over contact response and
	 * there is something for the constraint to act on.
	 */
	b3BodyId body;
	/**
	 * The suspension, as a joint between the wheel body and the chassis.
	 *
	 * Box3D's wheel joint is the right primitive and not a substitution: it is a
	 * spring along one axis with limits, free spin about the axle, and nothing
	 * else — which is the talk's suspension exactly. (Its axes are not the ones
	 * the header comments claim: the suspension axis is frame A's local X,
	 * `wheel_joint.c:459`, and the spin axis is frame B's local Z, `:472`.)
	 *
	 * The POINT CONSTRAINT is not a joint and never will be — no Box3D joint has
	 * that shape, and using one would be a substitution. It is solved by hand,
	 * below.
	 */
	b3JointId joint;
	/**
	 * Penetration the placement could NOT remove, because the wheel had reached
	 * its hard point — the talk's remainder, which the upward force on the hard
	 * point answers.
	 */
	float hardPointPenetration;
	/**
	 * How fast the PLACEMENT compressed the suspension this tick.
	 *
	 * The damper reads the bodies' velocities, which is blind to a teleport; this
	 * gives it the motion it cannot see. Differentiating the compression INSTEAD
	 * of reading velocities was tried and left a parked car undamped, drifting
	 * 4.89 m.
	 */
	/** Diagnostics only — what the last tick asked the placement and the hard
	 *  point to do, so a probe can see which of the two is carrying the car. */
	float placementNeeded;
	float hardPointForce;
	float springForce;
	float springRate;
	float springSpringTerm;
	float springCompression;
	/**
	 * The tyre's mass, taken OFF the chassis when the wheel became a body.
	 *
	 * A machine's total weight is what players tune around, so 24 wheels
	 * appearing as 24 new bodies must not make it heavier. The chassis is built
	 * from pieces that include the wheel, so the mass already exists — this moves
	 * it rather than adding it.
	 */
	float mass;

	float rotation;
	float deltaRotation;
	/**
	 * The wheel's OWN spin rate, in rad/s, positive rolling forward.
	 *
	 * Real state, integrated from the torques acting on the wheel, rather than
	 * read off the chassis. `deltaRotation` used to be defined as
	 * `groundSpeed / radius`, which makes the wheel match the ground by
	 * construction — so slip could not exist, and with it neither could wheel
	 * spin, engine RPM taken from the fastest wheel, differentials, or the
	 * torque feedback the Uncharted 4 talk builds its drivetrain out of. All of
	 * those need a wheel that is allowed to disagree with the ground.
	 */
	float angularVelocity;
	/**
	 * Rotational inertia, kg*m^2. A solid disc is 0.5*m*r^2; the default assumes
	 * a wheel of ~20 kg and is overridden per wheel from the game.
	 *
	 * It sets how quickly a tyre can break traction and how long it spins up
	 * once it has: too low and the wheel snaps to any speed asked of it, which
	 * is the no-slip behaviour this replaces.
	 */
	float inertia;
	/**
	 * The bump stop, and whether it has been let into the world yet.
	 *
	 * A wheel is not always added to a machine already standing on its wheels.
	 * Bolt one onto a hull resting on the ground and the stop — a sphere nearly
	 * a tyre wide, at the hard point — is born inside the surface, and the
	 * contact solve evicts it in a single tick: 262 mm on a plate with one
	 * wheel, which the player sees as a pop.
	 *
	 * So it enters the world only once a cast says it is clear of it, and never
	 * leaves again. Nothing is lost by waiting: while the suspension is past its
	 * hard point the machine is resting on whatever the stop is buried in, and a
	 * stop cannot catch a bottoming that has already happened.
	 */
	b3ShapeId stopShape;
	/**
	 * The tyre as COLLISION, PhysX's way (`SnippetVehicleContactMod.cpp`): a shape
	 * where the tyre is, meeting everything, whose contacts UNDER the wheel are
	 * dropped because the suspension owns those, and whose others — another car's
	 * door, its nose, a wall, the lip of a kerb too tall to roll onto — push the
	 * machine at the suspension's application point and carry the tread's speed.
	 * See `UpdateTyre` and `b3w_tyre_presolve`.
	 *
	 * On the CHASSIS, for the bump stop's reason: the wheel body is teleported
	 * onto the surface every tick. PhysX moves its wheel shape with the suspension
	 * every update; re-shaping one here throws its contacts away, so the box is
	 * sized once to span the travel the tyre actually occupies — from fully
	 * compressed down to its loaded ride height — and never reaches below a
	 * loaded tread.
	 */
	b3ShapeId tyreShape;
	float tyreTop;
	float tyreBottom;
	b3Filter stopFilter;
	bool stopArmed;

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
	/** The anchor in the GROUND BODY's frame, when that body moves. */
	bool anchorOnGround;
	b3Vec3 anchorLocal;

	float clippedInvContactDotSuspension;
	float suspensionRelativeVelocity;
	float wheelSuspensionForce;
	float skidInfo;
	/** Did the friction circle bind for this tyre this tick? */
	bool isSliding;
	/** How much of this wheel's mass is already off the chassis, so a repeat call
	 *  moves the difference instead of the whole thing again. */
	float massTakenFromChassis;
	/** Seconds left of the two transient slippery states — see `UpdateFriction`. */
	float landedSlipperyTime;
	float brakePressSlipperyTime;
	bool brakeWasHeld;
	/**
	 * The friction of the ground under this tyre, from the shape it is standing
	 * on. 1.0 when it is not standing on anything.
	 *
	 * The talk keeps tyre and surface separate and multiplies them: "Define
	 * friction for both the tires and the surfaces they drive on... Calculate
	 * the coefficient of friction by multiplying tire friction by surface
	 * friction." A tyre grade is then a property of the build and the ground is
	 * a property of the world, which is what lets ice be slippery for everyone
	 * without touching a single vehicle.
	 */
	float surfaceFriction;
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
	/** No tyre latches a static-friction anchor during the next step — see
	 *  `b3w_vehicle_release_anchors`. Cleared, like `armed`, as it completes. */
	bool anchorsReleased;
	b3WorldId world;
	/** The shim's world handle, for the body-owner table. */
	int worldHandle;
	b3BodyId chassis;
	int wheelCount;
	/*
	 * Wheels are a GROWABLE array, not a fixed one, because the player decides
	 * how many wheels a machine has and any constant we pick is a constant
	 * someone builds past. This was `Wheel wheels[32]`, and a 76-wheel tank
	 * silently drove on 32 of them: `add_wheel` refused the rest and the caller
	 * had no way to find out.
	 *
	 * Anything holding a `Wheel*` across an add is holding a dangling pointer —
	 * the grow reallocates. Take the index, not the address.
	 */
	int wheelCapacity;
	Wheel* wheels;
	b3Vec3* forwardWs;
	b3Vec3* axle;
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
	/**
	 * The chassis mass as the wheels' takes last left it — see `RetakeWheelMass`.
	 * Zero until a wheel has taken anything.
	 */
	float chassisMassAsLeft;
	/** This vehicle's slot, so a tyre shape can name its wheel. */
	int handle;
} Vehicle;

static bool MachineFilter( const Vehicle* v, b3Filter* filter );

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
	/** The friction of the SURFACE that was hit — mud, rock, road. */
	float surfaceFriction;
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
	/**
	 * The hard point this cast started above, and the suspension's own down.
	 *
	 * The cast deliberately begins `2r + travel` ABOVE the hard point so a wheel
	 * that starts a tick inside terrain is not answered with fraction 0. That
	 * head start is for reaching PAST geometry the wheel is already in — it is
	 * not a licence to find ground overhead, and without a test it is exactly
	 * that: a surface most of a metre above the hard point sails through the
	 * up-facing filter, because a roof genuinely does face up.
	 *
	 * Measured on two cars: 0.875 m of head start puts the cast start above
	 * another vehicle's roof at 1.0 m, so the suspension stood on that roof and
	 * pushed the car up onto it, 70 ticks before the two chassis ever touched.
	 */
	b3Pos hardPoint;
	/**
	 * Where the swept tyre starts and how far it goes, the suspension's own
	 * down and the axle: what PhysX's acceptance test asks of a hit — see
	 * `IsUnderTheWheel`.
	 */
	b3Pos castStart;
	b3Vec3 translation;
	b3Vec3 suspensionDown;
	b3Vec3 axle;
	/** How many hits the cast offered, before dedup and the three-slot cap. A
	 *  shape cast reports the earliest hit PER SHAPE, and terrain is one mesh
	 *  shape holding every triangle — this is how that was caught. */
	int considered;
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

/**
 * The category a tyre shape carries on top of its machine's. Nothing stands on,
 * picks or places onto a tyre (the shim's ray queries skip it) and no wheel's
 * sweep takes one for ground.
 */
#define B3W_TYRE_CATEGORY 0x0010ull
/** How much smaller the tyre's box is than the tyre the cast carries, so the two never argue (archlab's figure). */
static const float TYRE_CLEARANCE = 0.01f;
/** Rubber against what it runs into — archlab's tyre friction. */
static const float TYRE_FRICTION = 0.6f;
/**
 * The share of the tread's speed a rigid wheel contact is given: PhysX's
 * WHEEL_TANGENT_VELOCITY_MULTIPLIER (`SnippetVehicleContactMod.cpp`).
 */
static const float TYRE_TREAD_SHARE = 0.1f;
/** A tyre's user-data tag: which vehicle and wheel it belongs to. */
#define B3W_TYRE_TAG 0x40000000

/**
 * PhysX's threshold for "under the wheel", as a cosine: the SDK's contact-
 * modification sample sets both of its reject angles to 45 degrees
 * (`SnippetVehicleContactMod.cpp`, POINT_REJECT_ANGLE and NORMAL_REJECT_ANGLE).
 */
static const float UNDER_THE_WHEEL_COS = 0.70710678f;
/**
 * Past what angle a TYRE treats a surface as something it runs into: 50
 * degrees, five past the sweep's 45. A surface right at PhysX's boundary
 * flipped between road and wall from tick to tick as the machine pitched — a
 * single wheel climbing 30 degrees pitched 15 the other way and was kicked off
 * the slope at 4 m/s. Between the two it is neither, so nothing pushes on it
 * twice.
 */
static const float TYRE_WALL_COS = 0.64278761f;
/** Two normals within 15 degrees describe the same surface. */
static const float SAME_SURFACE_COS = 0.96592583f;

/**
 * Is this a surface the wheel ROLLS on, or something it runs INTO?
 *
 * PhysX's sweep acceptance (`PxVehicleUpdate.cpp`, intersectCylinderPlane),
 * both of its tests:
 *
 * - the surface faces the suspension: its normal within 45 degrees of the
 *   suspension's up;
 * - the contact is under the wheel: the direction from the swept tyre's centre
 *   to the contact, in the wheel's own plane, within 45 degrees of the
 *   suspension's down.
 *
 * Everything else is the tyre's FRONT or side meeting something — another
 * car's door, its wedge nose, a wall, the lip of a kerb taller than the tyre
 * can take — and is answered by the tyre's collision shape, not by the spring.
 * The filter this replaced accepted anything facing within 80 degrees of world
 * up anywhere on the tyre, so a nose or a door the tread brushed became ground
 * and the suspension threw the car up onto it.
 */
static bool IsUnderTheWheel( const WheelCast* ctx, b3Pos point, b3Vec3 normal, float fraction )
{
	if ( -b3Dot( normal, ctx->suspensionDown ) < UNDER_THE_WHEEL_COS )
	{
		return false;
	}

	b3Pos centre = AddPosVec( ctx->castStart, Scale( ctx->translation, fraction ) );
	b3Vec3 toPoint = SubPos( point, centre );
	toPoint = Sub3( toPoint, Scale( ctx->axle, b3Dot( toPoint, ctx->axle ) ) );
	float length = b3Length( toPoint );
	if ( length < 1.0e-6f )
	{
		return true;
	}
	return b3Dot( toPoint, ctx->suspensionDown ) >= UNDER_THE_WHEEL_COS * length;
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

	/*
	 * A wheel is not standing on something ABOVE its own hard point.
	 *
	 * The cast starts well above the hard point on purpose, and everything in
	 * that head start is geometry the wheel is passing THROUGH, not geometry it
	 * rests on. The up-facing filter cannot tell the difference: another
	 * vehicle's roof faces up as honestly as a road does.
	 *
	 * Measured, and it is the whole of "a car rides up another car": at the tick
	 * the front suspension spiked to four times its load, the front wheels'
	 * ground body flipped from the world to the other vehicle while the two
	 * chassis were still 17 cm apart and stayed untouching for another 70 ticks.
	 * The suspension was pushing off a roof 0.5 m above the hard point.
	 *
	 * The hard point is the datum rather than the axle because it is known
	 * before the cast resolves — the axle's position is the answer the cast
	 * exists to produce — and because it is the conservative choice: everything
	 * the tyre can legitimately reach is below it, including terrain a wheel has
	 * sunk into on a climb, which is the case the head start was added for.
	 *
	 * Against WORLD UP, for exactly the reason written above the surface filter:
	 * chassis tilt must not move the accept boundary around. Measured with the
	 * suspension direction instead — a T-boned car slid 164 m rather than 16,
	 * because once it was knocked askew the rejection plane swung with it and
	 * threw away the ground the car was sitting on, leaving it to coast with no
	 * wheel touching anything.
	 */
	if ( b3Dot( SubPos( point, ctx->hardPoint ), ctx->worldUp ) > 0.0f )
	{
		return -1.0f;
	}

	// Another machine's tyre is not a surface either: it is where a wheel is.
	if ( b3Shape_GetFilter( shapeId ).categoryBits & B3W_TYRE_CATEGORY )
	{
		return -1.0f;
	}

	if ( !IsUnderTheWheel( ctx, point, normal, fraction ) )
	{
		return -1.0f;
	}

	ctx->considered++;

	WheelHit hit = { .fraction = fraction,
					 .surfaceFriction = b3Shape_GetFriction( shapeId ),
					 .normal = normal,
					 .point = point,
					 .body = body,
					 .bodyId = hitBody };
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
/*
 * A ground that is ANIMATED: kinematic.
 *
 * Every velocity here was measured against a ground assumed to be still, gated
 * on `b3_dynamicBody`. A KINEMATIC body is the engine's own notion of "animated
 * by something else at a known velocity" — a ferry deck driven from a network
 * stream — and a tyre that reads it as stationary lets the deck slide out from
 * under the car at the deck's full speed. Only a DYNAMIC ground can take an
 * impulse back; a kinematic one is infinitely massive and merely has a velocity.
 */
static bool GroundMoves( b3BodyId ground )
{
	return B3_IS_NON_NULL( ground ) && b3Body_IsValid( ground ) && b3Body_GetType( ground ) == b3_kinematicBody;
}

static bool GroundIsDynamic( b3BodyId ground )
{
	return B3_IS_NON_NULL( ground ) && b3Body_IsValid( ground ) && b3Body_GetType( ground ) == b3_dynamicBody;
}

static b3Vec3 GroundVelocityAt( b3BodyId ground, b3Pos point )
{
	return GroundMoves( ground ) ? b3Body_GetWorldPointVelocity( ground, point ) : B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.0f };
}

/*
 * The static-friction anchor lives in the ground's own frame when the ground
 * moves. Carrying a world point by the ground's velocity was measured wrong by
 * about a fifth of a tick's travel, which the bias read as slip and turned into
 * a parked car creeping forward at 1.2% of the deck's speed.
 */
static void LatchAnchor( Wheel* w, b3Pos point )
{
	w->hasAnchor = true;
	w->anchorOnGround = GroundMoves( w->info.groundBodyId );
	if ( w->anchorOnGround )
	{
		w->anchorLocal = b3Body_GetLocalPoint( w->info.groundBodyId, point );
	}
	else
	{
		w->staticFrictionAnchor = point;
	}
}

static b3Pos AnchorWs( const Wheel* w )
{
	if ( w->anchorOnGround && GroundMoves( w->info.groundBodyId ) )
	{
		return b3Body_GetWorldPoint( w->info.groundBodyId, w->anchorLocal );
	}
	return w->staticFrictionAnchor;
}

static void ShapeCastWheel( Vehicle* v, int wheelId, float dt )
{
	b3Vec3 worldUp = AxisVector( v->indexUpAxis );
	Wheel* w = &v->wheels[wheelId];
	b3Pos source = w->info.hardPointWs;

	b3Vec3 axle = TryNormalize( w->wheelAxleWs, B3_LITERAL( b3Vec3 ){ 0.0f, 1.0f, 0.0f } );
	b3Vec3 direction = TryNormalize( w->wheelDirectionWs, w->wheelDirectionWs );

	/*
	 * Start WELL above the hard point, not just clear of it.
	 *
	 * The talk is specific about this and about why: "another tweak was to cast
	 * the ray from above the hard point — in this case well above the hard
	 * point. There are cases where the nose of the vehicle would spend a frame
	 * already in collision and get stuck. Moving the start of the cast up made
	 * sure that never happens."
	 *
	 * One radius was not enough. Climbing a slope the ground rises faster than
	 * the suspension can travel — 0.375 m per tick at 15 m/s on a 60% grade,
	 * three times the whole travel — so the wheel begins the next cast already
	 * inside the terrain. Box3D answers an overlapped cast with fraction 0 and
	 * gives up on continuous collision, which is the wheel getting stuck exactly
	 * as described.
	 *
	 * Starting a full suspension travel higher costs nothing but cast length:
	 * the offset is subtracted straight back out of the time of impact, so the
	 * suspension length is unchanged for every cast that was already working.
	 */
	float r = w->radius;
	/*
	 * As far as the suspension can reach. The talk casts "from the hard point to
	 * the furthest point the suspension can extend"; this stopped at rest plus a
	 * wheel RADIUS, which is not a suspension quantity at all.
	 */
	float raylen = w->suspensionRestLength + w->maxSuspensionTravel;

	/*
	 * Plus how far this wheel will TRAVEL toward the ground this tick.
	 *
	 * The suspension's reach is the right length for a car that is driving: the
	 * wheel cannot go further than that, so nothing beyond it matters. It is the
	 * wrong length for one that is falling. At 22 m/s a tick moves the wheel
	 * 0.9 m while the cast looks 0.5 m ahead, so it reports NOTHING, the point
	 * constraint has no contact to hold, and the tyre travels the whole tick
	 * unopposed and arrives a quarter of a metre inside the terrain. The next
	 * tick's cast finds it already buried, which is when a player sees it.
	 *
	 * Measured across drop heights and impact speeds: penetration tracked impact
	 * speed almost exactly, which is the signature of a per-tick distance rather
	 * than of any force or rate. The talk casts the suspension's reach because at
	 * their tick rate that already covers a tick of falling; at 24 Hz it does not,
	 * so the distance the wheel is about to cover is added explicitly.
	 */
	b3Vec3 wheelVel = B3_IS_NULL( w->body ) ? b3Body_GetLinearVelocity( v->chassis )
											: b3Body_GetLinearVelocity( w->body );
	float closing = b3Dot( direction, wheelVel ) * dt;
	if ( closing > 0.0f )
	{
		raylen += closing;
	}
	float offset = 2.0f * r + w->maxSuspensionTravel;

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
					   .hardPoint = source,
					   .castStart = castStart,
					   .translation = Scale( direction, raylen + offset ),
					   .suspensionDown = direction,
					   .axle = axle,
					   .considered = 0,
					   .count = 0 };

	float maxToi = raylen + offset;
	/*
	 * EVERY hit, not just the nearest — the talk's step 5. The ordinary cast
	 * reports the earliest hit PER SHAPE and terrain is one mesh shape holding
	 * every triangle, so the three slots, the dedup and the coupled multi-row
	 * solve below never saw more than one input.
	 */
	b3World_CastShapeAll( v->world, castStart, &proxy, Scale( direction, maxToi ), b3DefaultQueryFilter(), OnWheelHit,
						  &cast );

	// Filter: chassis pose. Drop every hit when the suspension does not point
	// broadly downward — sideways, nose-standing or inverted, where a
	// heightfield's always-outward normal would mislead a normal-only test. The
	// surface filter is applied per hit, inside the cast callback.
	float suspensionDownward = -b3Dot( direction, worldUp );
	/*
	 * Where the cast WENT, recorded whether or not it found anything.
	 *
	 * These three only exist for the debug view, and they were written inside the
	 * accepted branch — so a wheel that found no ground kept the last values it
	 * had and its prism stayed at a stale angle while the machine turned. On a
	 * tank the corner wheels are off the ground most of the time, so most of the
	 * casts on screen were frozen.
	 */
	w->info.lastCastStart = castStart;
	w->info.lastAxleWs = axle;
	w->info.lastToi = maxToi;

	bool accepted = cast.count > 0 && suspensionDownward > cosf( w->suspensionRejectAngle );

	w->info.groundBody = -1;
	w->info.groundBodyId = b3_nullBodyId;
	w->info.considered = cast.considered;
	w->info.contactCount = 0;
	w->surfaceFriction = 1.0f;

	if ( accepted )
	{
		float maxSuspensionLength = w->suspensionRestLength + w->maxSuspensionTravel;
		b3Vec3 axial = Scale( w->wheelDirectionWs, -1.0f );

		for ( int c = 0; c < cast.count; c++ )
		{
			const WheelHit* hit = &cast.hits[c];
			WheelContact* contact = &w->info.contacts[c];

			// EVERY contact normal is the suspension direction. Not the surface's.
			//
			// Straight from the Uncharted 4 talk, which is explicit about it:
			// "We didn't like the collision response to surfaces whose normal
			// isn't aligned with the suspension direction. Hitting a bump could
			// cause the vehicle to completely lose momentum... so we fake it and
			// tell Havok that all normals are in the direction of the suspension.
			// This produces a much smoother, player-favoring response."
			//
			// Using the true normal within 60 degrees, as this did, means every
			// facet of the ground tilts the spring force and steals a little
			// forward momentum. On a dual-contoured collider — flat plates meeting
			// at creases — that is a bump at every crease, including across ground
			// that looks and IS flat. The surface is not rough; the response was.
			//
			// Drivability still reads the REAL normal, in the cast callback: a wall
			// is rejected on its own geometry, not on this one. What changes here
			// is only the direction the suspension pushes once a surface has been
			// accepted, which is the one thing the talk says to lie about.
			contact->surfaceNormalWs = hit->normal;
			contact->normalWs = axial;

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
			// Within the REAL surface, not the faked one: `normalWs` is deliberately
			// a lie, and removing the component along it leaves a vector in a plane
			// perpendicular to the suspension, which on a slope is not the ground.
			// Measured with one cast on a 30 degree plane: the contact point ended
			// 76.86 mm above it.
			b3Vec3 alongPlane = Sub3( toCenter, Scale( hit->normal, b3Dot( hit->normal, toCenter ) ) );
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

		/*
		 * The placement correction that stood here is GONE, with the port: it
		 * compensated for a wheel that was neither a body nor constrained. See
		 * `PlacementAlongSuspension`.
		 */

		w->surfaceFriction = cast.hits[0].surfaceFriction > 0.0f ? cast.hits[0].surfaceFriction : 1.0f;

		w->info.lastToi = cast.hits[0].fraction * maxToi;
		w->info.lastCastOffset = offset;
		w->info.lastCastStart = castStart;
		w->info.lastAxleWs = axle;

		b3Vec3 velAtContact = Sub3( b3Body_GetWorldPointVelocity( v->chassis, w->info.contactPointWs ),
									GroundVelocityAt( w->info.groundBodyId, w->info.contactPointWs ) );
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

/**
 * Re-measure every wheel's contact against the LIVE chassis, once per substep.
 *
 * The cast runs once per tick and the chassis moves under it, so both the
 * suspension LENGTH and the contact POINT go stale between casts. Sliding the
 * point along under the wheel keeps its lever arm honest; doing that in the
 * plane of the REAL surface is what makes it follow a hill.
 *
 * The spring itself is gone from here: the suspension is a force between the
 * wheel body and the chassis now (`ApplySuspensionForce`), and the ground's push
 * comes from the point constraint.
 */
static void UpdateContacts( Vehicle* v, float h )
{
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		if ( !w->info.isInContact )
		{
			continue;
		}

		b3WorldTransform t = b3Body_GetTransform( v->chassis );
		b3Pos hardPointNow = AddPosVec( t.p, b3RotateVector( t.q, w->chassisConnectionPointCs ) );
		float maxSuspensionLength = w->suspensionRestLength + w->maxSuspensionTravel;

		for ( int c = 0; c < w->info.contactCount; c++ )
		{
			WheelContact* contact = &w->info.contacts[c];

			float gap = b3Dot( contact->normalWs, SubPos( hardPointNow, contact->pointWs ) ) - contact->planeOffset;
			StoreSuspensionLength( contact, gap * contact->clippedInvContactDotSuspension, maxSuspensionLength );

			// Under the wheel BODY's centre, and within the REAL surface.
			b3Pos centerNow = B3_IS_NULL( w->body )
								  ? AddPosVec( hardPointNow, Scale( w->wheelDirectionWs, contact->suspensionLength ) )
								  : b3Body_GetPosition( w->body );
			b3Vec3 slideNormal = contact->surfaceNormalWs;
			b3Vec3 toCenterNow = SubPos( centerNow, contact->pointWs );
			b3Vec3 inPlane = Sub3( toCenterNow, Scale( slideNormal, b3Dot( slideNormal, toCenterNow ) ) );
			contact->pointWs = AddPosVec( contact->pointWs, inPlane );

			b3Vec3 velNow = Sub3( b3Body_GetWorldPointVelocity( v->chassis, contact->pointWs ),
								  GroundVelocityAt( w->info.groundBodyId, contact->pointWs ) );
			if ( c == 0 )
			{
				w->suspensionRelativeVelocity =
					b3Dot( contact->normalWs, velNow ) * contact->clippedInvContactDotSuspension;
			}
		}

		const WheelContact* primary = &w->info.contacts[0];
		w->info.suspensionLength = primary->suspensionLength;
		w->info.bottomedOut = primary->bottomedOut;
		w->info.bottomOutOvershoot = primary->bottomOutOvershoot;
	}

	(void)h;
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

	if ( GroundIsDynamic( ground ) )
	{
		denom1 = ImpulseDenominator( ground, contactPos, dir );
	}

	/*
	 * SafeInv, because both denominators can be zero at once.
	 *
	 * `ImpulseDenominator` is `1/m + n . ((I^-1 (r x n)) x r)`, and a STATIC body
	 * contributes nothing to either term: its mass reads 0, so `InvMass` returns
	 * 0, and its world inverse inertia is the zero matrix. `denom1` is only ever
	 * non-zero for a DYNAMIC ground, and terrain is static. So a chassis that has
	 * been frozen, with its wheels on the ground, divides 1 by 0 here: the
	 * friction impulse comes back +/-Inf, or NaN when `vrel + biasVelocity` is
	 * itself 0 and the product is 0 * Inf.
	 *
	 * That value is then applied to bodies, and `b3FinalizeBodiesTask` reports
	 * `unstable` on the next step. In the game it presents as the whole world
	 * going non-finite the moment a machine is frozen with the physgun.
	 *
	 * Zero is the physically right answer, not a fudge: the denominator IS how
	 * much the contact point moves per unit impulse, so zero means the point
	 * cannot be moved at all, and no impulse can change it. This is the same
	 * convention `SafeInv` above already documents.
	 */
	float jacDiagAbInv = SafeInv( denom0 + denom1 );

	b3Vec3 vel1 = b3Body_GetWorldPointVelocity( chassis, contactPos );
	// A DYNAMIC ground keeps the velocity term it always had here; a KINEMATIC
	// one gains it. Subtracting a dynamic ground's velocity elsewhere in the
	// controller was tried and measured worse on a rider parked on a driven
	// ferry, so the kinematic case alone carries the new terms.
	b3Vec3 vel2 = GroundIsDynamic( ground ) || GroundMoves( ground ) ? b3Body_GetWorldPointVelocity( ground, contactPos )
																	   : B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.0f };
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
		bool groundIsDynamic = hasGround && GroundIsDynamic( w->info.groundBodyId );
		// Speed RELATIVE TO THE GROUND. A car parked on a deck doing 25 m/s is
		// parked, and the anchor it latches is carried along with the deck.
		b3Vec3 groundVel = hasGround ? GroundVelocityAt( w->info.groundBodyId, w->info.contactPointWs )
									 : B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.0f };
		// The tick's own road-speed figure on still ground, exactly as before;
		// only a moving ground needs the difference taken here.
		float relSpeed = GroundMoves( w->info.groundBodyId )
							 ? b3Length( Sub3( b3Body_GetLinearVelocity( v->chassis ), groundVel ) )
							 : vehicleSpeed;
		bool canStick = hasGround && w->engineForce == 0.0f && relSpeed < STATIC_FRICTION_MAX_SPEED &&
						!groundIsDynamic && !v->anchorsReleased;

		if ( canStick )
		{
			if ( !w->hasAnchor )
			{
				LatchAnchor( w, w->info.contactPointWs );
			}

			// A tyre sticks only up to a finite slip distance; past that it has
			// broken traction and grips somewhere new. Without this the anchor
			// hauls a slow vehicle back to where it stood seconds ago.
			b3Vec3 drift = SubPos( w->info.contactPointWs, AnchorWs( w ) );
			if ( b3Length( drift ) > MAX_STATIC_FRICTION_SLIP )
			{
				LatchAnchor( w, w->info.contactPointWs );
			}
		}
		else
		{
			w->hasAnchor = false;
		}

		w->sideImpulse = 0.0f;
		w->forwardImpulse = 0.0f;
		w->isSliding = false;
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
			b3Vec3 slid = SubPos( w->info.contactPointWs, AnchorWs( w ) );
			lateralBias = b3Dot( slid, v->axle[i] ) * STATIC_FRICTION_RATE;
		}

		// Against the ground's own velocity whenever it has one. A kinematic
		// ground contributes no mass to the denominator and keeps the anchor
		// bias, so it is the unilateral solve measured in the ground's frame.
		if ( GroundIsDynamic( w->info.groundBodyId ) )
		{
			w->sideImpulse = ResolveSingleBilateral( v->chassis, w->info.contactPointWs, w->info.groundBodyId,
													w->info.contactPointWs, v->axle[i], 0.0f );
		}
		else if ( GroundMoves( w->info.groundBodyId ) )
		{
			w->sideImpulse = ResolveSingleBilateral( v->chassis, w->info.contactPointWs, w->info.groundBodyId,
													w->info.contactPointWs, v->axle[i], lateralBias );
		}
		else
		{
			w->sideImpulse =
				ResolveSingleUnilateral( v->chassis, w->info.contactPointWs, v->axle[i], lateralBias );
		}

		w->sideImpulse *= w->sideFrictionStiffness;

		/*
		 * A braked tyre loses SIDEWAYS grip, which is what a handbrake is.
		 *
		 * The talk lists it — "handbrake = reduced rear tyre friction" — and it
		 * was dropped here for measuring 0.8 degrees, because the reduction was
		 * mild. The arcade controller in ~/dev/ArcadeCarPhysics shows what the
		 * number wants to be: `handBrakeSlipperyK = 0.01`, a NINETY-NINE percent
		 * cut to lateral friction, ramped in over a couple of seconds. Nothing
		 * about slip angles or friction circles; the sideways force is simply
		 * mostly gone while the brake is held.
		 *
		 * Scaled by how hard the brake is asked for against what the tyre could
		 * hold, so a coasting car's token brake does nothing and a real one
		 * unsticks the back of the car.
		 */
		/*
		 * Transient slippery states, the way `ArcadeCarPhysics` does it.
		 *
		 * Three of them, each a coefficient on the sideways force that decays on
		 * its own clock, and the SMALLEST wins — held brake, the moment a brake is
		 * first pressed, and the moment after a landing. None of them is a slip
		 * angle or a friction circle; they are the arcade admission that a tyre's
		 * grip is a story about what just happened to it, not only about the load
		 * on it right now.
		 */
		float sideGrip = 1.0f;

		// Held brake: the handbrake. Scaled by how hard the brake is asked for
		// against what the tyre can hold, so a coasting car's token brake does
		// nothing and a real one unsticks the back of the car.
		float hold = w->wheelSuspensionForce * v->stepDt * w->frictionSlip * w->surfaceFriction;
		if ( w->brake > 0.0f && hold > 0.0f )
		{
			float demandRatio = w->brake * SafeInv( hold );
			if ( demandRatio > 1.0f )
			{
				demandRatio = 1.0f;
			}

			float braked = 1.0f + demandRatio * ( BRAKE_SIDE_GRIP - 1.0f );
			sideGrip = braked < sideGrip ? braked : sideGrip;
		}

		// The moment a brake is FIRST pressed: a short, milder loss, so a stab of
		// brake mid-corner steps the car out before the held-brake term takes
		// over. `brakeSlipperyK` in the arcade controller, on a rising edge.
		bool braking = w->brake > 0.0f;
		if ( braking && !w->brakeWasHeld )
		{
			w->brakePressSlipperyTime = BRAKE_PRESS_SLIPPERY_SECONDS;
		}
		w->brakeWasHeld = braking;

		if ( w->brakePressSlipperyTime > 0.0f )
		{
			w->brakePressSlipperyTime -= v->stepDt;
			float t = w->brakePressSlipperyTime * SafeInv( BRAKE_PRESS_SLIPPERY_SECONDS );
			t = t < 0.0f ? 0.0f : t;
			float pressed = 1.0f + t * ( BRAKE_PRESS_SIDE_GRIP - 1.0f );
			sideGrip = pressed < sideGrip ? pressed : sideGrip;
		}

		// And after a landing: full sideways grip the instant a wheel touches is a
		// constraint appearing under a moving machine, which is what snaps a car
		// sideways when it comes down on one corner.
		if ( !w->info.isInContact )
		{
			w->landedSlipperyTime = LANDING_SLIPPERY_SECONDS;
		}
		else if ( w->landedSlipperyTime > 0.0f )
		{
			w->landedSlipperyTime -= v->stepDt;
			float t = w->landedSlipperyTime * SafeInv( LANDING_SLIPPERY_SECONDS );
			t = t < 0.0f ? 0.0f : t;
			// Smoothstep, so grip returns without a step in it.
			t = t * t * t * ( t * ( t * 6.0f - 15.0f ) + 10.0f );
			float landed = 1.0f + t * ( LANDING_SIDE_GRIP - 1.0f );
			sideGrip = landed < sideGrip ? landed : sideGrip;
		}

		w->sideImpulse *= sideGrip;

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
				/*
				 * A tyre can never deliver more than its grip, braking included.
				 *
				 * This took the LARGER of the brake and the grip, so any brake
				 * number above the tyre's budget let the wheel cancel the car's
				 * whole forward velocity outright — braking was a constraint, not
				 * a force. Measured on the builtin Car from 24 m/s: stopped in
				 * 0.21 s over 2.2 m, peaking at 16.6 g. A real car manages about
				 * 1 g and these tyres about 1.5; 16.6 is not a value physics can
				 * produce, and it is why braking felt like hitting a wall.
				 *
				 * A STATIONARY anchored tyre is still held by its full grip —
				 * that is what stops a parked car sliding downhill, and it is a
				 * limit rather than a licence, so it stays.
				 */
				// dt is the SUBSTEP, not the tick. This budget caps an impulse
				// applied on every one of the 32 substeps, so charging it a whole
				// tick's worth each time let a tyre deliver 32 times its own grip.
				float grip = w->wheelSuspensionForce * dt * w->frictionSlip * w->surfaceFriction;
				float maxImpulse = w->brake < grip ? w->brake : grip;
				if ( w->hasAnchor && w->brake > 0.0f )
				{
					maxImpulse = grip;
				}

				float biasVelocity = 0.0f;
				if ( w->hasAnchor )
				{
					b3Vec3 slid = SubPos( w->info.contactPointWs, AnchorWs( w ) );
					biasVelocity = b3Dot( slid, v->forwardWs[i] ) * STATIC_FRICTION_RATE;
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
					// The tyre has broken traction, so what it still delivers is the
					// CLAMPED impulse — never the raw brake number. Using w->brake
					// here meant exceeding the limit applied more force than the
					// limit allowed, so tightening the cap made braking harsher
					// instead of gentler: 16.6 g became 28.1 g.
					rollingFriction = ( demand < 0.0f ? -1.0f : 1.0f ) * maxImpulse;
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
			// Tyre grade times what it is standing on, times the load it carries —
			// the talk's "multiply tire friction by surface friction", then "multiply
			// this coefficient by the normal force to obtain the maximum friction
			// force". The surface half was missing, so every ground drove like every
			// other and a mud slope gripped exactly like tarmac.
			float maxImp = w->wheelSuspensionForce * v->stepDt * w->frictionSlip * w->surfaceFriction;
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
				w->isSliding = true;
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

/**
 * The rotation that takes the wheel's own frame onto the chassis: local X along
 * the suspension, local Z along the axle.
 *
 * Those are the axes Box3D's wheel joint reads (`wheel_joint.c:459` and `:472`),
 * so building the body in them lets frame B be the identity and keeps one
 * definition of "which way is this wheel pointing".
 */
static b3Quat WheelFrameRotation( b3Vec3 direction, b3Vec3 axle )
{
	b3Matrix3 m;
	m.cx = TryNormalize( direction, B3_LITERAL( b3Vec3 ){ 0.0f, -1.0f, 0.0f } );
	m.cz = TryNormalize( axle, B3_LITERAL( b3Vec3 ){ 1.0f, 0.0f, 0.0f } );

	// Re-orthogonalise: a mirrored or scaled build can leave the authored axle
	// and suspension direction a degree apart, and a non-orthogonal basis
	// becomes a quaternion that shears the wheel.
	m.cz = TryNormalize( Sub3( m.cz, Scale( m.cx, b3Dot( m.cx, m.cz ) ) ), m.cz );
	m.cy = b3Cross( m.cz, m.cx );

	return b3MakeQuatFromMatrix( &m );
}

static void ConfigureWheelJoint( Vehicle* v, Wheel* w );

/**
 * How far the wheel must move so its corner rests on the surface.
 *
 * The talk's own sequence, in its own terms, and it exists because of the faked
 * normals: "our solution is to reimplement wheel collision response ourselves.
 * Using the highest point from before we place the wheel so that it tangents the
 * point."
 *
 *   P1 = Center + Right - Up      CP1 = P1 - Center      S1 = CP1 . N
 *   P2 = Center - Right - Up      CP2 = P2 - Center      S2 = CP2 . N
 *   S3 = (CP1 - Contact Point) . N
 *
 * The corners come from the WHEEL's own axes, not from a support point derived
 * off the surface normal — a support point is the geometrically exact deepest
 * point of a cylinder and it is not what the talk does.
 *
 * One departure from the wording: the talk says the LARGER of S1 and S2 is the
 * deeper corner, but a corner further down the surface normal gives the SMALLER
 * dot product, so the smaller is taken. Reading it literally selects the
 * shallower corner, which contradicts "furthest penetrating".
 *
 * Returns the distance to move ALONG THE SUSPENSION, positive meaning the wheel
 * must come up; zero when the corner is already clear.
 */
static float PlacementAlongSuspension( const Wheel* w, const WheelContact* contact, b3Pos center )
{
	b3Vec3 n = contact->surfaceNormalWs;

	b3Vec3 up = Scale( w->wheelDirectionWs, -1.0f );
	b3Vec3 right = TryNormalize( w->wheelAxleWs, w->wheelAxleWs );

	b3Vec3 down = Scale( up, -w->radius );
	b3Vec3 half = Scale( right, w->width * 0.5f );

	b3Pos p1 = AddPosVec( center, Add3( down, half ) );
	b3Pos p2 = AddPosVec( center, Sub3( down, half ) );

	float s1 = b3Dot( n, SubPos( p1, center ) );
	float s2 = b3Dot( n, SubPos( p2, center ) );
	b3Pos deepest = s1 < s2 ? p1 : p2;

	float s3 = b3Dot( n, SubPos( deepest, contact->pointWs ) );
	if ( s3 >= 0.0f )
	{
		return 0.0f;
	}

	// "If we just separate along the contact normal the wheel will be pulled out
	// of alignment with the suspension, so the next step is projecting the
	// separation vector onto the suspension direction and moving along that
	// vector."
	float denom = b3Dot( n, up );
	if ( denom < 0.1f )
	{
		denom = 0.1f;
	}

	return -s3 / denom;
}

/**
 * Place every wheel so its corner tangents the surface, and record what the hard
 * point must answer for.
 *
 * "To avoid clipping the wheel well we limit the distance the wheel can travel
 * upwards though it never passes the hard point. To counteract the fact that you
 * would now see the wheel penetrating the ground we apply an upward force on the
 * hard point."
 *
 * A PLACEMENT, not a force: faking every normal to the suspension direction buys
 * momentum through bumps and costs correct contact, and this buys the contact
 * back. An impulse-based recovery is a first-order lag, and ground rising faster
 * than the lag can answer leaves the tyre standing in the floor.
 *
 * Called from `prepare`, BEFORE the world steps, and again after it: Box3D
 * copies body state into its solver sets for the duration of a step, so a
 * `SetTransform` from inside one is overwritten by the solver's integration.
 */
/**
 * How far the tyre's deepest corner is ABOVE the surface it found, along the
 * surface normal. Negative means it is in the ground.
 *
 * `isInContact` only says the cast found ground somewhere within the
 * suspension's reach, which is up to a whole rest length plus travel away. That
 * is the right question for the friction budget and the wrong one for "is the
 * tyre standing on anything", and the two were being answered by the same flag.
 */
static float SurfaceClearance( const Wheel* w, const WheelContact* contact, b3Pos center )
{
	b3Vec3 n = contact->surfaceNormalWs;

	b3Vec3 up = Scale( w->wheelDirectionWs, -1.0f );
	b3Vec3 right = TryNormalize( w->wheelAxleWs, w->wheelAxleWs );

	b3Pos p1 = AddPosVec( center, Add3( Scale( up, -w->radius ), Scale( right, w->width * 0.5f ) ) );
	b3Pos p2 = AddPosVec( center, Sub3( Scale( up, -w->radius ), Scale( right, w->width * 0.5f ) ) );

	float s1 = b3Dot( n, SubPos( p1, center ) );
	float s2 = b3Dot( n, SubPos( p2, center ) );
	b3Pos deepest = s1 < s2 ? p1 : p2;

	return b3Dot( n, SubPos( deepest, contact->pointWs ) );
}

static void PlaceWheels( Vehicle* v )
{
	b3WorldTransform chassisT = b3Body_GetTransform( v->chassis );

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		w->hardPointPenetration = 0.0f;


		if ( B3_IS_NULL( w->body ) )
		{
			continue;
		}

		b3Pos hardPoint = AddPosVec( chassisT.p, b3RotateVector( chassisT.q, w->chassisConnectionPointCs ) );

		if ( !w->info.isInContact )
		{
			/*
			 * A wheel touching nothing still belongs to the machine.
			 *
			 * This skipped it entirely, so an airborne wheel's body kept whatever
			 * pose it last had while the machine turned and pitched underneath
			 * it — and the joint, with nothing pressing on the wheel, is in no
			 * hurry to correct it. On a tank the corner wheels spend most of
			 * their time off the ground, so their casts and bodies sat at stale
			 * angles while the tracks they belong to moved.
			 *
			 * Only its ORIENTATION is taken back. Where it hangs is the joint's
			 * business and the suspension's: moving it here as well put a parked
			 * car's tyre 234 mm above the ground, because a cast that misses for
			 * one tick would then yank the wheel out to full droop.
			 */
			b3Quat frame = b3MulQuat( chassisT.q, WheelFrameRotation( w->directionCs, w->axleCs ) );
			b3Body_SetTransform( w->body, b3Body_GetPosition( w->body ), frame );
			continue;
		}
		b3Pos now = b3Body_GetPosition( w->body );

		// Every contact this tyre found, not just the nearest: it has to clear the
		// deepest of them, which is what "the highest point" means.
		float needed = 0.0f;
		for ( int c = 0; c < w->info.contactCount; c++ )
		{
			float move = PlacementAlongSuspension( w, &w->info.contacts[c], now );
			needed = move > needed ? move : needed;
		}

		b3Vec3 up = Scale( w->wheelDirectionWs, -1.0f );
		float travel = b3Dot( w->wheelDirectionWs, SubPos( now, hardPoint ) );

		/*
		 * A tyre HANGING above the surface is placed down onto it, the same way one
		 * standing in it is placed out of it.
		 *
		 * The placement only ever lifted, so a wheel that ended a tick clear of the
		 * ground stayed clear — and `isInContact` is true for anything the cast can
		 * reach, which is a whole rest length plus travel away. The suspension then
		 * pushed the chassis up off a tyre touching nothing and the machine
		 * levitated: 105 mm on the game's own Car, parked, and it sagged back only
		 * because the damper was reading its own placement sawtooth. A tyre is
		 * placed where the ground is, in both directions.
		 *
		 * Down is bounded by the REST length, which is as far as a spring extends:
		 * travel past it is compression room, not reach. Bounding by rest plus
		 * travel instead let a wheel a builder mounted high — a tank's raised
		 * idlers, which are meant never to touch flat ground — be dragged a whole
		 * extra travel down onto the floor.
		 */
		if ( needed <= 0.0f )
		{
			float droop = w->suspensionRestLength - travel;
			if ( droop <= 0.0f )
			{
				continue;
			}

			float clearance = SurfaceClearance( w, &w->info.contacts[0], now );
			for ( int c = 1; c < w->info.contactCount && c < B3W_MAX_CONTACTS; c++ )
			{
				float other = SurfaceClearance( w, &w->info.contacts[c], now );
				clearance = other < clearance ? other : clearance;
			}

			if ( clearance <= TYRE_TOUCH_CLEARANCE )
			{
				continue;
			}

			float down = clearance < droop ? clearance : droop;
			b3Body_SetTransform( w->body, AddPosVec( now, Scale( up, -down ) ), b3Body_GetRotation( w->body ) );
			continue;
		}

		// Never past the hard point.
		float allowed = travel > 0.0f ? travel : 0.0f;
		float move = needed < allowed ? needed : allowed;

		if ( move > 0.0f )
		{
			b3Body_SetTransform( w->body, AddPosVec( now, Scale( up, move ) ), b3Body_GetRotation( w->body ) );
		}

		w->hardPointPenetration = needed - move;
		w->placementNeeded = needed;
	}
}

/**
 * The upward force on the hard point.
 *
 * "It nets us two benefits: first it will get the 4x4 out of collision; secondly
 * when you slam it to the ground hard it creates a nice jerk that the player can
 * feel."
 *
 * It answers only what the placement could not, because the wheel had reached
 * its hard point. Solved as a FORCE proportional to the leftover, not a velocity
 * target: as a target it drove the mount to 3 m/s upward every substep on four
 * wheels at once and threw the car onto its roof.
 *
 * The closing speed is measured along the SURFACE NORMAL, not along the
 * suspension: a machine climbing a 30 degree slope at 8 m/s rises at 4 m/s in
 * its own up direction purely by going uphill, and tested against the suspension
 * axis this force concluded the chassis was already leaving and never fired.
 */
static void ApplyHardPointForce( Vehicle* v, Wheel* w, float h )
{
	if ( w->hardPointPenetration <= 0.0f || w->info.contactCount == 0 )
	{
		return;
	}

	b3Vec3 up = Scale( w->wheelDirectionWs, -1.0f );
	b3Pos hardPoint = w->info.hardPointWs;

	b3Vec3 normal = w->info.contacts[0].surfaceNormalWs;
	float relVel = b3Dot( normal, b3Body_GetWorldPointVelocity( v->chassis, hardPoint ) );

	// The speed that would clear the leftover in one substep, capped — then read
	// as the force it takes to reach it. As an uncapped target this threw the car
	// onto its roof; the cap and the ceiling below are what make it a jerk the
	// player feels rather than a launch.
	float target = w->hardPointPenetration * SafeInv( v->stepDt );
	if ( target > MAX_RECOVERY_SPEED )
	{
		target = MAX_RECOVERY_SPEED;
	}

	float chassisMass = b3Body_GetMass( v->chassis );
	float force = chassisMass * ( target - relVel ) * SafeInv( h );
	if ( force <= 0.0f )
	{
		return;
	}

	float ceiling = w->maxSuspensionForce * HARD_POINT_FORCE_SCALE;
	if ( force > ceiling )
	{
		force = ceiling;
	}

	w->hardPointForce = force;
	b3Body_ApplyLinearImpulse( v->chassis, Scale( up, force * h ), hardPoint, false );
}

/**
 * The suspension, as the game's own spring and damper between wheel and chassis.
 *
 * "We use Havok as a solver for these constraints but provide many of our own
 * inputs." The joint holds the wheel on its axis and stops it passing the hard
 * point; the force along that axis is ours, and it is the model the game already
 * has — stiffness authored as ride-height SAG, damping as a fraction of critical
 * and asymmetric between compression and rebound, which a joint's single damping
 * ratio cannot express.
 */
/** Whether the tyre is actually standing on the surface it found, rather than
 *  hanging somewhere above it with the cast still able to see it. */
static bool IsTyreOnSurface( const Wheel* w )
{
	if ( !w->info.isInContact || w->info.contactCount == 0 || B3_IS_NULL( w->body ) )
	{
		return false;
	}

	b3Pos center = b3Body_GetPosition( w->body );
	float best = SurfaceClearance( w, &w->info.contacts[0], center );

	for ( int c = 1; c < w->info.contactCount && c < B3W_MAX_CONTACTS; c++ )
	{
		float clearance = SurfaceClearance( w, &w->info.contacts[c], center );
		best = clearance < best ? clearance : best;
	}

	return best <= TYRE_TOUCH_CLEARANCE;
}

static void ApplySuspensionForce( Vehicle* v, Wheel* w, float chassisMass, float h )
{
	if ( B3_IS_NULL( w->body ) )
	{
		w->springForce = 0.0f;
		return;
	}

	b3WorldTransform chassisT = b3Body_GetTransform( v->chassis );
	b3Pos hardPoint = AddPosVec( chassisT.p, b3RotateVector( chassisT.q, w->chassisConnectionPointCs ) );
	b3Vec3 up = Scale( w->wheelDirectionWs, -1.0f );

	b3Pos wheelPos = b3Body_GetPosition( w->body );
	float travel = b3Dot( w->wheelDirectionWs, SubPos( wheelPos, hardPoint ) );
	float compression = w->suspensionRestLength - travel;

	/*
	 * How fast the suspension is compressing — measured against the GROUND while
	 * the tyre is on it, and against the wheel only while it is not.
	 *
	 * A grounded tyre is placed, held by a constraint and stood up by a contact
	 * solve, and its body's velocity is an artefact of those three rather than a
	 * fact about the suspension: between placements it falls, and the fall is
	 * undone by a teleport that carries no velocity. The damper read that
	 * sawtooth as real extension — 0.36 m/s on dead flat ground — and cancelled
	 * 500 N of an 817 N spring, so the machine sagged the faster it went and by
	 * different amounts in each direction. It is what the placement-rate term
	 * was added to cancel, and that term never reached a substep: the placement
	 * runs after the world steps and `prepare` zeroed the rate before the next
	 * one.
	 *
	 * While the tyre is on the ground the suspension closes at exactly the speed
	 * the mount approaches the contact, and the mount is the chassis, which is
	 * simulated honestly. In the air there is no contact to measure against and
	 * the wheel's own motion IS the suspension's.
	 */
	b3Vec3 mountVel = b3Body_GetWorldPointVelocity( v->chassis, hardPoint );
	float compressionRate = b3Dot( w->wheelDirectionWs, mountVel );

	bool onSurface = IsTyreOnSurface( w );

	if ( !onSurface )
	{
		b3Vec3 wheelVel = b3Body_GetLinearVelocity( w->body );
		compressionRate = -b3Dot( w->wheelDirectionWs, Sub3( wheelVel, mountVel ) );
	}

	float damping = compressionRate > 0.0f ? w->dampingCompression : w->dampingRelaxation;
	float implicit = 1.0f / ( 1.0f + h * damping );

	float force = chassisMass * ( w->suspensionStiffness * compression + damping * compressionRate * implicit );

	// A suspension pushes apart and never pulls together, and it has a ceiling.
	if ( force <= 0.0f )
	{
		w->springForce = 0.0f;
		return;
	}
	if ( force > w->maxSuspensionForce )
	{
		force = w->maxSuspensionForce;
	}

	w->springForce = force;
	w->springRate = compressionRate;
	w->springCompression = compression;
	w->springSpringTerm = chassisMass * w->suspensionStiffness * compression;

	b3Vec3 impulse = Scale( up, force * h );
	b3Body_ApplyLinearImpulse( v->chassis, impulse, hardPoint, false );

	/*
	 * The chassis only. The reaction does NOT go on the wheel.
	 *
	 * A tyre is placed on the surface, not simulated into it — the talk holds it
	 * there with a constraint — so pushing it down with the chassis-scaled
	 * suspension force only gives the placement something to undo. It undid it
	 * every tick: a PARKED 1763 kg car reported a 0.53 m/s suspension velocity
	 * and a 16 mm-per-tick placement, and the damper, reading that phantom
	 * motion, cancelled 7.3 kN of a 9.6 kN spring. The car then rested on its
	 * bump stop with no travel left, which is why a heavy machine sank into the
	 * terrain and a light one did not.
	 *
	 * The reaction is not discarded, it is passed THROUGH the tyre to whatever
	 * the tyre is standing on. Static terrain absorbs it and there is nothing to
	 * apply; another machine must feel the weight, or a car parked on a platform
	 * would be weightless to it.
	 */
	b3Vec3 reaction = Scale( impulse, -1.0f );

	if ( !w->info.isInContact )
	{
		// In the air there is nothing under the tyre to take it, and the spring's
		// whole job is to push the wheel DOWN into droop. Skipping it here left a
		// leaping machine with its wheels tucked under it.
		b3Body_ApplyLinearImpulse( w->body, reaction, wheelPos, false );
	}
	else if ( B3_IS_NON_NULL( w->info.groundBodyId ) && b3Body_IsValid( w->info.groundBodyId ) &&
			  b3Body_GetType( w->info.groundBodyId ) == b3_dynamicBody )
	{
		b3Body_ApplyLinearImpulse( w->info.groundBodyId, reaction, w->info.contactPointWs, false );
	}
}

/**
 * The point constraint, solved for one wheel against everything it is standing
 * on this substep.
 *
 * The rows are unilateral — the ground pushes a tyre up and never pulls it down
 * — and they are solved SIMULTANEOUSLY rather than one after another, because a
 * tyre spanning two planks shares one body between them and iterating would let
 * each row undo the last. That is the machinery the bump stop already used for
 * bottomed wheels; it is general now, which is what makes multiple contact
 * points do what the talk takes them for: "the tire sits on multiple planks"
 * instead of ping-ponging between them.
 *
 * Each row is written the way Box3D writes its own contacts
 * (`contact_solver.c:444`): while the tyre is clear of the surface the bias is
 * speculative, permitting an approach of at most the remaining gap in one
 * substep, so the wheel cannot pass through however fast the ground rises; once
 * it is buried the bias is soft, with `massScale`/`impulseScale` taking the
 * correction's energy back out instead of leaving it in the body as a launch.
 */
static void SolveWheelContacts( Vehicle* v, Wheel* w, float h )
{
	int count = w->info.contactCount;
	if ( count > B3W_MAX_CONTACTS )
	{
		count = B3W_MAX_CONTACTS;
	}

	// Box3D's own contact softness, so a tyre resting on a surface behaves like
	// everything else resting on a surface. See CONTACT_DAMPING_RATIO.
	float contactHertz = CONTACT_HERTZ < 0.125f * SafeInv( h ) ? CONTACT_HERTZ : 0.125f * SafeInv( h );
	float omega = 2.0f * 3.14159265358979f * contactHertz;
	float a1 = 2.0f * CONTACT_DAMPING_RATIO + h * omega;
	float a2 = h * omega * a1;
	float a3 = 1.0f / ( 1.0f + a2 );
	float softBiasRate = omega / a1;
	float softMassScale = a2 * a3;
	float softImpulseScale = a3;

	b3Pos com = CenterOfMass( w->body );
	float invMass = InvMass( w->body );
	b3Matrix3 invInertia = WorldInvInertia( w->body );

	WheelContact* rows[B3W_MAX_CONTACTS];
	float rowVel[B3W_MAX_CONTACTS];
	float rowImpulseScale[B3W_MAX_CONTACTS];
	int n = 0;

	b3Pos center = b3Body_GetPosition( w->body );

	for ( int c = 0; c < count; c++ )
	{
		WheelContact* contact = &w->info.contacts[c];

		// Against the REAL surface, by the same P1/P2/S3 the placement uses — not
		// against the faked plane, which reads clear while the tyre's lower edge
		// is buried. Positive is clear of the surface.
		float separation = -PlacementAlongSuspension( w, contact, center );

		b3Vec3 vel = Sub3( b3Body_GetWorldPointVelocity( w->body, contact->pointWs ),
						   GroundVelocityAt( w->info.groundBodyId, contact->pointWs ) );
		float relVel = b3Dot( contact->normalWs, vel );

		float bias;
		float massScale;
		float impulseScale;
		if ( separation > 0.0f )
		{
			bias = separation * SafeInv( h );
			massScale = 1.0f;
			impulseScale = 0.0f;
		}
		else
		{
			float soft = softMassScale * softBiasRate * separation;
			bias = soft < -MAX_RECOVERY_SPEED ? -MAX_RECOVERY_SPEED : soft;
			massScale = softMassScale;
			impulseScale = softImpulseScale;
		}

		float velocity = massScale * relVel + bias;
		if ( velocity >= 0.0f && contact->normalImpulse == 0.0f )
		{
			continue;
		}

		rows[n] = contact;
		rowVel[n] = velocity;
		rowImpulseScale[n] = impulseScale;
		n++;
	}

	if ( n == 0 )
	{
		w->wheelSuspensionForce = 0.0f;
		return;
	}

	float a[B3W_MAX_CONTACTS * B3W_MAX_CONTACTS];
	float b[B3W_MAX_CONTACTS];
	float lambda[B3W_MAX_CONTACTS];

	while ( n > 0 )
	{
		for ( int i = 0; i < n; i++ )
		{
			b3Vec3 ri = SubPos( rows[i]->pointWs, com );
			b3Vec3 rxni = b3Cross( ri, rows[i]->normalWs );
			b3Vec3 irxni = b3MulMV( invInertia, rxni );
			b[i] = -rowVel[i];

			for ( int j = 0; j < n; j++ )
			{
				b3Vec3 rj = SubPos( rows[j]->pointWs, com );
				b3Vec3 rxnj = b3Cross( rj, rows[j]->normalWs );
				a[i * n + j] = invMass * b3Dot( rows[i]->normalWs, rows[j]->normalWs ) + b3Dot( irxni, rxnj );
			}
		}

		if ( !SolveDense( a, b, n, lambda ) )
		{
			// Singular: two contacts describing the same point. Fall back to one
			// row at a time rather than dropping the wheel's support entirely.
			for ( int i = 0; i < n; i++ )
			{
				b3Vec3 ri = SubPos( rows[i]->pointWs, com );
				b3Vec3 rxni = b3Cross( ri, rows[i]->normalWs );
				float denom = invMass + b3Dot( b3MulMV( invInertia, rxni ), rxni );
				lambda[i] = -rowVel[i] * SafeInv( denom > 1.0e-6f ? denom : 1.0e-6f );
			}
			break;
		}

		int worst = -1;
		float worstVal = 0.0f;
		for ( int i = 0; i < n; i++ )
		{
			float total = rows[i]->normalImpulse + lambda[i] - rowImpulseScale[i] * rows[i]->normalImpulse;
			if ( total < 0.0f && ( worst < 0 || total < worstVal ) )
			{
				worst = i;
				worstVal = total;
			}
		}

		if ( worst < 0 )
		{
			break;
		}

		// This row would have to pull the tyre down. Drop it and solve what is
		// left; its accumulated impulse goes with it.
		rows[worst]->normalImpulse = 0.0f;
		for ( int k = worst; k < n - 1; k++ )
		{
			rows[k] = rows[k + 1];
			rowVel[k] = rowVel[k + 1];
			rowImpulseScale[k] = rowImpulseScale[k + 1];
		}
		n--;
	}

	float total = 0.0f;
	for ( int i = 0; i < n; i++ )
	{
		float delta = lambda[i] - rowImpulseScale[i] * rows[i]->normalImpulse;
		float accumulated = rows[i]->normalImpulse + delta;
		if ( accumulated < 0.0f )
		{
			accumulated = 0.0f;
		}

		float applied = accumulated - rows[i]->normalImpulse;
		rows[i]->normalImpulse = accumulated;
		// The impulse accumulates over the TICK's substeps, so the force it stands
		// for is that total spread over the tick — divided by `h` it reads 32 times
		// too large, and this is the number the talk hands to friction.
		rows[i]->force = accumulated * SafeInv( v->stepDt );
		total += rows[i]->force;

		if ( applied != 0.0f )
		{
			b3Body_ApplyLinearImpulse( w->body, Scale( rows[i]->normalWs, applied ), rows[i]->pointWs, false );
		}
	}

	// The upward force on the hard point, on what the placement could not absorb.
	ApplyHardPointForce( v, w, h );

	/*
	 * THE normal force: what the tyre presses on the ground with.
	 *
	 * The talk reads it straight out of the constraint, because in their build the
	 * constraint carries the machine — the suspension pushes the chassis up and
	 * the wheel down, and the ground answers the pair. Here the suspension's
	 * reaction is deliberately NOT put on the wheel (a placed tyre would only have
	 * to undo it), so this constraint holds up the WHEEL and nothing else, and on
	 * a car with 1 kg wheels it read 10 N. Friction is a coefficient times this
	 * number: an 800 kg car had the grip of a shopping trolley, which is why it
	 * slid 314 m down a 10 degree slope parked and delivered a fifth of the
	 * acceleration its engine force commanded.
	 *
	 * The load is on the other side of the same equation: what the suspension
	 * pushed the chassis up with is what it pushed the tyre down with, plus the
	 * tyre's own weight.
	 *
	 * NOT the contact solve's own total. That number is whatever it took to stop
	 * the wheel this substep, so it spikes on every bump — and friction is a
	 * coefficient times it, so grip spiked with it. On a light machine carrying a
	 * heavy wheel the spikes were larger than anything the machine weighs: a
	 * one-block chassis with a single wheel left at 292 m/s. Load is a steady
	 * quantity and this is its steady form.
	 */
	float gravity = b3Length( b3World_GetGravity( v->world ) );
	w->wheelSuspensionForce = w->springForce + b3Body_GetMass( w->body ) * gravity;
}


static void ApplyWheelMass( Vehicle* v, Wheel* w )
{
	if ( B3_IS_NULL( w->body ) )
	{
		return;
	}

	// A dynamic body with no shapes has NO MASS until it is given one, and a
	// zero-mass body makes every effective mass in the contact solve zero: the
	// matrix goes singular, the fallback divides by its own epsilon floor, and
	// the machine leaves for infinity. The floor is a solver guard, not a
	// physical claim — the game sets the real mass in the same breath as
	// creating the wheel.
	float mass = w->mass > 0.0f ? w->mass : 0.01f;

	float r = w->radius;
	float halfWidth = 0.5f * w->width;

	b3MassData md;
	md.mass = mass;
	md.center = B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.0f };
	md.inertia = b3Mat3_zero;
	// A disc about its axle (local Z), and across it on the other two.
	float across = 0.25f * mass * r * r + ( 1.0f / 3.0f ) * mass * halfWidth * halfWidth;
	float spin = 0.5f * mass * r * r;

	// Floored for the same reason the mass is, and it matters more: a body with
	// mass but no inertia is only singular once something applies a torque to it,
	// which for a free wheel is never and for a JOINTED one is every step.
	float floorInertia = 1.0e-4f * mass;
	across = across > floorInertia ? across : floorInertia;
	spin = spin > floorInertia ? spin : floorInertia;

	md.inertia.cx.x = across;
	md.inertia.cy.y = across;
	md.inertia.cz.z = spin;
	b3Body_SetMassData( w->body, md );

	// The spin integrator upstream reads this; one definition of the tyre's
	// inertia now, not two.
	w->inertia = md.inertia.cz.z;

	if ( w->mass <= 0.0f )
	{
		// Nothing authored yet: the guard mass is not the machine's, so it is not
		// taken off the chassis either.
		return;
	}

	/*
	 * ONCE per wheel, and never more than the wheels' share of the machine.
	 *
	 * A wheel becoming a body must not make the machine heavier, so its mass
	 * comes off the chassis. Two things were wrong with doing that here.
	 *
	 * It was not idempotent: this runs again every time the game sets a wheel's
	 * mass, and each call subtracted afresh. And it was unbounded: a tank's
	 * twelve tracked wheels want more than the whole machine, so the chassis
	 * ended at the 0.1 kg floor — a paper hull carrying twelve heavy wheels on
	 * joints, suspensions slammed to their stops, wheels pushing seven times the
	 * machine's weight.
	 *
	 * Removing the transfer entirely was tried and is worse: the machine then
	 * weighs its wheels twice and the Car stops turning.
	 *
	 * `massTakenFromChassis` is what this wheel has already moved, so a repeat
	 * call moves only the difference. The share is half the machine between all
	 * the wheels, because a chassis lighter than the wheels hanging off it is not
	 * a vehicle any solver can hold together.
	 */
	int n = v->wheelCount > 0 ? v->wheelCount : 1;
	b3MassData chassis = b3Body_GetMassData( v->chassis );

	float whole = chassis.mass + w->massTakenFromChassis;
	float budget = whole / (float)( 2 * n );
	float want = w->mass < budget ? w->mass : budget;

	float remaining = whole - want;
	if ( remaining < 0.1f )
	{
		remaining = 0.1f;
	}

	w->massTakenFromChassis = want;
	chassis.mass = remaining;
	b3Body_SetMassData( v->chassis, chassis );
	v->chassisMassAsLeft = remaining;

	// The spring is specified against these masses, so it is stale the moment
	// they change.
	ConfigureWheelJoint( v, w );
}

/**
 * The chassis was given the WHOLE machine's mass by something else: take the
 * wheels' share off it again.
 *
 * The wheels' share comes off the chassis here, behind the host's back, and the
 * host has its own idea of what the chassis weighs — every part of the machine,
 * wheels included — which it puts back whenever a collider is attached or
 * removed, because Box3D re-derives a body's mass from its shapes at each. So
 * the first collider change after the wheels were built handed their share back
 * to the chassis for good: a car whose colliders land a few frames after it is
 * built weighed 40 kg more than the same car built at once, and two clients
 * holding the same machine disagreed about its mass.
 *
 * A chassis mass this vehicle did not leave can only be such a whole, so the
 * takes start over from it.
 */
static void RetakeWheelMass( Vehicle* v )
{
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		v->wheels[i].massTakenFromChassis = 0.0f;
		ApplyWheelMass( v, &v->wheels[i] );
	}
	v->chassisMassAsLeft = b3Body_GetMass( v->chassis );
}

/**
 * Push the travel limits into the joint.
 *
 * Only the limits. The spring is ours — see `ApplySuspensionForce` — so none of
 * the game's rates have to be re-expressed in a form a joint understands. The
 * lower limit IS the talk's hard point: "we limit the distance the wheel can
 * travel upwards though it never passes the hard point", as the constraint it is
 * rather than as a clamp applied afterwards.
 */
static void ConfigureWheelJoint( Vehicle* v, Wheel* w )
{
	(void)v;

	if ( B3_IS_NULL( w->joint ) )
	{
		return;
	}

	b3WheelJoint_SetSuspensionLimits( w->joint, -w->suspensionRestLength, w->maxSuspensionTravel );
}

static void CreateWheelBody( Vehicle* v, Wheel* w )
{
	b3WorldTransform chassisT = b3Body_GetTransform( v->chassis );
	w->info.hardPointWs = AddPosVec( chassisT.p, b3RotateVector( chassisT.q, w->chassisConnectionPointCs ) );
	w->wheelDirectionWs = b3RotateVector( chassisT.q, w->directionCs );
	w->wheelAxleWs = b3RotateVector( chassisT.q, w->axleCs );
	w->center = AddPosVec( w->info.hardPointWs, Scale( w->wheelDirectionWs, w->info.suspensionLength ) );

	b3Quat frame = WheelFrameRotation( w->directionCs, w->axleCs );

	b3BodyDef def = b3DefaultBodyDef();
	def.type = b3_dynamicBody;
	def.position = w->center;
	def.rotation = b3MulQuat( chassisT.q, frame );
	w->body = b3CreateBody( v->world, &def );

	// A tyre-shaped collider with its collision OFF, which is what the talk
	// describes — "the wheels are rigid bodies with their collision disabled" —
	// and not the same thing as no collider at all. A shapeless dynamic body has
	// no extent, so the solver cannot decide whether it moved far enough this
	// step to need a swept test, and a joint into one corrupted the continuous
	// pass badly enough to trap.
	b3ShapeDef sd = b3DefaultShapeDef();
	sd.enableContactEvents = false;
	sd.density = 0.0f;
	sd.filter.categoryBits = 0;
	sd.filter.maskBits = 0;

	b3Capsule tyre;
	tyre.center1 = B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, -0.5f * w->width };
	tyre.center2 = B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.5f * w->width };
	// A hair under the tread. This shape exists to give the body an EXTENT — a
	// shapeless one cannot tell the solver whether it moved far enough to need a
	// swept test — and it collides with nothing, so its exact size is free;
	// keeping it inside the tyre keeps it clear of everything the placement puts
	// the wheel against.
	tyre.radius = w->radius - WHEEL_SHAPE_CLEARANCE;
	if ( tyre.radius < 0.02f )
	{
		tyre.radius = 0.02f;
	}
	b3CreateCapsuleShape( w->body, &sd, &tyre );

	/*
	 * The bump stop: a small collider on the CHASSIS, at the hard point.
	 *
	 * It cannot live on the wheel. The placement teleports that body onto the
	 * surface every tick, and a teleport that lands a collider inside something
	 * is a depenetration impulse — measured at 293 KILOMETRES of altitude driving
	 * into a Rocket, whatever size the collider was. The chassis is never
	 * teleported, so a shape hung there is safe.
	 *
	 * Erwin Coumans' answer to this symptom: "add a small cylinder shape for each
	 * wheel in the bottomed out position to the car rigid body, so the physics
	 * takes over once the ground is hit too hard — make it smaller than the
	 * actual wheel so it does not hit the ground during normal driving." It sits
	 * at the hard point, under the tread, so it meets nothing until either the
	 * suspension is out of travel or something is genuinely inside the wheel —
	 * which is what driving at another blueprint wheels-first does.
	 */
	b3ShapeDef stopDef = b3DefaultShapeDef();
	stopDef.density = 0.0f;
	stopDef.enableContactEvents = false;

	// The machine's own filter, so it collides with exactly what the chassis
	// collides with and never with the machine itself — see `MachineFilter`.
	b3Filter machine;
	if ( MachineFilter( v, &machine ) )
	{
		stopDef.filter = machine;
	}

	// Held out of the world until a cast says it is clear of it — see
	// `stopArmed`. Collision is off both ways: a zero mask cannot match anything,
	// and a zero group cannot override that.
	w->stopFilter = stopDef.filter;
	w->stopArmed = false;
	stopDef.filter.maskBits = 0;
	stopDef.filter.groupIndex = 0;

	b3Sphere stop;
	stop.center = w->chassisConnectionPointCs;
	stop.radius = w->radius - BUMP_STOP_CLEARANCE;
	if ( stop.radius > 0.0f )
	{
		/*
		 * Adding a shape RE-DERIVES the body's mass from its shapes, and the
		 * game's chassis mass is authored, not derived — so creating this sphere
		 * silently replaced it. What was left after `ApplyWheelMass` took the
		 * wheels off it was the 0.1 kg floor, and a machine that weighs 0.1 kg
		 * answers its first real impact with a velocity of 1e13 and then NaN.
		 *
		 * So the mass is put back exactly as it was. The stop is a collider, not
		 * a part of the machine, and it has no business changing what the machine
		 * weighs.
		 */
		b3MassData before = b3Body_GetMassData( v->chassis );
		w->stopShape = b3CreateSphereShape( v->chassis, &stopDef, &stop );
		b3Body_SetMassData( v->chassis, before );
	}

	ApplyWheelMass( v, w );

	/*
	 * A wheel body does not get a vote on whether its machine is at rest.
	 *
	 * Box3D sleeps an island only when EVERY body in it is sleepy —
	 * "any single body in an island can keep it awake", `solver.c:813` — and it
	 * judges sleepiness on a body's own velocity. That question is meaningless
	 * here: this body is a REPRESENTATION. Its pose is authored every tick by the
	 * placement, its collision is off, and the velocity it carries is whatever it
	 * gained falling between two placements. Measured on the Car parked on flat
	 * ground: the chassis reports a sleep velocity of 0.0000 and does not move a
	 * tenth of a millimetre, while its four wheels report 3.8 to 6.4 against a
	 * threshold of 0.4 — so no wheeled machine in the world could ever sleep, for
	 * ever, which is the tick budget of every parked car on a server.
	 *
	 * A threshold nothing can exceed makes the wheel permanently sleepy, so the
	 * island sleeps exactly when the machine does: on the chassis, which is
	 * simulated honestly and which is what "parked" means. Nothing else changes —
	 * no force, no constraint, no wake path. A machine standing still with its
	 * wheels driven still wakes, because `prepare` wakes the chassis on engine
	 * force before it reads this at all.
	 */
	b3Body_SetSleepThreshold( w->body, FLT_MAX );

	b3WheelJointDef jd = b3DefaultWheelJointDef();
	jd.base.bodyIdA = v->chassis;
	jd.base.bodyIdB = w->body;
	// The frame sits at the wheel's REST position, not at the hard point: the
	// joint's spring drives translation toward zero and has no target of its own,
	// so anchoring at the hard point would mean the spring's happy place is the
	// wheel fully retracted into its mount. Zero is rest; negative is compression
	// toward the hard point; positive is droop.
	jd.base.localFrameA.p = Add3( w->chassisConnectionPointCs, Scale( w->directionCs, w->suspensionRestLength ) );
	jd.base.localFrameA.q = frame;
	jd.base.localFrameB.p = B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.0f };
	jd.base.localFrameB.q = b3Quat_identity;

	// Travel runs from the hard point (0) to full droop, and the wheel may never
	// pass the hard point — the talk's step 9, expressed as the limit it is
	// rather than as a clamp applied after the fact.
	// The wheel may never pass the hard point — the talk's step 9, as the limit
	// it is rather than a clamp applied afterwards — and may droop no further
	// than its travel.
	jd.enableSuspensionLimit = true;
	jd.lowerSuspensionLimit = -w->suspensionRestLength;
	jd.upperSuspensionLimit = w->maxSuspensionTravel;
	// The joint carries the KINEMATICS only. Its spring is specified as a
	// frequency over the REDUCED mass of a tyre against a chassis, and it takes a
	// single damping ratio where this game authors compression and rebound
	// separately — so the suspension is applied as a force instead.
	jd.enableSuspensionSpring = false;

	// Steering is the JOINT's, so the body turns with it. Left disabled, the
	// controller steered only the axle it hands to the cast and to friction, and
	// the debug view showed the rings square to the chassis while the machine
	// steered.
	jd.enableSteering = true;
	jd.steeringHertz = STEERING_HERTZ;
	jd.steeringDampingRatio = STEERING_DAMPING_RATIO;
	jd.maxSteeringTorque = FLT_MAX;
	jd.enableSpinMotor = false;

	w->joint = b3CreateWheelJoint( v->world, &jd );
	ConfigureWheelJoint( v, w );
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

}

/**
 * Is this vehicle still something we may drive?
 *
 * The chassis belongs to the CALLER, not to us, and the game's teardown destroys
 * every body before it frees the controllers — so between those two moments a
 * tick can arrive for a vehicle whose chassis is already gone. Body ids are
 * recycled, so a stale one is not null and not obviously wrong: it either reads
 * another machine's body or trips the validity check deep inside the solver,
 * which is where this surfaced.
 */
static bool IsVehicleLive( const Vehicle* v )
{
	return v->used && !B3_IS_NULL( v->chassis ) && b3Body_IsValid( v->chassis );
}

/**
 * Forget every handle that is no longer ours.
 *
 * The substep solve only checked for null, and a destroyed body's id is not
 * null. Reading one drives a stale index into the body array, and the garbage
 * that comes back reaches the continuous solver as a shape — which then calls
 * through a function pointer that was never a function.
 *
 * Two kinds go stale: the wheel body and its joint, if the caller tore the
 * machine down; and the body the wheel was resting ON, which belongs to someone
 * else entirely. That second one bites hardest for a SLEEPING machine, because
 * `prepare` arms the vehicle and returns before casting, so the substeps go on
 * pushing against a contact found before the thing under it was deleted.
 */
static void DropDeadHandles( Vehicle* v )
{
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		if ( !B3_IS_NULL( w->body ) && !b3Body_IsValid( w->body ) )
		{
			w->body = b3_nullBodyId;
			w->joint = b3_nullJointId;
		}

		if ( B3_IS_NON_NULL( w->info.groundBodyId ) && !b3Body_IsValid( w->info.groundBodyId ) )
		{
			w->info.groundBodyId = b3_nullBodyId;
			w->info.groundBody = -1;
			w->info.isInContact = false;
			w->info.contactCount = 0;
		}
	}
}


/**
 * Everything that must happen ONCE per full step: where the wheels are, how
 * fast the chassis is going, and the shape casts that find the ground.
 *
 * Casting is the expensive half and its answer is good for the whole step —
 * the chassis moves millimetres in one tick. Splitting it from the solve is
 * what makes solving 32 times affordable.
 */
/**
 * Where a wheel's centre is NOW, from the chassis as it stands: the hard point,
 * then down the suspension by its current length. (`Wheel::center` is set once,
 * when the wheel body is made, and judged from it a tyre measured its contacts
 * from where the wheel started.)
 */
static b3Pos WheelCentreNow( const Vehicle* v, const Wheel* w, b3Vec3* down )
{
	b3WorldTransform t = b3Body_GetTransform( v->chassis );
	*down = b3RotateVector( t.q, w->directionCs );
	b3Pos hard = AddPosVec( t.p, b3RotateVector( t.q, w->chassisConnectionPointCs ) );
	return AddPosVec( hard, Scale( *down, w->info.suspensionLength ) );
}

/**
 * The collision filter of the MACHINE: any chassis shape that is not one of this
 * vehicle's own bump stops or tyres.
 *
 * Box3D lists a body's shapes newest first, so "the chassis's first shape" was
 * whatever the vehicle added last — for every wheel after the first, the
 * previous wheel's bump stop, still held out of the world with a zero mask. So
 * every bump stop after the first copied a filter that matches nothing and
 * could never stop anything, and the first tyre shape did the same.
 */
static bool MachineFilter( const Vehicle* v, b3Filter* filter )
{
	int capacity = b3Body_GetShapeCount( v->chassis );
	b3ShapeId stack[64];
	b3ShapeId* shapes = capacity <= 64 ? stack : (b3ShapeId*)malloc( sizeof( b3ShapeId ) * (size_t)capacity );
	int count = b3Body_GetShapes( v->chassis, shapes, capacity );
	bool found = false;
	for ( int i = 0; i < count && !found; i++ )
	{
		bool ours = false;
		for ( int k = 0; k < v->wheelCount && !ours; k++ )
		{
			ours = B3_ID_EQUALS( shapes[i], v->wheels[k].stopShape ) || B3_ID_EQUALS( shapes[i], v->wheels[k].tyreShape );
		}
		if ( !ours )
		{
			*filter = b3Shape_GetFilter( shapes[i] );
			found = true;
		}
	}
	if ( shapes != stack )
	{
		free( shapes );
	}
	return found;
}

/**
 * Keep a wheel's tyre shape matching the wheel, and its tread turning with it.
 *
 * The shape is the tyre (less the clearance) swept along the suspension from
 * fully compressed down to LOADED — the sag the spring is rated for, `g / (n k)`, since a spring
 * here carries `chassisMass * k * compression` and the machine's weight shares
 * across its n wheels. Rebuilt only when that changes: re-shaping drops the
 * shape's contacts.
 *
 * Its contacts act at the hard point, which is where the suspension pushes the
 * chassis, and its surface turns at a tenth of the wheel's spin about the axle.
 */
static void UpdateTyre( Vehicle* v, Wheel* w, int index )
{
	float r = w->radius;
	if ( r <= TYRE_CLEARANCE || w->width <= 0.0f )
	{
		return;
	}

	// Only a wheel that CARRIES the machine — its suspension within 45 degrees of
	// the vehicle's down — is a wheel in PhysX's sense. One mounted sideways lies
	// on the ground on its face; a tyre there held the chassis up while the
	// wheel's own constraint pushed off the same ground, two supports fighting,
	// and a light machine never came to rest.
	b3Vec3 vehicleUp = AxisVector( v->indexUpAxis );
	if ( -b3Dot( b3Normalize( w->directionCs ), vehicleUp ) < UNDER_THE_WHEEL_COS )
	{
		if ( B3_IS_NON_NULL( w->tyreShape ) && b3Shape_IsValid( w->tyreShape ) )
		{
			b3MassData before = b3Body_GetMassData( v->chassis );
			b3DestroyShape( w->tyreShape, false );
			b3Body_SetMassData( v->chassis, before );
		}
		w->tyreShape = b3_nullShapeId;
		return;
	}

	float gravity = b3Length( b3World_GetGravity( v->world ) );
	float sag = w->suspensionStiffness > 0.0f && v->wheelCount > 0
					? gravity / ( w->suspensionStiffness * (float)v->wheelCount )
					: 0.0f;
	if ( sag > w->maxSuspensionTravel )
	{
		sag = w->maxSuspensionTravel;
	}
	float top = w->suspensionRestLength - w->maxSuspensionTravel - r;
	float bottom = w->suspensionRestLength - sag + r - TYRE_CLEARANCE;

	bool shaped = B3_IS_NON_NULL( w->tyreShape ) && b3Shape_IsValid( w->tyreShape );
	if ( !shaped || fabsf( top - w->tyreTop ) > 1.0e-3f || fabsf( bottom - w->tyreBottom ) > 1.0e-3f )
	{
		b3Filter machine;
		if ( !MachineFilter( v, &machine ) )
		{
			return;
		}

		// ROUND, as the tyre is: two treads of `TREAD_SEGMENTS` points around the
		// axle — the shape the cast already sweeps — one where the tyre sits fully
		// compressed and one where it sits loaded, and the hull of them: the tyre
		// over its travel. A box was tried first and its corners stood exactly 45
		// degrees from the hub, so a step's face read as ground under the wheel
		// and the machine drove straight up a 0.3 m step.
		b3Vec3 down = b3Normalize( w->directionCs );
		b3Vec3 up = Scale( down, -1.0f );
		b3Vec3 axle = Sub3( w->axleCs, Scale( up, b3Dot( w->axleCs, up ) ) );
		axle = b3Normalize( axle );
		b3Vec3 ahead = b3Cross( up, axle );
		float tread = r - TYRE_CLEARANCE;
		b3Vec3 points[4 * TREAD_SEGMENTS];
		int count = 0;
		for ( int end = 0; end < 2; end++ )
		{
			float along = end == 0 ? top + r : bottom - r + TYRE_CLEARANCE;
			b3Vec3 hub = Add3( w->chassisConnectionPointCs, Scale( down, along ) );
			for ( int i = 0; i < TREAD_SEGMENTS; i++ )
			{
				float a = ( 2.0f * 3.14159265358979f * (float)i ) / (float)TREAD_SEGMENTS;
				b3Vec3 radial = Add3( Scale( ahead, tread * cosf( a ) ), Scale( up, tread * sinf( a ) ) );
				points[count++] = Add3( Add3( hub, radial ), Scale( axle, 0.5f * w->width ) );
				points[count++] = Add3( Add3( hub, radial ), Scale( axle, -0.5f * w->width ) );
			}
		}
		b3HullData* hull = b3CreateHull( points, count, B3_MAX_HULL_VERTICES );
		if ( hull == NULL )
		{
			return;
		}

		b3ShapeDef def = b3DefaultShapeDef();
		def.density = 0.0f;
		def.baseMaterial.friction = TYRE_FRICTION;
		def.filter = machine;
		def.filter.categoryBits |= B3W_TYRE_CATEGORY;
		def.enablePreSolveEvents = true;
		def.enableCustomFiltering = true;
		def.userData = (void*)(intptr_t)( B3W_TYRE_TAG | ( v->handle << 12 ) | index );

		// A collider, not a part of the machine: the bump stop's reason, and the
		// same cure — a shape added to a chassis re-derives its mass.
		b3MassData before = b3Body_GetMassData( v->chassis );
		if ( shaped )
		{
			b3DestroyShape( w->tyreShape, false );
		}
		w->tyreShape = b3CreateHullShape( v->chassis, &def, hull );
		b3Body_SetMassData( v->chassis, before );
		b3DestroyHull( hull );

		b3Shape_SetContactAnchor( w->tyreShape, true, w->chassisConnectionPointCs );
		w->tyreTop = top;
		w->tyreBottom = bottom;
	}

	// The tread: the wheel's spin about its (steered) axle, through its centre, as
	// the body sees it — a tenth of it, as PhysX gives a rigid wheel contact.
	// Forward is up x axle (see `forwardWs`) and a positive spin rolls that way,
	// which puts the spin vector at -axle: the bottom of the tyre moves back.
	b3WorldTransform chassisT = b3Body_GetTransform( v->chassis );
	b3Vec3 spinWs = Scale( w->wheelAxleWs, -w->angularVelocity * TYRE_TREAD_SHARE );
	b3Vec3 down;
	b3Vec3 centre = b3InvRotateVector( chassisT.q, SubPos( WheelCentreNow( v, w, &down ), chassisT.p ) );
	b3Shape_SetSurfaceSpin( w->tyreShape, true, centre, b3InvRotateVector( chassisT.q, spinWs ) );
}

/** The wheel a tyre shape belongs to, or NULL for any other shape. */
static Wheel* TyreWheel( b3ShapeId shapeId, Vehicle** vehicle );

/**
 * Keep this rigid contact against a tyre, or leave it to the suspension?
 *
 * `PxVehicleModifyWheelContacts`, both of its tests, against the wheel where it
 * is now:
 *
 * - the contact is UNDER the wheel — the direction from the wheel's centre to
 *   the point, in the wheel's plane, within 45 degrees of the suspension's
 *   down;
 * - the surface pushes the wheel UP — the normal from what it hit toward the
 *   wheel within 45 degrees of the suspension's up.
 *
 * Either one makes it the suspension's, and the rigid contact is dropped.
 * Everything else is the tyre running into something, and stays.
 *
 * And people are never a tyre's to push: a host marks them never-ground.
 */
static bool KeepTyreContact( Vehicle* v, Wheel* w, b3ShapeId other, b3Pos point, b3Vec3 towardWheel )
{
	b3BodyId otherBody = b3Shape_GetBody( other );
	intptr_t tagged = (intptr_t)b3Body_GetUserData( otherBody );
	if ( tagged > 0 && b3w_owner_of_body( v->worldHandle, (int)( tagged - 1 ) ) == B3W_OWNER_NEVER_GROUND )
	{
		return false;
	}

	// A surface the wheel's sweep is STANDING ON this tick is the suspension's,
	// whatever angle the tyre now sees it at. PhysX's sweep and contact tests share
	// their thresholds so one surface can never be both; here they look from
	// slightly different places (the sweep's swept centre, the contact's live one),
	// and on a slope right at 45 degrees the same ground was carried by the spring
	// AND pushed by the tyre, and threw the wheel off it. A kerb's face on the same
	// body still meets the tyre: its normal is nowhere near the road's.
	if ( w->info.isInContact && B3_ID_EQUALS( otherBody, w->info.groundBodyId ) )
	{
		int count = w->info.contactCount < B3W_MAX_CONTACTS ? w->info.contactCount : B3W_MAX_CONTACTS;
		for ( int c = 0; c < count; c++ )
		{
			if ( b3Dot( w->info.contacts[c].surfaceNormalWs, towardWheel ) > SAME_SURFACE_COS )
			{
				return false;
			}
		}
	}

	b3Vec3 down;
	b3Pos centre = WheelCentreNow( v, w, &down );
	if ( -b3Dot( towardWheel, down ) > TYRE_WALL_COS )
	{
		return false;
	}

	b3Vec3 toPoint = SubPos( point, centre );
	toPoint = Sub3( toPoint, Scale( w->wheelAxleWs, b3Dot( toPoint, w->wheelAxleWs ) ) );
	float length = b3Length( toPoint );
	if ( length > 1.0e-6f && b3Dot( toPoint, down ) > TYRE_WALL_COS * length )
	{
		return false;
	}
	return true;
}

/**
 * The world's pre-solve, for tyres: -1 when neither shape is a tyre, 0 to drop
 * the contact, 1 to keep it (the host's own rules still apply after). Box3D's
 * normal points from A to B, so toward a tyre that is A it is reversed.
 */
int b3w_tyre_presolve( b3ShapeId shapeIdA, b3ShapeId shapeIdB, b3Pos point, b3Vec3 normal )
{
	Vehicle* va = NULL;
	Vehicle* vb = NULL;
	Wheel* wa = TyreWheel( shapeIdA, &va );
	Wheel* wb = TyreWheel( shapeIdB, &vb );
	if ( wa == NULL && wb == NULL )
	{
		return -1;
	}
	if ( wa != NULL && !KeepTyreContact( va, wa, shapeIdB, point, Scale( normal, -1.0f ) ) )
	{
		return 0;
	}
	if ( wb != NULL && !KeepTyreContact( vb, wb, shapeIdA, point, normal ) )
	{
		return 0;
	}
	return 1;
}

void b3w_vehicle_prepare( Vehicle* v, float dt )
{
	// Nothing to prepare against a chassis the caller has already destroyed, and
	// arming one would hand the substep solver a stale body id — see
	// `IsVehicleLive`.
	if ( !IsVehicleLive( v ) )
	{
		v->armed = false;
		return;
	}

	v->stepDt = dt > 0.0f ? dt : 1.0f / 60.0f;
	v->armed = true;

	/*
	 * Drop any wheel whose body is no longer ours, ONCE for the tick.
	 *
	 * The substep solve only checked for null, and a destroyed body's id is not
	 * null — it is recycled, so within a few rebuilds it addresses somebody
	 * else's body. Reading one drove a stale index into the body array, and the
	 * garbage that came back reached the continuous solver as a shape, which
	 * called through a function pointer that was never a function.
	 *
	 * Checked here rather than per substep because validity cannot change during
	 * a step: nothing destroys a body while the solver is running.
	 */
	DropDeadHandles( v );

	if ( v->chassisMassAsLeft > 0.0f && b3Body_GetType( v->chassis ) == b3_dynamicBody &&
		 b3Body_GetMass( v->chassis ) != v->chassisMassAsLeft )
	{
		RetakeWheelMass( v );
	}

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		UpdateTyre( v, &v->wheels[i], i );
	}

	// A fresh friction budget for the tick the substeps are about to share.
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		v->wheels[i].accumForward = 0.0f;
		v->wheels[i].accumSide = 0.0f;
		for ( int c = 0; c < B3W_MAX_CONTACTS; c++ )
		{
			// The accumulation MUST be reset. Left running across ticks it is a
			// total with nothing to discharge it — measured at 1,734,543 N on a
			// vehicle weighing 2,250.
			v->wheels[i].info.contacts[c].normalImpulse = 0.0f;
		}
		for ( int c = 0; c < B3W_MAX_CONTACTS; c++ )
		{
			v->wheels[i].info.contacts[c].normalImpulse = 0.0f;
		}
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
	//
	// KeepAwake, not SetAwake: a wake alone buys one step. The sleep timer is
	// judged on measured velocity, so a heavy hull whose engine force cannot
	// reach the threshold within B3_TIME_TO_SLEEP slept mid-throttle, lost its
	// velocity to the wake, and ratcheted from a standstill every sleep period.
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		if ( v->wheels[i].engineForce != 0.0f )
		{
			b3Body_KeepAwake( v->chassis );
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

		/*
		 * The steer the game asked for, NEGATED: `UpdateWheelTransforms` steers the
		 * cast's axle about `-wheelDirectionWs` (up), while the joint steers about
		 * frame A's local X, which this port builds along the suspension direction
		 * (down). Passed through unchanged the body turns the wrong way while the
		 * shapecast turns the right way.
		 */
		Wheel* steered = &v->wheels[i];
		if ( !B3_IS_NULL( steered->joint ) )
		{
			b3WheelJoint_SetTargetSteeringAngle( steered->joint, -steered->steering );
		}
	}

	/*
	 * Catch a chassis that TELEPORTED and bring its wheels with it.
	 *
	 * A wheel body is created by the controller and lives outside the host's body
	 * table, so everything that moves a machine — placing it, loading a save, the
	 * physgun — moves the chassis and leaves the wheels behind. The suspension
	 * joint then spans the whole distance and the solver answers with NaN: three
	 * cars spawned 25 m apart destroyed each other on the first step.
	 *
	 * Guarding here rather than at each of those call sites is deliberate. They
	 * do not know a wheel is a body, there is no reason they should, and the next
	 * one added would reintroduce this. A wheel further from its hard point than
	 * its suspension could ever carry it has not travelled, it has been moved,
	 * and the only sane response is to put it back under its mount.
	 */
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		if ( B3_IS_NULL( w->body ) )
		{
			continue;
		}

		float reach = w->suspensionRestLength + w->maxSuspensionTravel + w->radius;
		b3Vec3 fromHardPoint = SubPos( b3Body_GetPosition( w->body ), w->info.hardPointWs );
		if ( b3Length( fromHardPoint ) <= reach )
		{
			continue;
		}

		b3WorldTransform t = b3Body_GetTransform( v->chassis );
		b3Pos rest = AddPosVec( w->info.hardPointWs, Scale( w->wheelDirectionWs, w->suspensionRestLength ) );
		b3Body_SetTransform( w->body, rest, b3MulQuat( t.q, WheelFrameRotation( w->directionCs, w->axleCs ) ) );
		// MOVING WITH THE CHASSIS, not stopped. A wheel put back at rest under a
		// chassis doing 25 m/s is 40 kg of dead weight the joints average in: the
		// car read 19.5 m/s on the very next tick, every time it was placed.
		b3Body_SetLinearVelocity( w->body, b3Body_GetWorldPointVelocity( v->chassis, rest ) );
		b3Body_SetAngularVelocity( w->body, B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.0f } );
	}

	b3Vec3 linvel = b3Body_GetLinearVelocity( v->chassis );
	// RELATIVE TO WHAT THE WHEELS STAND ON. Road speed is what the drivetrain
	// and the static-friction latch are about, and a car parked on a deck doing
	// 25 m/s has none — read in world terms it sat past its own rev limiter and
	// could not pull away at all.
	{
		b3Vec3 groundSum = { 0.0f, 0.0f, 0.0f };
		int grounded = 0;
		for ( int i = 0; i < v->wheelCount; i++ )
		{
			Wheel* w = &v->wheels[i];
			if ( w->info.isInContact && GroundMoves( w->info.groundBodyId ) )
			{
				groundSum = Add3( groundSum, GroundVelocityAt( w->info.groundBodyId, w->info.contactPointWs ) );
				grounded++;
			}
		}
		if ( grounded > 0 )
		{
			linvel = Sub3( linvel, Scale( groundSum, 1.0f / (float)grounded ) );
		}
	}
	v->currentVehicleSpeed = b3Length( linvel );

	b3Quat chassisRot = b3Body_GetRotation( v->chassis );
	b3Vec3 forwardW = b3RotateVector( chassisRot, AxisVector( v->indexForwardAxis ) );
	if ( b3Dot( forwardW, linvel ) < 0.0f )
	{
		v->currentVehicleSpeed *= -1.0f;
	}

	// The cast, but NOT the placement. The substeps need contacts to push
	// against; the wheel's POSE is corrected once, after the solver has moved the
	// machine — see `CastAndPlaceWheels`. Placing here as well meant two
	// corrections per tick, and the second one had to undo the first: measured on
	// a slope climb, the extra shove added 20 mm of depth and half a metre per
	// second of speed the driver never asked for.
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		ShapeCastWheel( v, i, dt );

		// The cast has just said how far the ground is from the hard point, which
		// is the one measurement that says whether the stop is clear of it. See
		// `stopArmed`. With no contact there is nothing within the tyre's whole
		// reach, so it is clear by definition.
		Wheel* wheel = &v->wheels[i];
		if ( !wheel->stopArmed && B3_IS_NON_NULL( wheel->stopShape ) )
		{
			/*
			 * A tyre that has never stood on anything keeps its stop out of the
			 * world, and that is the half I got wrong first time: arming while the
			 * wheel was AIRBORNE let a machine whose wheels spawn level with its
			 * hull land on the stops instead of the tyres. The stop then carries
			 * the weight, the tyre stays under the surface reporting bottomed, and
			 * it never gets out — a machine resting 240 mm high on nothing, with
			 * no contact and no traction, for ever. Measured on a four-wheeled rig
			 * built from blocks: 4 of 4 wheels in contact became 0 of 4.
			 *
			 * The stop exists to catch a suspension that has run out of travel, so
			 * a suspension that has never worked yet has nothing for it to catch.
			 */
			float toGround = wheel->info.isInContact
								 ? wheel->info.suspensionLength + wheel->radius - wheel->info.bottomOutOvershoot
								 : -1.0f;
			if ( toGround > wheel->radius - BUMP_STOP_CLEARANCE )
			{
				b3Shape_SetFilter( wheel->stopShape, wheel->stopFilter, true );
				wheel->stopArmed = true;
			}
		}
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
		// Against the ground: a tyre rolling on a moving deck spins at its speed
		// OVER THE DECK, which is what the engine's revs are read from.
		b3Vec3 vel = Sub3( b3Body_GetWorldPointVelocity( v->chassis, w->info.hardPointWs ),
						   GroundVelocityAt( w->info.groundBodyId, w->info.contactPointWs ) );

		if ( w->inertia <= 0.0f )
		{
			// Solid disc, ~20 kg. Only a default: the game sets this per wheel.
			w->inertia = 0.5f * 20.0f * w->radius * w->radius;
		}

		/*
		 * The wheel is its own body about one axis, and this integrates it.
		 *
		 * Drive torque spins it up; the friction the tyre actually delivered
		 * spins it back down; the brake resists rotation. Whatever is left over
		 * IS wheel spin — the Uncharted 4 talk's "the leftover difference is
		 * applied back to the wheel as additional rotational velocity because
		 * the tire is slipping".
		 *
		 * The previous version simply defined the wheel's speed as the ground's,
		 * which is the same as declaring that tyres never slip. Everything the
		 * talk builds downstream — engine RPM from the fastest-spinning wheel,
		 * differentials, the slip feedback loop — needs this number to be able
		 * to disagree with the chassis.
		 */
		float driveTorque = w->engineForce * w->radius;
		// forwardImpulse is what the contact actually delivered over the tick;
		// as a force it is the reaction the ground applied back to the tyre.
		float frictionTorque = ( w->forwardImpulse * SafeInv( dt ) ) * w->radius;

		w->angularVelocity += ( driveTorque - frictionTorque ) * SafeInv( w->inertia ) * dt;

		if ( w->brake > 0.0f )
		{
			// A brake resists rotation and cannot reverse it, so it is clamped
			// at the speed that brings the wheel to a stop this tick.
			float brakeDeltaV = ( w->brake * w->radius ) * SafeInv( w->inertia ) * dt;
			float magnitude = fabsf( w->angularVelocity );
			if ( brakeDeltaV > magnitude )
			{
				brakeDeltaV = magnitude;
			}
			w->angularVelocity -= ( w->angularVelocity < 0.0f ? -1.0f : 1.0f ) * brakeDeltaV;
		}

		if ( w->info.isInContact )
		{
			/*
			 * The direction this tyre actually rolls: perpendicular to both the
			 * surface and its own axle.
			 *
			 * NOT `indexForwardAxis`. That is set to 0 — the X axis — when a
			 * vehicle is created and there is no API to configure it, while the
			 * axle on a normal car IS X. So the "forward" it named was the axle,
			 * dot(forward, velocity) was ~0 whenever the car drove forward, and
			 * the wheel's rolling speed came out as zero at 50 m/s. Engine RPM
			 * reads the fastest wheel, so the whole drivetrain silently sat at
			 * idle, the rev limiter never engaged, and top speed doubled from
			 * 23.8 to 50.3 m/s.
			 *
			 * The cross product needs no configuration, and it follows steering
			 * for free because the axle already has.
			 */
			// Axle-cross-normal, not the reverse: this is the order that makes a
			// wheel rolling the way the vehicle drives report a POSITIVE spin, so
			// it shares a sign convention with the drive torque that accelerates
			// it. Reversed, the two fight every time a tyre slips.
			b3Vec3 fwd = TryNormalize( b3Cross( w->wheelAxleWs, w->info.contactNormalWs ),
									   B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 0.0f } );

			// The spin a rolling tyre would have at this ground speed.
			float rollingOmega = b3Dot( fwd, vel ) * SafeInv( w->radius );

			// A tyre that still has grip is coupled to the ground: the talk's
			// "apply an opposing braking force to the wheel's angular velocity so
			// it slows toward the rotational velocity required by the vehicle's
			// movement". skidInfo is exactly how much of the friction circle is
			// left, so it is how hard that coupling pulls — at full grip the
			// wheel is dragged onto the rolling speed within the tick, and a tyre
			// that has broken traction is left to spin.
			// A tyre that has NOT broken traction is locked to the ground — the
			// talk's "with the clutch applied, the wheels and motor are directly
			// linked". Relaxing toward the rolling speed instead of onto it left
			// the wheel permanently a little slow, and since the engine takes its
			// revs from the fastest wheel, the whole drivetrain under-read: the
			// limiter stopped biting and top speed on a flat plane went from 23.8
			// to 50.3 m/s. A gripping wheel does not lag the road.
			//
			// Only once the friction circle has been exceeded is the wheel allowed
			// to run away from the ground, and then by exactly how much grip it
			// has lost.
			float grip = w->skidInfo > 1.0f ? 1.0f : ( w->skidInfo < 0.0f ? 0.0f : w->skidInfo );
			if ( !w->isSliding )
			{
				w->angularVelocity = rollingOmega;
			}
			else
			{
				w->angularVelocity += ( rollingOmega - w->angularVelocity ) * grip;
			}
		}

		w->deltaRotation = w->angularVelocity * dt;
		w->rotation += w->deltaRotation;
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

	/*
	 * Contact response: a unilateral point constraint per cast contact, holding
	 * the tyre to the point the cast found.
	 *
	 * This is the talk's second and third architectural facts, and it replaces
	 * the spring force that used to be computed here and applied to the CHASSIS.
	 * "This system allows us to pass a point that we want the tires to stick to
	 * and it also generates a normal force based on the motion and mass of the
	 * chassis and the wheels." The normal force is now that constraint's
	 * accumulated impulse rather than a number we invent and hand over, and it
	 * acts on the WHEEL, which carries it to the chassis through the suspension
	 * joint the engine solves.
	 *
	 * What went with the spring: the per-wheel force cap, the equal share across
	 * a wheel's contacts, and the implicit damper divided by the grounded-wheel
	 * count. Every one of those existed because N springs were pushing one body
	 * explicitly; N joints on N bodies are solved together, so none of them has
	 * anything left to correct.
	 */
	UpdateContacts( v, h );

	/*
	 * The spring is sized against the MACHINE, not against the hull alone.
	 *
	 * `computeSuspensionRates` derives a rate from the ride height a player
	 * authors — `k = |g| / (n * sag)` — and the force is that rate times the mass
	 * it carries. Passing only the chassis's mass forgot the wheels, which stopped
	 * being part of the chassis the moment each one became a body: the suspension
	 * was then sized for a load lighter than the machine and settled below its own
	 * sag. On a tank, whose wheels are a third of it, the inner wheels sat 94% of
	 * the way to their stops while parked on flat ground, with nothing left to
	 * absorb anything.
	 *
	 * A wheel resting on the ground carries itself, so this is a small correction
	 * for a car and a large one for a machine that is mostly wheels — which is
	 * exactly the difference between the two that kept showing up.
	 */
	float machineMass = b3Body_GetMass( v->chassis );
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		machineMass += v->wheels[i].massTakenFromChassis;
	}

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		ApplySuspensionForce( v, &v->wheels[i], machineMass, h );
	}

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		if ( B3_IS_NULL( w->body ) || w->info.contactCount == 0 )
		{
			/*
			 * A wheel touching nothing pushes on nothing.
			 *
			 * `SolveWheelContacts` clears this when it finds no rows, but skipping
			 * the call skipped the clearing too, so a wheel that left the ground
			 * kept the last normal force it ever had — for as long as it stayed in
			 * the air. A tank's raised idlers, which never touch at all, reported
			 * six kilonewtons apiece on a machine that weighs one. Friction grip is
			 * a coefficient times this number, so they carried the grip to match.
			 */
			w->wheelSuspensionForce = 0.0f;
			continue;
		}

		SolveWheelContacts( v, w, h );
	}

	UpdateFriction( v, h );
}

// ---------------------------------------------------------------------------
// Registry and the C API the wasm shim wraps
//
// Vehicles live here rather than in the shim so the shim stays a translation
// layer: it turns JavaScript's integers into ids and calls straight through.
// ---------------------------------------------------------------------------

#define B3W_MAX_VEHICLES 64

static Vehicle g_vehicles[B3W_MAX_VEHICLES];

static bool ReserveWheel( Vehicle* v );
static void FreeWheelStorage( Vehicle* v );

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
		if ( IsVehicleLive( &g_vehicles[i] ) && g_vehicles[i].armed && b3StoreWorldId( g_vehicles[i].world ) == stepping )
		{
			DropDeadHandles( &g_vehicles[i] );
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
			g_vehicles[i].anchorsReleased = false;
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
			// The bodies die with the world, but the wheel ARRAYS are ours and
			// outlive it. Flipping `used` alone leaked them on every teardown.
			FreeWheelStorage( &g_vehicles[i] );
			g_vehicles[i].used = false;
		}
	}
}

/**
 * Make room for one more wheel, doubling. False only when the heap is out.
 *
 * The three arrays grow together and are indexed in lockstep by every solver
 * loop, so a partial grow would be worse than no grow: it is all three or the
 * add fails.
 */
static bool ReserveWheel( Vehicle* v )
{
	if ( v->wheelCount < v->wheelCapacity )
	{
		return true;
	}

	int prev = v->wheelCapacity;
	int next = prev > 0 ? prev * 2 : 8;

	Wheel* wheels = (Wheel*)realloc( v->wheels, (size_t)next * sizeof( Wheel ) );
	if ( wheels == NULL )
	{
		return false;
	}
	v->wheels = wheels;

	b3Vec3* forwardWs = (b3Vec3*)realloc( v->forwardWs, (size_t)next * sizeof( b3Vec3 ) );
	if ( forwardWs == NULL )
	{
		return false;
	}
	v->forwardWs = forwardWs;

	b3Vec3* axle = (b3Vec3*)realloc( v->axle, (size_t)next * sizeof( b3Vec3 ) );
	if ( axle == NULL )
	{
		return false;
	}
	v->axle = axle;

	/*
	 * ZERO the new tail. These were inline arrays inside `Vehicle`, so the
	 * memset in `b3w_vehicle_create` covered them; realloc does not, and the
	 * difference is not academic.
	 *
	 * `axle` and `forwardWs` are written only for wheels that found ground
	 * (see the `groundBody < 0` skip in the friction pass) but READ wherever a
	 * wheel has a non-zero impulse — so an airborne wheel holding a stale
	 * impulse reads whatever the allocator last left here and applies it as a
	 * direction. Uninitialised, that is an arbitrary force on the chassis; a
	 * drive test caught it as 3.2 m/s of upward velocity out of nowhere.
	 */
	memset( v->wheels + prev, 0, (size_t)( next - prev ) * sizeof( Wheel ) );
	memset( v->forwardWs + prev, 0, (size_t)( next - prev ) * sizeof( b3Vec3 ) );
	memset( v->axle + prev, 0, (size_t)( next - prev ) * sizeof( b3Vec3 ) );

	v->wheelCapacity = next;
	return true;
}

/**
 * Release the wheel arrays and leave the vehicle empty-but-valid.
 *
 * Called from BOTH teardown paths. `b3w_vehicle_create` memsets the whole
 * struct, so a slot that is reused must arrive with its pointers already freed
 * and nulled — that invariant is what makes the memset safe, and it is why
 * `destroy_world` can no longer just flip `used`.
 */
static void FreeWheelStorage( Vehicle* v )
{
	free( v->wheels );
	free( v->forwardWs );
	free( v->axle );
	v->wheels = NULL;
	v->forwardWs = NULL;
	v->axle = NULL;
	v->wheelCount = 0;
	v->wheelCapacity = 0;
}

static Wheel* TyreWheel( b3ShapeId shapeId, Vehicle** vehicle )
{
	intptr_t tag = (intptr_t)b3Shape_GetUserData( shapeId );
	if ( ( tag & B3W_TYRE_TAG ) == 0 )
	{
		return NULL;
	}
	int handle = (int)( ( tag & ~(intptr_t)B3W_TYRE_TAG ) >> 12 );
	int index = (int)( tag & 0xFFF );
	if ( handle < 0 || handle >= B3W_MAX_VEHICLES || !g_vehicles[handle].used )
	{
		return NULL;
	}
	Vehicle* v = &g_vehicles[handle];
	if ( index >= v->wheelCount || !B3_ID_EQUALS( v->wheels[index].tyreShape, shapeId ) )
	{
		return NULL;
	}
	*vehicle = v;
	return &v->wheels[index];
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
		v->handle = i;
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
	if ( v == NULL )
	{
		return;
	}

	// Only here, never in `b3w_vehicle_destroy_world`: a world takes its bodies
	// with it, so destroying them afterwards would be a free of freed memory.
	bool chassisLives = !B3_IS_NULL( v->chassis ) && b3Body_IsValid( v->chassis );

	for ( int i = 0; i < v->wheelCount; i++ )
	{
		// Only what we still own. The game's teardown destroys every body before
		// it frees the controllers, and Box3D takes a body's joints with it — so
		// by the time we get here our joint may already be gone.
		//
		// Validity is NOT the test for that. Ids are recycled, so a freed joint
		// id becomes a live id belonging to another machine within a few
		// rebuilds, and destroying it on that machine's behalf corrupts a solver
		// that has no idea it happened. The chassis is the honest witness: while
		// it lives, our joints live; once it is gone, so are they.
		if ( chassisLives && !B3_IS_NULL( v->wheels[i].joint ) )
		{
			b3DestroyJoint( v->wheels[i].joint, false );
		}
		v->wheels[i].joint = b3_nullJointId;

		// The tyre is ours too, on a chassis that may outlive us (a rebuild hands the
		// same chassis a new controller). Its mass is restored as it was added.
		if ( chassisLives && B3_IS_NON_NULL( v->wheels[i].tyreShape ) && b3Shape_IsValid( v->wheels[i].tyreShape ) )
		{
			b3MassData before = b3Body_GetMassData( v->chassis );
			b3DestroyShape( v->wheels[i].tyreShape, false );
			b3Body_SetMassData( v->chassis, before );
		}
		v->wheels[i].tyreShape = b3_nullShapeId;

		// The wheel BODY is ours alone — nothing else destroys it — so it is
		// still here whatever became of the chassis.
		if ( !B3_IS_NULL( v->wheels[i].body ) && b3Body_IsValid( v->wheels[i].body ) )
		{
			b3DestroyBody( v->wheels[i].body );
		}
		v->wheels[i].body = b3_nullBodyId;
	}

	FreeWheelStorage( v );
	v->used = false;
}

/**
 * Give the WHOLE machine a velocity: chassis and every wheel body.
 *
 * A wheel is a body of its own, so writing the chassis velocity alone left four
 * wheels at whatever they had before and the joints averaged the two: a car
 * placed at 25 m/s was doing 20.3 a tick later, which is exactly the momentum
 * share of 40 kg of wheels against a 174 kg chassis. Every placement of a
 * remote car paid it.
 */
void b3w_vehicle_set_velocity( int handle, float vx, float vy, float vz, float wx, float wy, float wz )
{
	if ( handle < 0 || handle >= B3W_MAX_VEHICLES || !IsVehicleLive( &g_vehicles[handle] ) )
	{
		return;
	}
	Vehicle* v = &g_vehicles[handle];
	b3Vec3 linear = { vx, vy, vz };
	b3Vec3 angular = { wx, wy, wz };
	b3Body_SetLinearVelocity( v->chassis, linear );
	b3Body_SetAngularVelocity( v->chassis, angular );
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		Wheel* w = &v->wheels[i];
		if ( B3_IS_NULL( w->body ) || !b3Body_IsValid( w->body ) )
		{
			continue;
		}
		// Linear only. A wheel's spin about its axle is its own, and overwriting
		// it every placement with the chassis rate fought the joint hard enough
		// to roll cars onto their roofs in the rear-end and t-bone stagings.
		b3Vec3 arm = SubPos( b3Body_GetPosition( w->body ), b3Body_GetPosition( v->chassis ) );
		b3Body_SetLinearVelocity( w->body, Add3( linear, b3Cross( angular, arm ) ) );
	}
}

/**
 * The machine was MOVED, not driven: let every tyre grip wherever it now is.
 *
 * A parked tyre latches the point it stands on and hauls the car back to it
 * for any slip under `MAX_STATIC_FRICTION_SLIP`. That is right for a car the
 * solver moves and wrong for one the host moves: a network correction closes a
 * gap a centimetre a tick, every centimetre was pulled straight back, and a
 * remote car at rest sat a quarter metre from where its owner had it,
 * indefinitely.
 *
 * For the WHOLE next step, not only its first substep: friction is solved
 * every substep, and a tyre that re-latched in the first one held the car
 * against the rest of the step's motion. Latched again on the step after.
 */
void b3w_vehicle_release_anchors( int handle )
{
	if ( handle < 0 || handle >= B3W_MAX_VEHICLES || !IsVehicleLive( &g_vehicles[handle] ) )
	{
		return;
	}
	Vehicle* v = &g_vehicles[handle];
	v->anchorsReleased = true;
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		v->wheels[i].hasAnchor = false;
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
 * How many wheels the readback buffer must hold.
 *
 * `b3w_vehicle_read_wheels` writes `wheelCount * WHEEL_READBACK_STRIDE` floats
 * into a pointer it cannot size-check. The host used to allocate that buffer
 * from its own copy of the old `32`, so the two agreed only by luck; raising
 * the cap on one side alone wrote past the allocation. Ask, do not assume.
 */
int b3w_vehicle_wheel_capacity( int handle )
{
	Vehicle* v = b3w_vehicle_get( handle );
	return v != NULL ? v->wheelCapacity : 0;
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
	if ( v == NULL || !ReserveWheel( v ) )
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
	w->angularVelocity = 0.0f;
	w->inertia = 0.0f; // derived from the radius on first use
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

	CreateWheelBody( v, w );

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

/**
 * The tyre's mass, in the same units the chassis was built from.
 *
 * Not a `WHEEL_SETTER`: it has to move the mass off the chassis and rebuild the
 * body's inertia, which is what makes a machine weigh the same after its wheels
 * became bodies as it did before.
 */
void b3w_vehicle_set_wheel_mass( int handle, int index, float mass )
{
	Vehicle* v = b3w_vehicle_get( handle );
	if ( v == NULL || index < 0 || index >= v->wheelCount )
	{
		return;
	}

	Wheel* w = &v->wheels[index];
	if ( w->massTakenFromChassis > 0.0f )
	{
		// Give back what this wheel TOOK — not its mass, which the share capped,
		// and which a weightless wheel (a placed replica's) never took at all —
		// so a second call is not a second subtraction. Giving back `w->mass`
		// left `massTakenFromChassis` standing after a wheel went to zero, and
		// the next call counted it into the machine again: every placed and
		// simulated round trip made a car 10 kg a wheel heavier.
		b3MassData chassis = b3Body_GetMassData( v->chassis );
		chassis.mass += w->massTakenFromChassis;
		b3Body_SetMassData( v->chassis, chassis );
		w->massTakenFromChassis = 0.0f;
		v->chassisMassAsLeft = chassis.mass;
	}

	w->mass = mass;
	ApplyWheelMass( v, w );
}

WHEEL_SETTER( b3w_vehicle_set_engine_force, engineForce )
WHEEL_SETTER( b3w_vehicle_set_brake, brake )
/**
 * Steering is a COMMAND, so it wakes what it commands.
 *
 * Every other way a seated player can ask a machine to move already wakes it —
 * engine force in `prepare`, joint motors at their setters, thrusters through
 * the force accumulator. Steering was the one that only wrote its field, and a
 * blanket "keep every seated machine awake" in the host was covering for it, at
 * the cost of sleep for every car a player sits in.
 *
 * On CHANGE, not on non-zero: a machine parked with its wheels turned is still
 * parked, and waking on the standing value would be the same hold under another
 * name. The host dedups identical commands, so this fires when the player
 * actually moves the wheel.
 */
void b3w_vehicle_set_steering( int handle, int index, float value )
{
	Vehicle* v = b3w_vehicle_get( handle );
	if ( v == NULL || index < 0 || index >= v->wheelCount )
	{
		return;
	}

	if ( v->wheels[index].steering != value )
	{
		v->wheels[index].steering = value;
		b3Body_SetAwake( v->chassis, true );
	}
}
WHEEL_SETTER( b3w_vehicle_set_side_friction, sideFrictionStiffness )
WHEEL_SETTER( b3w_vehicle_set_friction_slip, frictionSlip )
/**
 * The rates, which have to reach the JOINT and not just the struct.
 *
 * These arrive after the wheels are built — the spring depends on how many
 * wheels end up sharing the machine's weight, which the caller only knows once
 * the loop finishes — so a setter that only wrote the field would leave every
 * suspension on its creation-time default forever.
 */
#define WHEEL_RATE_SETTER( name, field )                                                                               \
	void name( int handle, int index, float value )                                                                    \
	{                                                                                                                  \
		Vehicle* v = b3w_vehicle_get( handle );                                                                        \
		if ( v != NULL && index >= 0 && index < v->wheelCount )                                                        \
		{                                                                                                              \
			v->wheels[index].field = value;                                                                            \
			ConfigureWheelJoint( v, &v->wheels[index] );                                                               \
		}                                                                                                              \
	}

WHEEL_RATE_SETTER( b3w_vehicle_set_suspension_stiffness, suspensionStiffness )
WHEEL_RATE_SETTER( b3w_vehicle_set_suspension_compression, dampingCompression )
WHEEL_RATE_SETTER( b3w_vehicle_set_suspension_relaxation, dampingRelaxation )
WHEEL_RATE_SETTER( b3w_vehicle_set_suspension_rest_length, suspensionRestLength )
WHEEL_RATE_SETTER( b3w_vehicle_set_max_suspension_travel, maxSuspensionTravel )
WHEEL_SETTER( b3w_vehicle_set_wheel_inertia, inertia )
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
		float* o = out + i * WHEEL_READBACK_STRIDE;

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
		o[19] = w->angularVelocity;
		/*
		 * The wheel body's pose — where the wheel IS, which is now the same thing
		 * as where it is drawn.
		 *
		 * The Uncharted 4 port's fifth Phase 1 item: one source for physics and
		 * rendering. What used to sit at [20] was a suspension length the
		 * controller corrected after the fact so the renderer could rebuild a
		 * position from it; the body makes that round trip unnecessary, and the
		 * correction is deleted rather than moved.
		 */
		bool hasBody = !B3_IS_NULL( w->body );
		o[20] = hasBody ? 1.0f : 0.0f;
		o[21] = w->skidInfo;

		/*
		 * Where the wheel sits relative to its hard point, in CHASSIS-LOCAL axes.
		 *
		 * The wheel is a body now, so this is measured off that body rather than
		 * derived from a spring length and then corrected — the port's fifth
		 * Phase 1 item, one source for physics and rendering.
		 *
		 * It crosses as a displacement in the chassis's frame, not as a world
		 * transform, because the renderer composes `bodyMatrix * localMatrix` on
		 * the GPU: a world position handed to a pooled instance is multiplied by
		 * the body matrix a second time, which looks right at the origin and is
		 * wrong everywhere else. Three floats in the frame the consumer actually
		 * works in beat seven that it has to undo.
		 *
		 * A lateral or twisting component survives this, where a scalar length
		 * could not — which matters the moment the constraint puts the tyre
		 * somewhere the suspension axis alone cannot express.
		 */
		// Against the hard point as it is NOW, rebuilt from the live chassis
		// transform. `info.hardPointWs` is a prepare-time value: read here it
		// measures the wheel against a chassis pose the step has already left
		// behind, and the tyre is drawn off the machine by however far the
		// chassis travelled during the tick.
		b3WorldTransform chassisT = b3Body_GetTransform( v->chassis );
		b3Pos hardPointWs = AddPosVec( chassisT.p, b3RotateVector( chassisT.q, w->chassisConnectionPointCs ) );

		b3Vec3 offsetWs = hasBody ? SubPos( b3Body_GetPosition( w->body ), hardPointWs )
								  : Scale( w->wheelDirectionWs, w->info.suspensionLength );
		b3Vec3 offsetCs = b3InvRotateVector( chassisT.q, offsetWs );

		o[22] = offsetCs.x;
		o[23] = offsetCs.y;
		o[24] = offsetCs.z;
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

/**
 * Draw every wheel body of every vehicle in a world, appended to the shim's
 * debug buffers.
 *
 * The shim's own debug draw cannot reach these: it walks the body table it
 * created and draws SHAPES, and a wheel body is in neither — it is created here
 * and it carries no shapes, because "we turned off collision on the wheels" is
 * expressed as having nothing to collide with. So a wheel was invisible in F3
 * exactly when it most needed looking at, and "I don't see rigid body wheels"
 * was a true reading of a real gap.
 *
 * What is drawn is the tyre the body stands for — a ring of its radius in the
 * plane across its axle, plus the axle itself — from the body's own transform.
 * It is deliberately a different colour from anything with a collider: yellow
 * for a wheel that is on the ground, orange for one in the air. Seeing a ring
 * with no collider box around it IS the port's first architectural fact.
 */
int b3w_vehicle_debug_draw( int worldHandle, float* verts, float* colors, int capacityVerts, int used )
{
	if ( verts == NULL || colors == NULL )
	{
		return used;
	}

	const int segments = 20;

	for ( int i = 0; i < B3W_MAX_VEHICLES; i++ )
	{
		Vehicle* v = &g_vehicles[i];
		if ( !v->used || v->worldHandle != worldHandle )
		{
			continue;
		}

		for ( int wi = 0; wi < v->wheelCount; wi++ )
		{
			Wheel* w = &v->wheels[wi];
			if ( B3_IS_NULL( w->body ) || !b3Body_IsValid( w->body ) )
			{
				continue;
			}

			b3WorldTransform t = b3Body_GetTransform( w->body );
			b3Vec3 axle = b3RotateVector( t.q, B3_LITERAL( b3Vec3 ){ 0.0f, 0.0f, 1.0f } );
			b3Vec3 u = b3RotateVector( t.q, B3_LITERAL( b3Vec3 ){ 1.0f, 0.0f, 0.0f } );
			b3Vec3 n = b3RotateVector( t.q, B3_LITERAL( b3Vec3 ){ 0.0f, 1.0f, 0.0f } );

			float r = w->radius;
			float half = 0.5f * w->width;
			float cr = w->info.isInContact ? 1.0f : 1.0f;
			float cg = w->info.isInContact ? 0.85f : 0.5f;
			float cb = w->info.isInContact ? 0.1f : 0.1f;

			// Both rims, so the tyre reads as a wheel and not as a hoop.
			for ( int side = 0; side < 2; side++ )
			{
				float offset = side == 0 ? half : -half;

				for ( int k = 0; k < segments; k++ )
				{
					if ( used + 2 > capacityVerts )
					{
						return used;
					}

					float a0 = ( 2.0f * 3.14159265358979f * (float)k ) / (float)segments;
					float a1 = ( 2.0f * 3.14159265358979f * (float)( k + 1 ) ) / (float)segments;

					b3Vec3 p0 = Add3( Add3( Scale( u, r * cosf( a0 ) ), Scale( n, r * sinf( a0 ) ) ), Scale( axle, offset ) );
					b3Vec3 p1 = Add3( Add3( Scale( u, r * cosf( a1 ) ), Scale( n, r * sinf( a1 ) ) ), Scale( axle, offset ) );

					float* out = verts + used * 3;
					out[0] = (float)t.p.x + p0.x;
					out[1] = (float)t.p.y + p0.y;
					out[2] = (float)t.p.z + p0.z;
					out[3] = (float)t.p.x + p1.x;
					out[4] = (float)t.p.y + p1.y;
					out[5] = (float)t.p.z + p1.z;

					float* col = colors + used * 4;
					for ( int c = 0; c < 2; c++ )
					{
						col[c * 4 + 0] = cr;
						col[c * 4 + 1] = cg;
						col[c * 4 + 2] = cb;
						col[c * 4 + 3] = 1.0f;
					}
					used += 2;
				}
			}

			// The axle, which is also the spin axis the joint gives the body.
			if ( used + 2 <= capacityVerts )
			{
				float* out = verts + used * 3;
				out[0] = (float)t.p.x - axle.x * half;
				out[1] = (float)t.p.y - axle.y * half;
				out[2] = (float)t.p.z - axle.z * half;
				out[3] = (float)t.p.x + axle.x * half;
				out[4] = (float)t.p.y + axle.y * half;
				out[5] = (float)t.p.z + axle.z * half;

				float* col = colors + used * 4;
				for ( int c = 0; c < 2; c++ )
				{
					col[c * 4 + 0] = cr;
					col[c * 4 + 1] = cg;
					col[c * 4 + 2] = cb;
					col[c * 4 + 3] = 1.0f;
				}
				used += 2;
			}
		}
	}

	return used;
}

/**
 * Look, then fix the wheels — the post-step half of the tick.
 *
 * The cast in `prepare` describes the world as it was BEFORE the solver
 * integrated anything, so placing against it afterwards corrects the wheel
 * toward a surface the chassis has already left. At 12 m/s a tick is half a
 * metre, which is most of a wheel, and the faster the approach to a rise the
 * deeper the tyre ends up buried in it. Casting again here costs a second cast
 * per wheel per tick and answers the question the placement is actually asking:
 * where is the ground under the wheel NOW.
 */
static void CastAndPlaceWheels( Vehicle* v )
{
	for ( int i = 0; i < v->wheelCount; i++ )
	{
		UpdateWheelTransforms( v, i );
		ShapeCastWheel( v, i, v->stepDt );
	}

	PlaceWheels( v );
}

/**
 * Place every armed vehicle's wheels, after the world has stepped.
 *
 * The talk's step 7 is a POSE assignment, and doing it only before the solve
 * leaves the solver free to undo it inside the step: measured on a rig, 11.7 mm
 * of tyre below flat ground placed before the step, 0.00 mm placed after it as
 * well. It costs no casts — the contact geometry is the one the tick already
 * found.
 */
void b3w_vehicle_place_wheels( int worldHandle )
{
	for ( int i = 0; i < B3W_MAX_VEHICLES; i++ )
	{
		Vehicle* v = &g_vehicles[i];
		if ( !IsVehicleLive( v ) || v->worldHandle != worldHandle )
		{
			continue;
		}

		if ( b3Body_GetType( v->chassis ) == b3_dynamicBody && !b3Body_IsAwake( v->chassis ) )
		{
			continue;
		}

		CastAndPlaceWheels( v );
	}
}

/** How many wheel bodies are awake, across every vehicle in the world. They are
 *  invisible to the body iteration — the controller owns them — so this is the
 *  only way to see one holding an island awake. */
int b3w_vehicle_awake_wheels( int worldHandle )
{
	int awake = 0;
	for ( int i = 0; i < B3W_MAX_VEHICLES; i++ )
	{
		Vehicle* v = &g_vehicles[i];
		if ( !v->used || v->worldHandle != worldHandle )
		{
			continue;
		}

		for ( int k = 0; k < v->wheelCount; k++ )
		{
			if ( !B3_IS_NULL( v->wheels[k].body ) && b3Body_IsAwake( v->wheels[k].body ) )
			{
				awake++;
			}
		}
	}

	return awake;
}

/**
 * The worst wheel body's sleep bookkeeping, for the same question the chassis
 * answers through `b3w_body_sleep_state`: wheel bodies are invisible to the
 * shim's body walk, so this is the only way to see one holding an island awake.
 * `out[0]` sleep velocity, `out[1]` threshold, `out[2]` sleep time, `out[3]` how
 * many wheel bodies are awake.
 */
void b3w_vehicle_wheel_sleep( int worldHandle, float* out )
{
	out[0] = 0.0f;
	out[1] = 0.0f;
	out[2] = 1e9f;
	out[3] = 0.0f;

	for ( int i = 0; i < B3W_MAX_VEHICLES; i++ )
	{
		Vehicle* v = &g_vehicles[i];
		if ( !v->used || v->worldHandle != worldHandle )
		{
			continue;
		}

		for ( int k = 0; k < v->wheelCount; k++ )
		{
			Wheel* w = &v->wheels[k];
			if ( B3_IS_NULL( w->body ) || !b3Body_IsValid( w->body ) )
			{
				continue;
			}

			float velocity = b3Body_GetSleepVelocity( w->body );
			if ( velocity > out[0] )
			{
				out[0] = velocity;
				out[1] = b3Body_GetSleepThreshold( w->body );
			}

			float time = b3Body_GetSleepTime( w->body );
			if ( time < out[2] )
			{
				out[2] = time;
			}

			if ( b3Body_IsAwake( w->body ) )
			{
				out[3] += 1.0f;
			}
		}
	}

	if ( out[2] > 1e8f )
	{
		out[2] = 0.0f;
	}
}

/** Worst |v|, |w|, placement distance and hard-point force across every wheel
 *  body. Diagnostic only — four floats, written to `out`. */
void b3w_vehicle_wheel_speeds( int worldHandle, float* out )
{
	out[0] = 0.0f;
	out[1] = 0.0f;
	out[2] = 0.0f;
	out[3] = 0.0f;

	for ( int i = 0; i < B3W_MAX_VEHICLES; i++ )
	{
		Vehicle* v = &g_vehicles[i];
		if ( !v->used || v->worldHandle != worldHandle )
		{
			continue;
		}

		for ( int k = 0; k < v->wheelCount; k++ )
		{
			Wheel* w = &v->wheels[k];
			if ( B3_IS_NULL( w->body ) )
			{
				continue;
			}

			float linear = b3Length( b3Body_GetLinearVelocity( w->body ) );
			float angular = b3Length( b3Body_GetAngularVelocity( w->body ) );

			out[0] = linear > out[0] ? linear : out[0];
			out[1] = angular > out[1] ? angular : out[1];
			out[2] = w->placementNeeded > out[2] ? w->placementNeeded : out[2];
			out[3] = w->hardPointForce > out[3] ? w->hardPointForce : out[3];
		}
	}
}

