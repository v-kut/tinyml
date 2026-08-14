# Exact walls, a corner floor read off the car, a warped LiDAR fan

Findings behind `tinyml_racing/sim/`.

## Exact arcs, not a sampled polyline

The LiDAR and the termination test are the only consumers of wall geometry, and both run
every step. A sampled polyline cost 850 us and 39 us per step against the ~800-1,200
chords the walls come to at 4 m spacing, and answered wrong: the sagitta is 0.11 m at the
99th percentile, and roughly one ray in three thousand grazes a chord where it would have
missed the arc, reporting the wall behind it 47 m out.

Offsetting straights and arcs is closed form, so `WallGeometry` holds both walls as
`2 x n_corners` arcs and as many straights, and never samples them. The step went from
1,050 us to 185 us and the geometry from approximate to exact. It costs a constant
corridor width per layout, since only a constant offset of an arc is an arc.

The polyline survives as the test oracle: the geometry test asserts it converges onto the
exact caster quadratically as spacing shrinks, which makes "exact" a measurement.

## The corner floor had drifted away from the car

`min_corner_radius` was a literal 10.0 m, from the kart this started as, while `CarParams`
grew into a car whose full-lock radius is 8.6 m. That left nothing for understeer at the
limit, so the generator drew corners that could not be driven: pure pursuit crashed on
layouts it should have lapped and the geometry tests failed.

Now `2 * min_turn_radius` = 17.13 m, derived through a `default_factory` so the two cannot
drift apart. The expert laps every layout again, which is the check: it validates the
parameters, so a generator it cannot drive is a generator bug.

## Three things were not wrong

- `tangents` and `curvature` are exact closed forms. What disagreed was a finite
  difference of the samples: refining `sample_spacing` over a decade shrinks the tangent
  disagreement from 2.99 deg to 0.23 deg.
- The curvature residual cannot converge. Curvature is a square wave over exact arcs and
  straights, and a central difference across that step returns about half the arc's value
  at any spacing. `1/(2R)` is the floor, and the tolerance says so.
- The normals were never flipped. The assertion compared distances from the centroid,
  valid only on a convex lap; it is a signed-area comparison now.

## Spawns are drawn against a brake plan

`random_start_state` used to cap spawn speed by the curvature of the sample it landed on,
so a spawn two metres before a hairpin read "straight" and drew up to 43 m/s, a crash
charged to a policy that had not acted. It is now the backward pass of a speed plan,
`v <= sqrt(v_j^2 + 2 a d_j)` over every sample within braking reach, at the deceleration
`PurePursuit` permits itself. Audited over 2,400 spawns on 8 layouts, the worst ratio of
drawn to brakeable speed is 1.000.

## The fan is warped and widened

60 rays evenly over 180 deg is 60 numbers of which maybe 20 are distinct. Over 100k expert
states, the 20 rays past +/-60 deg report a median 6.8 m against a 6.5 m corridor
half-width, while the 14 inside +/-20 deg are the only ones with reach (39 m median, 118 m
at the 90th percentile) and the only ones that see a corner in time to brake.

So bearings are `phi(u) = half_fov * tan(a*u) / tan(a)`: evenly spaced points ahead of the
car projected back onto angles, so rays land at even distances along a corridor.
`ray_focus` states the warp as outermost/innermost spacing, `sec^2(a)`, so 1.0 is uniform.
At 28 rays over 240 deg at focus 9 that is 10 rays inside +/-20 deg, 3.9 deg apart,
against 4 at 8.9 deg spread evenly. The extra 60 deg costs no rays, bought by widening the
sparse end, and is the only thing that reports the wall a sliding car is heading for.

Not backed by an end-to-end win. Cloning under five fan geometries, capacity taken out of
the question, leaves steering error flat; throttle error improves with focus consistently,
the expected direction. A short PPO A/B over two seeds cannot separate them. Justified by
ray-budget efficiency and wider coverage at equal cost, not by a lap time.

## The vehicle

A 1300 kg GT3-class car: 2.65 m wheelbase, CG slightly forward, `mu` 1.55 on a dry slick,
cornering stiffness per unit normal load (rear stiffer, so understeer at the limit), load
transfer, a friction ellipse with longitudinal priority, a thrust curve decaying with road
speed, rolling resistance, drag, downforce split by static weight distribution. `dt` is
0.02 s with 8 substeps: a full-lock transient at 20 m/s held for 2 s lands ~0.9% from a
32-substep reference, against ~3% at one substep. Below `blend_speed` the model relaxes
onto the kinematic form, where the slip formulation is singular.

`../PROPOSAL.md` describes a 1/28-scale RC car with a regressor fit to pure-pursuit
labels. The simulator was scaled to a full-size car and the policy is trained by PPO.
Nothing downstream of `sim/` reads a vehicle parameter, so the choice never reaches
deployment.
