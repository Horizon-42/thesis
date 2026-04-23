# Runway Profile Altitude Anomaly Postmortem

## Summary

We found a data-driven altitude anomaly in the 2D runway trajectory profile, most visibly on **KRDU RW32**.

The symptom was:

- the **side view** `(x, z)` started with a strangely low or negative `z` value
- the dip happened near the **beginning of a transition route**
- the result looked like the aircraft or fix started **below the runway**

This was **not** a real procedure altitude. It came from **unknown fix altitude values being exported as `0`**, and then being interpreted by the profile renderer as if `0` were a valid geometric altitude.

The issue has now been fixed in the profile geometry pipeline.

## What We Observed

The most obvious case was on **RW32** in:

- `KRDU-R32-ACONCA`
- `KRDU-R32-ASINNO`

In [procedures.geojson](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/public/data/airports/KRDU/procedures.geojson:2038) we found the same pattern elsewhere too, for example on `KRDU-R23LY-ADUWON`.

For the bad pattern, the first transition fix was exported like this:

- `altitudeFt: null`
- `geometryAltitudeFt: 0`
- coordinate altitude `0.0`

but the next fix on the same route had a normal valid altitude, for example:

- `NOSIC`: `3400 ft`

So the route was not really descending to the ground. The first point simply had an **unknown altitude that had been defaulted to zero**.

## Root Cause

The error happened in two steps.

### 1. Data export used `0` as a placeholder for "unknown altitude"

Example pattern in the source data:

- `altitudeFt: null`
- `geometryAltitudeFt: 0`
- warning text like `altitude defaulted to 0 ft`

That means the data did **not** say "this fix is at sea level".
It really said:

- "we do not know the altitude from the parsed source"

But that uncertainty was represented as numeric zero.

### 2. The profile renderer trusted the route geometry altitude too literally

Before the fix, the runway profile code projected each route point using the raw coordinate altitude from the route geometry.

That meant:

- `0.0 m` was treated as a real altitude
- the point was then converted into runway-relative `z`
- because runway threshold elevation is above zero, the displayed `z` became artificially low or negative

So the rendered side view showed a fake dip at the beginning of the route.

## Why RW32 Made It Easy To Notice

RW32 exposed the issue clearly because:

- the bad point was at the **start** of the transition
- the next point had a valid high altitude
- that created a sharp, visually obvious jump in the side profile

In other words, RW32 was not necessarily unique. It was simply the cleanest example of a broader data pattern.

## The Fix

The solution is implemented in [runwayProfileGeometry.ts](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/src/utils/runwayProfileGeometry.ts:280).

### Step 1. Stop treating `0` as a trustworthy altitude when the data is really "unknown"

The new helper `preferredRouteAltitudeM(...)` now chooses altitude in this order:

1. coordinate altitude, but only if it is **positive and valid**
2. `geometryAltitudeFt`, but only if it is **positive and valid**
3. `altitudeFt`, but only if it is **positive and valid**
4. otherwise: `null` meaning "missing"

This is important because:

- `0` is no longer accepted as a meaningful profile altitude just because it is numeric

### Step 2. Fill missing route heights from nearby valid points

The helper `fillMissingAltitudes(...)` then repairs missing route heights:

- if both previous and next valid altitudes exist, it interpolates between them
- if only one side exists, it uses that nearest valid altitude
- only if **the whole route has no valid altitude at all** does it fall back to `0`

This is the key safeguard that removes the fake start-of-route dip.

### Step 3. Use the corrected route heights everywhere in the profile

The helper `buildProjectedRoutePoints(...)` now produces the projected profile points used by:

- the route lines in the 2D profile
- the horizontal-plate geometry
- the plotted fix/reference marks

So the fix is centralized. We are not patching RW32 by name.

## Why This Should Not Happen Again

For the specific class of error we just fixed, the protection is strong.

### The fix is route-agnostic

The code does **not** special-case:

- KRDU
- RW32
- ACONCA
- ASINNO

It applies to **all runways and all routes** that go through the runway profile pipeline.

That means any future route with:

- `altitudeFt: null`
- `geometryAltitudeFt: 0`
- coordinate altitude `0`
- but nearby valid route heights

will be repaired the same way.

### We added regression tests

See [runwayProfileGeometry.test.ts](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/src/utils/__tests__/runwayProfileGeometry.test.ts:217).

The tests now explicitly verify that:

- a transition endpoint with unknown zero altitude is **not** plotted near zero if the next point has valid altitude
- reference marks created from the same route also get a sensible height

That gives us automated evidence that this bug class stays fixed.

### We also found evidence this was not RW32-only

The same bad pattern appears elsewhere in the dataset, for example around:

- `KRDU-R23LY-ADUWON`

Because the new logic is generic, those routes are protected too.

## Important Boundary: What We Can Guarantee

We can confidently say this:

- **isolated or partial unknown altitudes that were previously encoded as zero will no longer create fake low points in the runway profile, as long as the route contains nearby valid altitude information**

That is the class of bug we fixed, and it is covered by tests.

We should not overclaim this:

- if an entire route has **no valid altitude anywhere**, the renderer still cannot invent the true altitude from nothing

In that case, the profile can only show a fallback value, because the source data itself is incomplete.

So the honest guarantee is:

- this specific "unknown altitude exported as zero causes fake dip" problem is now guarded against
- fully missing altitude data is still a source-data limitation, not something the renderer can solve perfectly on its own

## Files Involved

Data evidence:

- [KRDU procedures.geojson](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/public/data/airports/KRDU/procedures.geojson:2038)

Fix implementation:

- [runwayProfileGeometry.ts](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/src/utils/runwayProfileGeometry.ts:280)

Regression coverage:

- [runwayProfileGeometry.test.ts](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/src/utils/__tests__/runwayProfileGeometry.test.ts:217)

## Short Conclusion

The weird RW32 vertical-profile dip was caused by **missing fix altitude being exported as numeric zero**, then rendered as if zero were real geometry.

We fixed it by:

- treating zero-placeholder altitudes as missing
- reconstructing route heights from nearby valid points
- applying the correction generically across all runway-profile routes
- locking it in with tests

So this same failure mode should not recur for any route that has at least some valid altitude information to anchor the repair.
