# Known failures of the scene builder

Each entry: the defect, how it shows, how it is caught, the fix. Grows only from defects confirmed by eye or by
the student; never from a model's guess.

1. Floating objects. An object hangs above the ground or the support, with no contact shadow. Caught by the
   contact gate (lowest mesh point to its support plane). Fix: snap it down, or drop the mesh.
2. Sky on its side. The HDRI dome is turned, so the horizon is vertical or the ground shows sky texture (seen
   2026-10-06: dome pole on +Y while RTX expects +Z). Caught by "is the sky above the ground?". Fix: dome pole
   on +Z.
3. Camera inside a wall. A view shows one flat texture or the inside of a mesh. Caught by the clearance gate
   (distance from the camera to every surface) and "does the view show a street?". Fix: move or drop the camera.
4. Mixed mask. One SAM 3 instance joins two things (a hedge with the house behind it), so SAM 3D makes half a
   hedge, half a house. Caught by "is the mask only X?". Fix: drop the instance or segment with another phrase.
5. Dull exposure. Whole views grey and flat because the exposure aimed at the median with the sun in front.
   Caught by the luminance gate (p97 near 0.9, median 0.32 to 0.58). Fix: expose by the highlights.
6. Wrong sky. A desert or rock HDRI behind a town street. Caught by "does the sky fit the place?". Fix: pick the
   HDRI group from the route (urban for streets).
7. Tiny or sliver instance. A pillar or a building cut by the photo border gives a degenerate mesh. Caught by
   mask area and border-touch share before SAM 3D. Fix: skip the instance.
