# What the scene builder can and cannot do

We turn ONE real photo into a 3D place in Isaac Sim, then render many new views of it to train a small object
detector. Read this before planning: plan only what the builder can do, and say "out of scope" otherwise.

## Routes

- `village_street_day`: outdoors, daylight (sunny or overcast), camera at pedestrian height (0.5 to 4 m), a
  street, lane or small square of a town or village with buildings, walls or hedges along it. The ground must be
  roughly flat (a gentle slope is fine; stairs or a steep hill are not).
- `support`: indoors or outdoors, the camera looks at one working surface and little else matters: small
  objects on a table top, a desk or one shelf board, a conveyor belt or a workbench seen by a fixed camera, or a
  patch of ground (lawn, soil, floor) seen from above or close. Place `work_surface` or `ground_close`. The cheap
  route: the photo stays the background, only the surface is rebuilt, objects go on it.
- `indoor_floor`: a room seen from inside, furniture and objects on its floor.
- `out_of_scope`: anything else, for example night or dusk, rain or snow on the ground, an aerial or rooftop view,
  a landscape without a street, a crowd covering the ground, a steep street with stairs, water, a close-up of a
  single facade detail, a market under tents.

## Pieces the builder has

- Ground: one plane fitted to the photo's depth, with its material cut from the photo and rectified (asphalt,
  paving stones, sidewalk slabs, gravel, grass). It cannot model steps, ramps or two ground levels except a
  road 12 cm below a sidewalk with a curb between them.
- Facades and walls: flat vertical planes with a material cut from the photo (loses windows and doors), or
  better, whole SAM 3D meshes of the buildings and garden structures seen in the photo.
- Surroundings (SAM 3D, one mesh per segmented instance): works well for whole houses and buildings (windows,
  roofs, chimneys come out), hedges and bushes as blocks, low garden walls and fences as slabs. It fails on
  instances that are tiny, cut by the photo border so only a sliver shows, or merged with what stands behind them
  (a hedge in front of a house). Every surroundings phrase costs about 2 minutes of GPU per instance.
- Movable objects (cars, bikes, scooters, benches, bins, signs, people): segmented and erased from the materials.
  The final place is built WITHOUT them; the detector's objects are added later by a separate generator.
- Far things (more than ~25 m away): low-detail boxes with a picture on each face, or left to the sky image.
- Sky and light: an HDRI sky from one group (below) plus a sun.

## HDRI groups

- `urban`: city streets in daylight, tall buildings and tower blocks around (7 images).
- `countryside`: open fields, hills, coast and roads in daylight (13 images); the sky for village streets.

## Templates to extend the photo into a place seen from anywhere

- `crossroads`: the photo shows one street corner (the street turns or two streets meet, the corner block
  between them). The corner block is copied, turned 90, 180 and 270 degrees, into a crossing of two streets.
- `straight_street`: the photo looks along or across one street with buildings on one or both sides and no
  corner. The visible side is repeated along the street and faced by a copy across it.
- `square`: an open paved square with buildings on several sides.
- `room`: a closed room (indoor routes only).
- `none`: the support route keeps the surface and the photo around it only.
