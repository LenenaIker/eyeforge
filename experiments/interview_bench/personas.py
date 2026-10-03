"""Simulated users for the interview bench.

Each persona has a first message (fixed, so every variant starts the same), hidden facts the simulated user
answers from, photo folders (built as empty .jpg files: the bench stubs the photo analysis), a canned photo
analysis like the one scene_profile returns, and what a right spec must say.

{fx} in texts is replaced by the fixtures folder.
"""

PERSONAS = [
    {
        "id": "lawn", "lang": "Spanish",
        "first": "Quiero que mi robot cortacésped sepa si lo que tiene delante es césped, el borde del césped "
                 "o zona sin hierba. De momento lo quiero probar en el móvil.",
        "facts": "No device yet, you will test on your phone. The camera is on the front of the robot, about 30 cm "
                 "above the ground, looking forward and down. Every picture is one of the three (lawn, border, no "
                 "grass), there is never 'nothing'. You have 120 unsorted photos taken with the robot camera in the "
                 "folder {fx}/lawn. Missing a border is worse than a false alarm (the robot would leave the lawn).",
        "folders": {"training": {"path": "lawn", "loose": 120}},
        "profile": {"contexts": [{"kind": "ground", "share": 1.0, "distance_m": [0.5, 3.0]}],
                    "ground_tags": ["grass"], "weather": ["clear", "cloudy"],
                    "suggest": {"camera_height_m": [0.27, 0.33], "props": ["flower bed", "paving stones", "hose"],
                                "photos": ["IMG_001.jpg: close view of a mowed lawn ending at a gravel path.",
                                           "IMG_002.jpg: lawn edge next to a flower bed with bark mulch.",
                                           "IMG_003.jpg: paving stones of a terrace, no grass."]}},
        "expect": {"task": "roi_classification", "hardware": "generic", "background_class": None,
                   "user_data": "raw"},
    },
    {
        "id": "traffic", "lang": "Spanish",
        "first": "Tengo una ESP32-CAM en un cruce de una carrera ciclista y quiero que me avise si viene un "
                 "vehículo grande o uno pequeño.",
        "facts": "Device: ESP32-CAM. The camera is on a pole about 3 m high at a rural crossroads, looking at the "
                 "road coming towards the crossing, 10 to 40 m away. Big = trucks, buses, vans; small = cars and "
                 "motorbikes (sometimes with people riding them). Day only, any weather. Missing a vehicle is "
                 "dangerous for the riders. You have 40 unsorted photos from the camera in {fx}/cruce, but only "
                 "mention them when asked for photos.",
        "folders": {"training": {"path": "cruce", "loose": 40}},
        "profile": {"contexts": [{"kind": "ground", "share": 1.0, "distance_m": [8.0, 45.0]}],
                    "ground_tags": ["asphalt"], "weather": ["clear", "cloudy"],
                    "suggest": {"size_m": {"measure": "length", "range": [3.5, 12.0]},
                                "camera_height_m": [2.7, 3.3], "props": ["traffic sign", "fence", "hay bale"],
                                "photos": ["CAM_01.jpg: a rural road crossing seen from a pole, a white van far away.",
                                           "CAM_02.jpg: empty asphalt crossing with fields and a traffic sign.",
                                           "CAM_03.jpg: a car and a motorbike approaching on a cloudy day."]}},
        "expect": {"task": "roi_classification", "hardware": "esp32cam", "user_data": "raw"},
    },
    {
        "id": "pots", "lang": "Spanish",
        "first": "Quiero contar macetas en fotos hechas con el móvil, en balcones y fachadas. No tengo placa, "
                 "es para probar en el móvil. Mis fotos sin ordenar están en {fx}/macetas y en la subcarpeta "
                 "{fx}/macetas/modelos hay 6 fotos de macetas concretas para hacer modelos 3D.",
        "facts": "Phone only. Photos taken from the street and from balconies. A pot counts even if it is empty. "
                 "Hanging pots on walls do not matter.",
        "folders": {"training": {"path": "macetas", "loose": 31, "subs": {"modelos": 6}}},
        "profile": {"contexts": [{"kind": "facade", "share": 0.7, "distance_m": [4.0, 14.0],
                                  "supports": ["balcony_floor", "railing", "window_sill"]},
                                 {"kind": "balcony", "share": 0.3}],
                    "ground_tags": ["concrete"], "weather": ["clear"], "wall_tags": ["plaster", "brick"],
                    "railings": ["bars"],
                    "suggest": {"size_m": {"measure": "length", "range": [0.2, 0.45]},
                                "camera_height_m": [1.4, 1.8], "objects_per_photo": {"median": 5, "max": 14},
                                "props": ["patio chair", "small table", "bicycle"],
                                "photos": ["PXL_1.jpg: a plaster building front with pots on three balconies.",
                                           "PXL_2.jpg: a balcony seen from close with six pots on the floor.",
                                           "PXL_3.jpg: brick facade, pots on window sills, sunny."]}},
        "expect": {"task": "count", "hardware": "generic", "user_data": "raw", "names": True},
    },
    {
        "id": "boar", "lang": "English",
        "first": "Hi, I'm a biologist studying wild boar in a beech forest. I want to know when a boar walks past "
                 "my camera trap. I have an ESP32-S3 board with a camera.",
        "facts": "Device: ESP32-S3. Camera on a tree trunk at about 1 m, looking at a path 2 to 10 m away. Day "
                 "and night (infrared at night). Other animals (deer, foxes) also pass and should not count as "
                 "boar. Missing a boar is worse than a false alarm. You have 300 photos already sorted into "
                 "subfolders boar/ and nothing/ in {fx}/trap, but only mention them when asked for photos.",
        "folders": {"training": {"path": "trap", "subs": {"boar": 140, "nothing": 160}}},
        "profile": {"contexts": [{"kind": "ground", "share": 1.0, "distance_m": [2.0, 10.0]}],
                    "ground_tags": ["leaves"], "weather": ["cloudy", "night"],
                    "suggest": {"size_m": {"measure": "length", "range": [1.0, 1.6]},
                                "camera_height_m": [0.9, 1.1], "props": ["log", "fern", "rock"],
                                "photos": ["trap_0012.jpg: a wild boar crossing a forest path covered in leaves.",
                                           "trap_0047.jpg: infrared night view of an empty path between beeches.",
                                           "trap_0103.jpg: a roe deer near a fallen log."]}},
        "expect": {"task": "roi_classification", "hardware": "esp32s3", "user_data": "labeled",
                   "errors": "find_more"},
    },
    {
        "id": "excavator", "lang": "English",
        "first": "I manage a construction site and want to count how many excavators and dump trucks are working "
                 "at any time. We have a Jetson Nano with a camera on a mast.",
        "facts": "Device: Jetson Nano. Mast about 8 m high, machines 10 to 80 m away. You have no photos at all "
                 "and cannot take any this month; if asked, say so. Day only.",
        "folders": {},
        "profile": None,
        "expect": {"task": "count", "hardware": "jetson_nano", "user_data": None},
    },
    {
        "id": "bees", "lang": "French",
        "first": "Bonjour, je voudrais compter les abeilles qui sortent de ma ruche avec une petite caméra. Je "
                 "n'ai pas encore choisi le matériel, je testerai sur mon PC.",
        "facts": "No device yet, testing on a PC. Camera 20 cm in front of the hive entrance, looking at the "
                 "landing board. You have 25 unsorted photos in {fx}/ruche; only mention them when asked for "
                 "photos. Usually 5 to 40 bees per photo.",
        "folders": {"training": {"path": "ruche", "loose": 25}},
        "profile": {"contexts": [{"kind": "ground", "share": 1.0, "distance_m": [0.15, 0.3]}],
                    "ground_tags": ["wood"], "weather": ["clear"],
                    "suggest": {"size_m": {"measure": "length", "range": [0.012, 0.016]},
                                "camera_height_m": [0.15, 0.25], "objects_per_photo": {"median": 14, "max": 38},
                                "props": [],
                                "photos": ["ruche1.jpg: close view of a wooden hive entrance with about 15 bees.",
                                           "ruche2.jpg: landing board crowded with bees in the sun."]}},
        "expect": {"task": "count", "hardware": "generic", "user_data": "raw", "names": True},
    },
    {
        "id": "cookies", "lang": "English",
        "first": "We bake cookies and want a camera above the conveyor belt to tell good cookies from cracked "
                 "ones. It will run on a normal PC next to the line. Photos sorted by class are in {fx}/cookies.",
        "facts": "PC. Camera straight above the belt, 40 cm high, cookies pass one by one. A false alarm throws "
                 "away a good cookie, missing a cracked one is worse (customer complaints).",
        "folders": {"training": {"path": "cookies", "subs": {"good": 200, "cracked": 80}}},
        "profile": {"contexts": [{"kind": "ground", "share": 1.0, "distance_m": [0.35, 0.45]}],
                    "ground_tags": ["fabric"], "weather": ["clear"],
                    "suggest": {"size_m": {"measure": "length", "range": [0.06, 0.08]}, "indoor": True,
                                "camera_height_m": [0.36, 0.44], "props": [],
                                "photos": ["c_001.jpg: one round cookie on a grey belt, seen from above.",
                                           "c_150.jpg: a cookie with a crack across it, from above."]}},
        "expect": {"task": "roi_classification", "hardware": "generic", "user_data": "labeled",
                   "orientation": "any"},
    },
    {
        "id": "cat", "lang": "Spanish",
        "first": "Quiero una gatera que solo se abra para mi gata, con una ESP32-CAM. Solo con mis fotos, sin "
                 "imágenes sintéticas.",
        "facts": "Device: ESP32-CAM. Classes: your cat (Luna, grey tabby), other cats, nothing. The camera is "
                 "inside the door at 20 cm, the cat is 10 to 50 cm away. Opening for another cat is worse than "
                 "keeping Luna waiting. You have 90 unsorted photos in {fx}/gatera; mention them when asked. "
                 "You do not want synthetic images, ever.",
        "folders": {"training": {"path": "gatera", "loose": 90}},
        "profile": {"contexts": [{"kind": "indoor", "share": 1.0, "distance_m": [0.1, 0.5]}],
                    "ground_tags": ["tiles"], "weather": ["clear"],
                    "suggest": {"size_m": {"measure": "length", "range": [0.4, 0.5]}, "indoor": True,
                                "camera_height_m": [0.18, 0.22], "props": [],
                                "photos": ["g1.jpg: a grey tabby cat face very close to the camera at night.",
                                           "g2.jpg: an empty tiled floor outside a cat flap.",
                                           "g3.jpg: a black cat approaching the door."]}},
        "expect": {"task": "roi_classification", "hardware": "esp32cam", "user_data": "raw", "frames": 0,
                   "errors": "fewer_false"},
    },
    {
        "id": "parking", "lang": "English",
        "first": "Can I count the cars in my shop's parking lot with a camera on the roof? Just testing on my "
                 "laptop for now.",
        "facts": "Laptop only. Roof camera about 6 m high looking down at an angle, 20 spaces, cars 5 to 40 m "
                 "away. Day and night (lamps). No photos yet; if asked, say you can take some next week but want "
                 "to start now.",
        "folders": {},
        "profile": None,
        "expect": {"task": "count", "hardware": "generic", "user_data": None},
    },
    {
        "id": "chickens", "lang": "German",
        "first": "Hallo, ich möchte die Hühner in meinem Stall zählen, mit einem ESP32-S3 und Kamera an der Decke.",
        "facts": "Device: ESP32-S3. Camera on the ceiling 2.5 m high looking straight down at the straw floor. "
                 "8 to 15 chickens. You have 50 unsorted photos in {fx}/stall; mention them when asked.",
        "folders": {"training": {"path": "stall", "loose": 50}},
        "profile": {"contexts": [{"kind": "indoor", "share": 1.0, "distance_m": [2.0, 3.0]}],
                    "ground_tags": ["straw"], "weather": ["clear"],
                    "suggest": {"size_m": {"measure": "length", "range": [0.35, 0.5]}, "indoor": True,
                                "camera_height_m": [2.3, 2.7], "objects_per_photo": {"median": 10, "max": 15},
                                "props": ["feeder", "water trough"],
                                "photos": ["s1.jpg: top view of a straw floor with ten brown hens.",
                                           "s2.jpg: hens around a red feeder seen from the ceiling."]}},
        "expect": {"task": "count", "hardware": "esp32s3", "user_data": "raw", "names": True,
                   "orientation": "any"},
    },
]
