# Plan: escenas F3c en el pipeline sintético (2026-10-26)

Decisiones del usuario (2026-10-26):
- El trabajo de escena es independiente del entrenamiento. El usuario lo lanza aparte (CLI o web); desde el chat solo si lo pide, y nunca forma parte de un entrenamiento.
- El chat tiene dos caminos: "crear escena" y "entrenar". No es un gestor general.
- Flujo en la web (usuario, 2026-10-26):
  - En la página del dataset salen 6 fotos propuestas para sacar escenas. Las elige el pipeline, como antes, con SigLIP y el modelo de visión.
  - Un botón "Extraer" lanza el trabajo de escena. No hay pantalla nueva: solo un indicador con barra de progreso y un texto del paso en curso.
  - Al acabar aparecen en Assets la escena (Scenes).
  - Después, en el chat, el usuario le pide a Nemotron un modelo con datos sintéticos de esa escena. El entrenamiento coge la escena ya hecha y genera los datos según lo que Nemotron observa en las fotos del dataset, sin esperar a nada.
- Una escena es solo la escena (usuario, 2026-10-26), reutilizable con cualquier clase: los objetos grandes que la complican se borran, como ahora, y no se extrae ni se deja ninguno de ambientación. Los objetos que el modelo tiene que aprender se spawnean en el entrenamiento, con los assets que vienen de otra parte (extracción de objetos, búsqueda, subidas).
- Se quita del pipeline viejo la generación de escena (`scene_from_photo` con sus rutas de soporte, calle y habitación, `scenes/` y lo que solo sirve a eso). El resto se adapta a la escena F3c. No hay prueba aparte: la composición se comprueba en la primera ejecución de punta a punta.
- Alcance: puntos 1 (el entrenamiento usa la escena) y 2 (trabajo de escena). La herramienta del chat queda para después.
- El trabajo de escena corre en local por ahora (la estación hace casi todo y la VM solo WorldStereo). Para la entrega, todo en la VM de Nebius.

## Cómo queda

```
Trabajo de escena (aparte, ~2 h, ~1 $)          Entrenamiento
  foto del sitio -> ruta F3c -> biblioteca  ->  elige escenas listas
                               (solo la escena)         |
                                                       v
                                         colocar objetos + renderizar
                                                       v
                                             render -> train -> export
```

La biblioteca ya tiene `scenes` (ruta, estado) y `objects` ligados a su escena, y el entrenamiento ya recibe las escenas listas en `input/scenes.json` (`library.export_scenes`).

## Punto 2: trabajo de escena

- **2a. Paquete de escena.** Paso `bundle` al final de `ws_route`. Escribe la carpeta que guarda la biblioteca:
  - `splat.usdz` y `splat.ply` de F3c;
  - `scene.json` con unidades, altura de cámara, plano del suelo, centro de los paseos, rejilla del espacio libre y obstáculos (`student.free_space` y `student.obstacles`);
  - `card.jpg` y las puntuaciones.
- **2c. Biblioteca y lanzamiento.**
  - `library.add_scene(owner, photo_id, folder)`, sin trabajo de entrenamiento. Ruta `walk`; estado `building`, `ready` o `failed`.
  - `sim2chip scene <foto>` desde la CLI; en la web, las 6 fotos propuestas y el botón "Extraer" en la página del dataset.
  - Progreso en el sitio (barra y paso en curso); al acabar, la escena en Assets.
- **2d. Para la entrega:** todo el trabajo en la VM RTX PRO 6000. Hay que instalar en su disco los entornos de SAM 3, Qwen (borrador y panorámica, más el LoRA 360 de 282 MB), MapAnything, gsplat, Fixer y 3DGRUT. El servidor de Hetzner lanza la VM y recoge el resultado. Estimación: 1 a 1.5 días.

## Punto 1: el entrenamiento usa la escena

- **1a. Etapa `scene_render`.** Sustituye a `scene_from_photo` para la ruta `walk` y deja `frames.json` en el mismo formato. `render`, `train` y la biblioteca no cambian.
- **1b. Cámaras.** Salen de `student.poses`, con la altura, la inclinación y el campo de visión del spec (la cámara del hardware del usuario).
- **1c. Colocación.** Assets de las clases (los que prepara el pipeline: `convert_assets`, objetos extraídos de las fotos o subidos) sobre el suelo libre de `scene.json`, a la distancia de `camera_distance`, sin obstáculos y con orientación al azar. Se reaprovecha la lógica de `sphere/render_plan.py` y `extrapolate.py`.
- **1d. Render y etiquetas.** NuRec no aparece en profundidad ni segmentación (`investigacion_escena_sdg_2026-10-16.md`), así que van dos pasadas:
  - el fondo con gsplat (color y profundidad);
  - los objetos en Isaac (color, profundidad, máscara y sombra sobre un plano del suelo).
  - Se componen por profundidad, con la composición diferencial de `sphere/steps/render.py`; la caja sale de la máscara visible.
  - Después, `camera_fx` y `expose`.
  - Riesgo principal: se comprueba en la primera ejecución de punta a punta. Si falla, cajas sin oclusión y aviso.
- **1e. Replicator.** Se quita la generación de escena vieja (`scene_from_photo` con sus rutas, `scenes/`) y lo que solo sirve a ella; el resto de `render` se adapta a la escena F3c. Antes de borrar, la lista de lo que sale para que el usuario la vea.
- **1f. Prueba de punta a punta.** Dataset de carretera con una escena F3c: "solo fotos" contra "fotos + escena", con hoja de comparación. Presupuesto aprobado antes de usar Nebius.

## Calendario

| Día | Trabajo |
|---|---|
| 26 tarde | Paquete de escena (2a) y lista de lo que sale del pipeline viejo (1e) |
| 27 | `scene_render` (1a a 1d) y biblioteca (2c) |
| 28 | CLI y botón web (2c), punta a punta (1f) y hoja; empezar la VM (2d) |
| 29 | VM (2d), vídeo y despliegue |

## Hecho: paquete de escena (2a)

Paso `bundle` al final de la receta (`wsroute/bundle.py`, `ws_route.py bundle --case <caso>`). Deja `<caso>/bundle/`:
- `splat.ply` y `splat.usdz` del último brazo, como enlaces duros: no ocupan más.
- `sky.jpg`: la panorámica, que da el color de lo que el splat deja vacío (el cielo).
- `free.npz`: la rejilla del suelo con las celdas ocupadas y las libres, de 0.25 m.
- `card.jpg`: la vista desde el centro de los paseos en la dirección de la foto.
- `scene.json`: las convenciones (metros, z arriba, la cámara de la foto en el origen mirando a +x, el suelo en -1.6), el centro, el radio, la rejilla y las puntuaciones de `grid_final`.

`student.free_space` pasa a usar `free_grid`, que también devuelve la rejilla. Las cámaras salen idénticas: 1048 en photo21, igual que `grid.json`.

Está hecho para photo21, photo47 y photo66. Suelo libre: 171, 521 y 376 m². En la tarjeta de photo66 quedan un coche a la izquierda y dos peatones al fondo. Esa parte de la panorámica no viene de la foto, así que el borrado no la ve.

## Lista 1e: qué sale del pipeline viejo (pendiente de visto bueno)

**Se borra entero**
- `sim2chip/stages/scene_from_photo.py`.
- `sim2chip/photo2scene/`, menos `steps/erase_qwen.py` (lo usa `wsroute`): `__init__.py` (Flow, build, compose, gpu_turn), `place.py`, `surface.py`, `metrics.py` y `cloud.py` (los kits de Python en el bucket), los 40 scripts de `steps/` y los dos `.md`.
- De `photo2scene/sphere/`, todo lo que no se reaprovecha: `capture`, `context`, `erase_plan`, `geometry`, `replica`, `review`, `support`, `sun` y `steps/` menos `render.py`. Lo que sirve se copia adaptado a la etapa nueva antes de borrar:
  - de `steps/render.py`, las dos pasadas, la sombra diferencial y la máscara por instancia;
  - de `render_plan.py`, `resolve_asset` y el tamaño del sensor;
  - de `extrapolate.py`, `overlaps` y `asset_size`;
  - `camera_fx.py` se mueve entero.
- `sim2chip/scenes/`: plantillas (`road`, `generic`, `places`, `places_layout`) y `scatter.py`. `common.py` y `skies.py` se reducen a lo que usa `preview_assets` (`Props`, `look_at`, `make_quad`, `set_visible`, `SKIES`); se mueven a `sim2chip/isaac_util.py` o al propio `preview_assets`.
- `scripts/scene_kits.py`.
- Pruebas que solo prueban lo anterior: `test_place`, `test_places`, `test_scene_pick`, `test_flow_route`, `test_standing`, `test_p2s_cloud`, `test_p2s_metrics`, `test_gpu_turn`, `test_scatter`, `test_template_off` y los `test_sphere_*` de lo borrado. `test_counting` usa `scatter.cluster` (solo lo usa la plantilla): sale esa parte de la prueba.

**Se adapta**
- `stages/__init__.py`: `SceneRender` en el lugar de `SceneFromPhoto`.
- `stages/base.py`: fuera `TEMPLATE` y su comentario.
- `stages/render.py`: se queda con `scene_frames` (lee `scene_render/frames.json`), `like_photos`, `expose` y `no_synthetic`. Sale la plantilla: `template`, `_new_stage`, `_build_scene`, `_camera`, `_load_models`, `real_counts`, `synthetic_frames` y `_Shots`, unas 290 de sus 483 líneas.
- `stages/fetch_assets.py`: fuera `scene_assets`, `_road_skies` y `mixed_ground` (cielos y suelo de la plantilla).
- `spec.py`: fuera `_validate_scene` y los campos de plantilla (`scene.template`, `sky_tags`). `weather` se queda si lo usa el render nuevo para la luz.
- `review.py`: la fase de cada fotograma pasa a ser la escena F3c.
- `library.py`:
  - `take_scenes` sale, porque las escenas ya no las hace un entrenamiento;
  - entran `add_scene` y el estado `building`;
  - `export_scenes` copia solo `splat.ply`, `sky.jpg`, `free.npz` y `scene.json`, sin el USDZ;
  - la tabla `objects` mantiene `scene_id`, que queda sin uso.
- `backends/nebius.py`: fuera `SCENE_KITS`, las exportaciones de `SIM2CHIP_SCENE_PYTHON` y `SIM2CHIP_ERASE_PYTHON` y `scene_from_photo/*` en la subida. La GPU de escena (Isaac) se queda para `scene_render`.
- `server.py`: `PURPOSE_STAGE` sin `scene_*`; entra el lanzamiento del trabajo de escena.
- Web: `Job.tsx` e `Interview.tsx` (el nombre de la etapa), `api.ts` (la ruta `walk`) y `Library.tsx` (las etiquetas de ruta y la tira de objetos de la escena).
- Comentarios que nombran photo2scene en `likeness.py`, `local_vlm.py`, `llm.py`, `warm.py`, `label.py`, `crops.py` y `points.py`.
- `CLAUDE.md`: el punto "Sintético = photo2scene" se reescribe para F3c.

**Propuesta aparte**: mover `experiments/worldstereo/wsroute/` y `ws_route.py` a `sim2chip/walkscene/`, con `erase_qwen.py`. El trabajo de escena es ya producto, y el despliegue en la VM copia `sim2chip/`.

Los documentos viejos (`plan_photo2scene.md`, `roadmap_photo2scene.md`, `roadmap_photosphere.md`) se quedan como historia.

## Hecho el 26 por la tarde y noche: 1a a 1e y 2c

- **1e:** fuera `scene_from_photo`, `photo2scene/`, `scenes/`, la plantilla (`base.TEMPLATE`) y sus pruebas.
  - `erase_qwen.py` pasa a `wsroute/`.
  - Lo que usa la vista previa de assets queda en `sim2chip/isaac_util.py`.
  - La biblioteca sube a la versión 7: borra las escenas de rutas viejas (en local no había ninguna) y añade `public`.
- **Spec:** `generation.scenes` (ids de escenas de la biblioteca). Sin escenas no hay rama sintética, ni tampoco la búsqueda de modelos 3D. Con escenas, cada clase necesita `size_m` y `assets`, y la entrevista los rellena.
- **Pipeline:** `scene_render` entre `qa_assets` y `render`. `render` solo recoge los fotogramas y aplica el filtro de parecido. `crops` y `points` toman los fotogramas con las cajas de su etiqueta.
- **Trabajo de escena:** `sim2chip/scene_job.py` (foto de la biblioteca → `ws_route` hasta `bundle` → biblioteca, con el progreso en `<home>/scene_jobs/<id>.json`).
  - CLI: `sim2chip scene list|make|add|public|private`.
- **Propuestas:** `sim2chip/scene_pick.py`, sin llamadas nuevas al modelo de visión.
  - Lee las descripciones del perfil del dataset (Nemotron VL): exterior, a la altura de los ojos, mirando de frente, de día y con pocos objetos grandes.
  - SigLIP descarta las fotos casi repetidas.
  - Solo entran fotos con focal en el EXIF.
- **Servidor:**
  - `GET /api/library/datasets/{id}/scenes` (propuestas y su escena);
  - `POST /api/library/scenes/extract`, que cobra unos 1 $ en el servidor público;
  - `GET /api/library/scenes/{id}`;
  - las escenas públicas se ven y se pueden usar desde cualquier cuenta;
  - la escena elegida viaja con el trabajo sin el USDZ.
- **Web:**
  - tarjeta "Scenes from these photos" en el dataset, con Extract, barra y paso en curso;
  - Assets > Scenes con las escenas en curso y las públicas.
- **Biblioteca local** (`runs/web_local`): photo21, photo47 y photo66 importadas como públicas, con enlaces duros.
  - Llevan panorámica de FLUX y se quedan así (usuario, 2026-10-27).
- **Panorámica sin FLUX (27):**
  - `pano.py` usa Qwen 360 en el entorno del borrado y añade el paso `pano_paste` en el entorno `scene`. Detalle en `plan_sustituir_flux_2026-10-26.md`.
  - `scene.json` apunta con qué modelo se hizo la panorámica (`pano_model`).
  - Fuera `SIM2CHIP_PANOGEN_PYTHON`.

Audit antes de la primera ejecución (26 por la noche, sin ejecutar):
- la ruta comprueba la salida de cada paso, no paga la VM con un plan vacío y da por hechos los pasos de pago solo con todos los tramos;
- preflight de intérpretes, variables de la nube y boto3 antes del primer paso; `SIM2CHIP_ROUTE_PYTHON` fija el Python del driver;
- candados entre procesos (`sim2chip/locks.py`): la GPU local (pasos de la ruta y etapas GPU del backend local) y la VM de WorldStereo;
- el servidor pone en `failed` las escenas `building` sin proceso vivo al arrancar, y Extract da 503 si faltan los entornos;
- `isaac.py` crea un prim por objeto (contar con un solo modelo ya pone varias copias);
- la inclinación sale del spec, recortada a lo que aguanta el splat.

Pendiente:
- **Nube:** `scene_render` necesita en Nebius el Python de gsplat junto a Isaac. Hecho el 27 (kit `gsplat`, `docs/imagenes.md`; las escenas pasan por el bucket de caché). Kits e imágenes publicados con la etiqueta `4b1ac9c4e6d1`.
  - Probado en Nebius el 27 por la tarde: la ejecución A (`street_bins_nebius`), lanzada por la API de la web con el backend de Nebius, terminó en 42,4 min y unos 1,10 $ en una L40S. `scene_render` tardó 1424 s (en local, 1222 s). PicoDet-S, F1 0,542 ± 0,030 (en local, 0,457 ± 0,056; la verdad es SAM 3 sin revisar).
  - Antes no hubo L40S libres en eu-north1 durante unos 80 min (cuatro peticiones de 5 min). Una sonda de RTX PRO 6000 a las 17:24 consiguió máquina en uk-south2 y eu-south1, pero no en us-central1. Usarlas pediría gsplat compilado para sm_120 y dejar PicoDet fuera (Paddle cu126 no soporta Blackwell).
  - Falta en el servidor de Hetzner: `SIM2CHIP_IMAGE_TAG=4b1ac9c4e6d1`.
- **GPU caliente de escena** (visto el 27, arreglado el 27 por la noche en `warm_worker.py` y `nebius._drop_idle_scene_gpu`, sin probar aún en Nebius: latido cada 30 s durante la precarga, y la GPU de escena se cancela si la ejecución se va a la suya y no le queda otra tarea):
  - manda un solo latido en `setup` y calla mientras precarga los pesos de los profesores, así que a los 120 s (`STALE_S`) la tarea se da por perdida y la ejecución pide su propia GPU;
  - además, el servidor la enciende en cada lanzamiento de una ejecución que renderiza, aunque la ejecución acabe en su propia GPU. Las dos veces se canceló a mano.
- **Dónde se lanza** (27 por la noche, `nebius.platforms`): orden fijo para el mismo estado. Primero las L40S de eu-north1 (sin tráfico entre regiones, la más barata antes), luego las RTX PRO 6000 de eu-south1, uk-south2 y us-central1 por precio; las que no dieron máquina en los últimos 30 min, al final. La distancia a Hetzner no entra (20 a 40 ms en Europa, trabajos por lotes). Un sitio fuera de eu-north1 solo entra si su región tiene proyecto (`SIM2CHIP_NEBIUS_PROJECTS`) y todos los kits que pide la ejecución están probados en sm_120 (`SIM2CHIP_SM120_KITS`, vacío de momento: todo sigue en eu-north1). Fuera de eu-north1 el trabajo usa el proyecto de su región, un token IAM para bajar las imágenes y los secretos por ID (sin probar aún en Nebius); los datos siguen en los buckets de eu-north1 (0,015 $/GiB, unos 0,17 $ por ejecución). Falta el punto 2: gsplat para sm_120, ver si PicoDet tiene Paddle para Blackwell y si TensorFlow 2.18 corre, y una ejecución de prueba en eu-south1.
- **Aspecto de cámara (`look.py`):** `crops` y `points` lo aplican si el spec trae `look`, pero ya nada lo mide. Propuesta: medirlo solo de las fotos de entrenamiento del usuario, sin comando aparte, y comparar con y sin él en la primera ejecución de punta a punta.

Restos de la plantilla, retirados el 26 por la noche:
- fuera `ground_mix.py`, `match.py` (con el comando `sim2chip match`), `polyhaven.py` y la sección `scene` del spec (plantilla, contextos, paredes, barandillas, cielos, suelo, tiempo y distractores). Los specs viejos cargan igual: `from_dict` descarta `scene`, y se quitó de `specs/`;
- la entrevista usa solo el prompt corto. El largo de la plantilla, `check_scene` y los arreglos de Poly Haven salen. Antes, por un fallo, el servidor mandaba el prompt largo aunque la sesión se marcase como corta;
- `scene_profile` deja de proponer campos de escena. Sigue describiendo cada foto (lo usan las propuestas de escena y la altura e inclinación de la cámara en la entrevista) y resume los sitios en `places`;
- en la web, el resumen del spec enseña las escenas de la biblioteca, y la opción de sintético exige una escena elegida;
- `experiments/interview_bench/bench.py` usaba `lean` y `check_scene`: queda como historia y ya no corre tal cual.
