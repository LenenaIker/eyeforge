# Roadmap: escenas desde photospheres (2026-10-16)

Aprobado por el usuario el 2026-10-16. Se empieza el 17 con S.1 y S.2; ese día el usuario saca photospheres completas, también hacia abajo. Idea del usuario (2026-10-16): pedir una photosphere por cada escena que se quiera y generar la escena a partir de ella, con los objetos cercanos como objetos reales en la simulación y la propia panorámica de fondo para que todo encaje. Este documento dice qué se implementa, qué no y cómo, a partir de lo que ya hemos medido.

## Por qué este cambio tiene sentido

- **Lo único que ha ayudado al alumno es un fondo real.** La ruta de soporte compone objetos 3D sobre la foto del usuario con render diferencial (`bg_isaac.py`): +0,046 en bottles_few, estable con dos composiciones (`docs/investigacion_escena_sdg_2026-10-16.md`). La ruta de calle, que reconstruye el mundo en piezas 3D y lo extiende con plantillas, nunca pasó del ruido (V.6: 0,637 frente a ~0,65 solo fotos).
- **La literatura dice lo mismo.** Alhaija et al. (IJCV 2018) insertan coches 3D en fotos reales de KITTI, con panorámicas de 360° como luz:
  - Fondo: real 49,7 AP50; VKITTI sintético 47,7; Flickr 40,3; negro 21,5.
  - La panorámica correcta como luz apenas importa: 49,7 frente a 49,2 con otra panorámica y 49,1 sin ninguna.
  - El realismo de bajo nivel sí importa: sin retoque de cámara (color, desenfoque, curvas) baja a 43,8.
  - Hasta 5 coches por imagen; con más, empeora.
  - Colocación al azar sobre el plano del suelo igual que a mano; al azar en 3D, peor.
- **La mezcla de mallas de SAM 3D empeora** (POOL, 2026-10-16): lo que limita es la calidad de cada objeto, no la variedad.
- **Encaja con el producto:** las cámaras de los usuarios suelen estar fijas, y una photosphere hecha desde donde irá la cámara es justo su punto de vista.

Conclusión: la photosphere da lo que más pesa (un fondo real en todas las direcciones). El esfuerzo va al encaje del fondo, al retoque de cámara y a la calidad de los objetivos, no a la luz ni a reconstruir el entorno en 3D.

## La idea en una frase

Fondo real (la photosphere, girando desde su centro) + geometría invisible de proxy (profundidad de 360° de MoGe) para sombras, oclusión y apoyo + objetivos 3D compuestos encima con render diferencial. Es el método de Debevec (1998) y el de Alhaija (2018), y generaliza nuestra ruta de soporte de una foto a toda la esfera.

## Decisiones (aprobadas en bloque por el usuario, 2026-10-16)

- [x] **P.1 Ruta nueva dentro de photo2scene, no un pipeline aparte.**
  - Código en `sim2chip/photo2scene/sphere/`, diseñado desde cero para photospheres.
  - Reutiliza `Flow`, caché, biblioteca de escenas, la etapa `scene_from_photo`, kits de Nebius y web.
  - Rehacer esa infraestructura costaría una semana sin ganar nada.
- [x] **P.2 Las tres fases se mantienen, con otro contenido.**
  - **Extracción:** de la photosphere salen la geometría, el fondo borrado y los objetivos.
  - **Imitación:** pasa a ser un control. Se devuelven los objetivos a donde estaban y se compara con la panorámica real; así se ajustan la luz y los colores, como hoy en `bg_isaac`.
  - **Extrapolación:** pasa a ser real. Girar por la esfera y colocar objetivos en sitios nuevos, sin inventar entorno.
  - El reparto 90/10 deja de aplicar en esta ruta.
- [x] **P.3 La ruta de soporte (mesas, superficies) se queda como está**, porque funciona. La esfera sustituye a las rutas de calle y de habitación. Las plantillas de la fase 3 se retiran si la esfera gana en S.8.
- [x] **P.4 Lo cercano y fijo NO se convierte en objeto 3D.** Esto matiza la idea original.
  - Un banco, un bolardo o una fachada se ven mejor como píxeles de la propia foto que como malla de SAM 3D.
  - Solo necesitan geometría para tapar lo que queda detrás, recibir sombras y sostener objetos. Esa geometría sale de la profundidad, sin textura propia.
  - Son 3D solo los objetivos y, más adelante, distractores móviles (personas, bicis, contenedores).
- [x] **P.5 Objetivos 3D: mallas buenas antes que muchas.**
  - Assets de internet de la clase (Objaverse, ya en el pipeline: `fetch_assets`, `qa_assets`, `convert_assets`; decisión M.1).
  - Las mallas de SAM 3D sacadas de las fotos del usuario, solo si pasan un control de malla (`flow_check_meshes`).
  - Alhaija usó 28 modelos de coche.
- [x] **P.6 Cámara: giros desde el centro de captura,** con un desplazamiento pequeño (≤ 0,3 m) sobre la geometría de proxy. Alturas e inclinaciones las del spec, siempre que la photosphere se haya hecho a esa altura. Mover la cámara metros (lo que daría NuRec) queda fuera.
- [x] **P.7 Qué se le pide al usuario:**
  - una photosphere completa (incluida la fila que mira hacia abajo) por sitio, desde donde irá la cámara y a su altura, tantas como quiera;
  - fotos sueltas opcionales.

  El trabajo de calles (roads_cars) mira hasta 47° por debajo del horizonte (inclinación de hasta 15° y FOV vertical de 64°), y las photospheres del 2026-10-16 solo llegan a 27-30° hacia abajo.

## Modelos: qué se queda y qué sale

| Modelo o pieza | Hoy | En la ruta de esfera |
|---|---|---|
| SAM 3 | Máscaras de objetos, suelo y entorno | Se queda: objetivos, móviles, tipos de suelo y cielo, sobre cortes de la esfera, y de vuelta a la panorámica |
| MoGe-2 | Profundidad métrica de cada foto | Se queda, en modo panorámica: el reparto y la fusión de `moge/utils/panorama.py` con los pesos de MoGe-2 (el script de MoGe usa v1). Escala por la altura de captura y el plano del suelo |
| Qwen-Image-Edit + Object-Remover (Lightning) | Borrador | Se queda: borra los objetivos en cada corte y vuelve a la panorámica, fundiendo los solapes |
| SAM 3D Objects | Objetos y entorno (casas, setos) | Solo objetivos sacados de las fotos del usuario, con control de malla. Nada de entorno |
| Cosmos Reason 2 (local) | Observar, planear, revisar | Se queda para observar (la ruta) y revisar las vistas |
| Isaac Sim + render diferencial (`bg_isaac.py`) | Objetos sobre la foto, desde su cámara | Se queda y se generaliza a una cámara que gira. La luz sale de la panorámica, pasada a HDR lineal en un DomeLight, más el sol ajustado |
| SigLIP 2 (filtro de parecido, A.3) | Filtro final | Se queda |
| Assets de vegetación de NVIDIA | Árboles y arbustos | Fuera: los árboles son fondo real |
| Plantillas (`street_template`, `crossroad`, `room_template`, `flow_section`, `section_fill`, `piece_*`, `standing`, `env_add`, `flow_vegetation`) | Fases 2 y 3 de calle y habitación | Fuera de esta ruta. Se borran si la esfera gana |
| Cosmos Transfer, NuRec, Lyra, Marble | Investigados | Fuera. NuRec, como mucho, de demo aparte |

## Pasos

Cada paso cierra con una hoja de comparación para el usuario (JPG, con la vista desde la cámara de la foto). Se desarrolla con las 4 photospheres del 2026-10-16 (`data/real/photospheres-scenes`: corner_cars, court, townhouses y park).

### S.1 Entrada (0,5 día)

- [x] `sphere/capture.py` (sustituye a `photosphere.py`): lee GPano, pone la esfera en su lienzo completo con la máscara de lo capturado y hace cortes con giro exacto. Rechaza lo que no sea una photosphere. Guarda rumbo de brújula, hora y posición aproximada (para el sol).
- [x] Panorámica de trabajo de 4096×2048 y cortes para segmentar: 12 rumbos (cada 30°) y 2 inclinaciones (0° y 35° hacia abajo), 1024×768 con 75° de FOV (la misma densidad que la panorámica), solo los que tienen al menos la mitad capturada. Hecho el 2026-10-16, con tests.
- [x] Check: `experiments/sphere/s1_sheet.py`, hojas en `data/sphere_test/sheet_s1_*.jpg`. Con las esferas del 16 solo salen los 12 cortes a nivel: los de 35° quedan casi fuera de la captura (llega a 27-31° hacia abajo). Los coches cercanos salen cortados por abajo.

### S.2 Geometría (1 día)

- [x] Profundidad de 360° con MoGe-2, repartida y fundida (`sphere/steps/depth.py`): 12 vistas de 90°, lo no capturado fuera de la fusión, escala devuelta a la mediana métrica de las vistas. ~30 s por esfera.
- [x] Suelo y cielo con SAM 3 sobre los cortes, votados de vuelta a la panorámica con peso hacia el centro de cada corte (`sphere/steps/stuff.py`, sirve también para S.3). ~50 s.
- [x] Plano del suelo (RANSAC sobre el suelo de SAM 3 a menos de 10 m): vertical, altura de la cámara y escala (`sphere/geometry.py`).
- [x] Malla de proxy: la profundidad entera como una sola malla, cada vértice sobre el rayo de su píxel (coherente con la panorámica). El suelo a menos de 15 cm del plano se lleva al plano por su rayo. Bajo el borde inferior de la captura se rellena con el plano. Sin cielo ni caras que crucen saltos de profundidad, salvo entre píxeles de suelo.
- [x] Check: `experiments/sphere/s2_sheet.py`, hojas en `data/sphere_test/sheet_s2_*.jpg` (2026-10-16).

Medidas (2026-10-16):

| Esfera | Altura según MoGe-2 | Inclinación | Suelo en el plano | Suelo medido a 10-20 m sobre el plano |
|---|---|---|---|---|
| corner_cars | 1,68 m | 1,2° | 87 % | +32 cm |
| court | 1,81 m | 0,8° | 69 % | +42 cm |
| townhouses | 1,72 m | 1,2° | 81 % | +4 cm |
| park | 1,32 m | 1,9° | 39 % (loma real) | +148 cm (loma) |

- El cosido del Pixel viene nivelado: inclinación de 0,8-1,9°.
- **El suelo lejano sube sobre el plano de forma casi lineal con la distancia** (en la pista plana, +24 cm a 10 m y +126 cm a 30 m). Encaja con un error de ángulo vertical de 2-4° o con una compresión de la distancia lejana por MoGe; con estos datos no se distinguen. Por eso el plano solo se ajusta con el suelo cercano y el proxy usa el suelo medido, coherente con la imagen sea cual sea la causa. Al colocar objetivos (S.6), se apoyan sobre el proxy en su píxel, no sobre el plano.
- Escala: la de MoGe-2 por defecto. El usuario mide 1,85 m y sostiene el móvil a la altura de la cara, así que los 1,7-1,8 m de las tres esferas planas son correctos (usuario, 2026-10-16). Si el usuario da una altura de captura que difiere más de un 10 % de la de MoGe, se usa la suya (usuario, 2026-10-16). Nada de alturas fijas en el código.
- Park da 1,32 m: el plano cercano cae en la loma. Vigilarlo si el terreno inclinado da escalas raras.

### S.3 Segmentación y borrado (1 día)

- [x] SAM 3 sobre los cortes: objetivos (las clases del spec) y móviles (frases del paso de observar; de momento por parámetro), unidos en un objeto por cosa entre cortes, con su corte más centrado y si llega al borde de la captura (`sphere/steps/instances.py`). Tipos de suelo (calzada, aparcamiento, acera, césped, suelo) votados en la panorámica (`stuff.py`), para las zonas de S.6.
- [x] Borrado (`sphere/steps/erase.py`, `sphere/erase_plan.py`): cada objeto se borra una sola vez, en una vista hecha a su medida. Los vecinos van juntos, la vista se centra en el grupo y tiene al menos 768 px. Se pega solo dentro de las máscaras del grupo y de los objetos que lo tocan. Lo que un grupo deja se reintenta solo, en su vista ajustada.
- [x] Tras el borrado, MoGe-2 pasa otra vez sobre el fondo vacío y el proxy sale de esa profundidad: los coches borrados no quedan como geometría invisible. El plano y la escala siguen siendo los de la primera.
- [ ] Lo que el borrador no quite lo etiqueta el profesor en cada vista, nunca una regla aparte (en S.6, con las etiquetas).
- [ ] Los objetivos borrados pasan por SAM 3D y van a la mezcla de objetivos, con control de malla. Pendiente de decidir: la medida POOL del 16 dice que las mallas de SAM 3D empeoran; P.5 pone primero los assets de internet.
- [x] Check: `experiments/sphere/s3_sheet.py`, hojas en `data/sphere_test/sheet_s3_*.jpg` (2026-10-16).

Medidas (2026-10-16, objetivo `car`):
- corner_cars: 24 coches y 4 móviles, 9 vistas, 394 s. Queda 1 coche lejano casi tapado.
- townhouses: 17 coches (1 truncado), 6 vistas, 270 s, nada sin borrar.
- court: 4 móviles, 1 sin borrar. park: 1 móvil.
- Qwen falla con varios coches grandes y pegados en la misma vista (corner_cars: 4 de 7 tras 3 pasadas). Los reintentos de uno en uno lo resuelven.
- Antes de pegar también sobre los vecinos, al borrar un coche delante de una furgoneta Qwen dibujaba la mitad oculta de la furgoneta, y esa mitad quedaba.
- Queda: manchas oscuras de sombra donde estaban los coches cercanos. Posible arreglo: una pasada que nombre las manchas o las sombras.

### S.4 Render (1,5 días)

- [ ] `bg_isaac` generalizado. Cámara con rumbo, inclinación y altura. El fondo es el corte de la panorámica borrada, con los mismos intrínsecos que el render.
- [ ] El proxy recibe sombras (cociente A/B, como ahora) y tapa los objetivos que quedan detrás: la composición respeta la profundidad.
- [ ] Luz: la panorámica en HDR lineal como DomeLight, más el sol ajustado en la réplica.
- [ ] Retoque de cámara según el perfil de hardware: desenfoque, ruido, color, JPEG. Alhaija: 43,8 → 49,7 AP50.
- Check: vistas giradas con un coche puesto a mano en cada esfera.

### S.5 Imitación como control (0,5 día)

- [ ] Réplica: los objetivos borrados vuelven a donde estaban y se renderizan en sus rumbos, frente a los cortes reales.
- [ ] Control fotométrico (`flow_photometry`) y revisión del VLM, como hoy.
- Check: hoja de corte real frente a réplica.

### S.6 Extrapolación (1 día)

- [ ] Vistas: rumbos al azar (con más peso hacia la dirección de la cámara del usuario, si la da), e inclinación y altura del spec con un poco de variación.
- [ ] De 1 a 5 objetivos por vista, sobre el plano del suelo, dentro de las zonas donde la esfera mostraba objetivos o de su mismo tipo de suelo.
- [ ] Orientación según la de los objetivos borrados o la dirección de la calle. Sin solapes y visibles.
- [ ] Etiquetas del simulador; revisión del VLM y filtro de parecido.
- Check: hoja de 20 vistas por esfera.

### S.7 Producto (1 día, solo si S.8 gana)

- [ ] Entrada separada de escenas: `input/places/<sitio>/` con la photosphere, y biblioteca de escenas.
- [ ] Subida mínima en la web, kit de Nebius y la página Scenes.

### S.8 Medida (1 día)

- [ ] Brazos: solo fotos, producto actual y esfera. 3 ejecuciones de 5 semillas por brazo.
- [ ] Test real con etiquetas: roads_cars (C.0). Lo ideal son photospheres hechas en la misma ruta que sus fotos.
- Check: tabla en `experiments/NOTES.md` y decisión.

## Qué no se hace, y por qué

- **Reconstruir el entorno en 3D (casas, árboles, setos):** el fondo real es mejor, y es justo lo que no ha ayudado en la ruta de calle.
- **Coser photospheres a partir de fotos sueltas:** se usa la del móvil. Se puede añadir después con el stitcher de OpenCV.
- **Mover la cámara metros:** necesitaría NuRec/3DGUT, con luz horneada y fuera de las pasadas de datos sintéticos.
- **Reconstruir HDR:** la luz importa poco según Alhaija, así que se usa la panorámica pasada a lineal más el sol ajustado.
- **Mesas y superficies con photosphere:** de cerca, el paralaje de unos centímetros ya rompe el encaje, y la ruta de soporte funciona.

## Riesgos

- **Captura incompleta hacia abajo:** sin la fila de abajo no hay suelo cercano. Lo mitiga P.7 y un aviso en la subida.
- **Costuras y fantasmas del cosido del móvil** (se ven en los cortes del 2026-10-16): pueden enseñar medio coche. Lo mitiga el borrado, y si no basta, las etiquetas del profesor.
- **Profundidad de 360° con escala o curvatura mala:** la escala se corrige con el suelo y la altura de captura, y el check de S.2 lo detecta pronto.
- **Pocas mallas buenas de objetivos:** lo cubren los assets de internet (P.5).
- **Tiempo:** si S.8 no gana el 21, la ruta queda como demo y el producto sigue con la escena actual.

## Calendario

| Día | Trabajo |
|---|---|
| 17 oct | S.1 y S.2 |
| 18 oct | S.3 |
| 19 oct | S.4 |
| 20 oct | S.5 y S.6, con hoja |
| 21 oct | S.8 y decisión |
| 22-23 oct | S.7 si gana; si no, limpieza |
| 24 oct en adelante | Congelar, vídeo, deploy y entrega (30 oct) |

## Fuentes

- [Alhaija et al., Augmented Reality Meets Computer Vision (IJCV 2018)](https://arxiv.org/abs/1708.01566), con los resultados en [ar5iv](https://ar5iv.arxiv.org/html/1708.01566)
- Debevec, Rendering Synthetic Objects into Real Scenes (SIGGRAPH 1998)
- [Isaac Sim 5.1: Neural Volume Rendering](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_nurec.html)
- [DomeLight de USD (formato latlong)](https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/_apidocs/classusdrt_1_1UsdLuxDomeLight.html)
- Medidas propias: `docs/investigacion_escena_sdg_2026-10-16.md`, `docs/roadmap_photo2scene.md` (V.6, D.1)
