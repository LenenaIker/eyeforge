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
- Distancias por contacto con el suelo (usuario, 2026-10-17). El muro izquierdo de corner_cars, recto y continuo, salía torcido.
  - La prueba descarta el cosido: la base del muro se ajusta a un círculo máximo de la panorámica con 0,33° de mediana (unos 2 píxeles).
  - El culpable es MoGe, que acorta las distancias lejanas y oblicuas: 94-96 % de la real delante de la cámara, 75-83 % en los extremos del muro, medido contra el rayo que corta el suelo.
  - Arreglo, métrica de una vista (Criminisi et al. 2000, Hoiem et al. 2006): el suelo pavimentado (calzada, aparcamiento, acera, suelo) se toma plano y su distancia es la del rayo al plano, hasta 25 m. Cada muro, valla o fachada apoyado en él toma, en toda su columna, la distancia horizontal del punto donde su base lo toca. Ese contacto se suaviza a lo largo del muro y los huecos cortos se rellenan.
  - El césped se queda con MoGe (park tiene una loma). Más allá de 25 m, también MoGe: allí un píxel en la base ya son metros.
  - El enderezado por planos sigue después, para lo que el contacto no cubre (los edificios lejanos de court).
  - Hojas: `data/sphere_test/sheet_contact.jpg` y `sheet_contact_leftwall.jpg` (`experiments/sphere/contact_sheet.py`).
- Muros rectos (usuario, 2026-10-17): los muros rectos salían combados un 1-2 % de su distancia, por el sesgo de MoGe en cada vista repartido por la fusión. `geometry.py` busca planos verticales con RANSAC entre los puntos de normal horizontal que SAM 3 llama edificio, muro o valla (`structure` en `stuff_bg.npz`), cada muro como un tramo continuo, y mueve sus puntos al plano a lo largo de su rayo (como mucho un 6 %). Sus puntos pasan de 5-43 cm rms fuera del plano a 0-16 cm. Sin el filtro de SAM tomaba copas de árboles por muros (park). El bordillo de la pista de court (0,2-0,3 m) queda fuera: va con el suelo y su deriva. Hoja: `data/sphere_test/sheet_walls.jpg` (`experiments/sphere/walls_sheet.py`).

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

- [x] Render diferencial generalizado a una cámara que gira (`sphere/render_plan.py` en el entorno principal, `sphere/steps/render.py` en Isaac). Rumbo e inclinación en el marco nivelado. El fondo es el corte de la panorámica borrada a lo largo de los mismos rayos, con los mismos intrínsecos que el render. Pasadas B (sin objetos) y A (con ellos); fondo × A/B y objetos a la exposición que lleva B al fondo.
- [x] El proxy lleva como albedo el color de la panorámica con el color de la luz del cielo dividido (sin eso, el tono del cielo volvía dos veces y los coches salían cálidos y saturados). Recibe sombras y tapa lo que queda detrás. El cociente A/B se calcula sobre las pasadas suavizadas y vale 1 donde B es casi negro (el ruido de dos renders se disparaba).
- [x] Luz: domo del color medio del cielo (Alhaija: la panorámica correcta apenas importa) más el sol cuando está arriba. La posición del sol sale de GPS, hora y rumbo de brújula (`sphere/sun.py`); su intensidad se ajusta en S.5. Las esferas del 16 se hicieron con el sol ya puesto (−0,1 a −1,9°).
- [x] Path tracing por defecto: con luz de cielo sombrea mejor los coches y cuesta 0,3 s por vista en la 3090.
- [x] Retoque de cámara por perfil (`sphere/camera_fx.py`). ESP32-CAM: OV2640 de 66°, RGB565 a QVGA (como lee nuestro firmware), desenfoque, aberración, ruido y dominante. Móvil: ruido ligero y JPEG. Los rangos son una primera estimación: hay que medirlos en las fotos del usuario (S.5).
- [x] Colocación sobre el suelo medido: el centro se busca entre los píxeles de suelo del rumbo, y la huella entera del objeto (rejilla sobre su rectángulo) debe ser suelo y estar a la profundidad esperada a la altura del centro (`render_plan.Ground`). Con el plano fallaba por la deriva del suelo lejano, y con solo ±1 m el coche se metía en bordillos y fachadas.
- [x] Check: `experiments/sphere/s4_sheet.py`, hojas en `data/sphere_test/sheet_s4_*.jpg` (2026-10-16).

Hallazgos y pendientes:
- Los USD de Objaverse convertidos antes del 2026-10-01 llevan rutas absolutas a `datasets/nvidia/usd/raw`, que ya no existe. `render_plan.resolve_asset` escribe una copia con la referencia buena, sin tocar los datos.
- La escala de `assets.json` no vale para los envoltorios (ya están en metros). El render escala cada asset a su longitud real (`size_m`) midiendo su caja.
- La calidad de los assets manda. Varios coches del catálogo local salen grises, de pocos polígonos o transparentes (`5639713f`, `ed4d1dca`, `b8b5705f`). La clase `small_vehicle` incluye carritos de golf. En el producto, los objetivos vienen de `fetch_assets` y `qa_assets` (P.5).
- Park no encuentra sitio para un coche: su suelo es una loma y la huella se comprueba a la altura del centro. Falta inclinar el objeto según el terreno.
- Las sombras de contacto con luz de cielo salen suaves. Hay que compararlas con las reales en la réplica (S.5).

### S.5 Imitación como control (0,5 día)

- [x] Réplica (`sphere/replica.py`, hoja `experiments/sphere/s5_sheet.py`): cada objetivo borrado vuelve a donde estaba como asset de su clase y se renderiza en una vista centrada en él sobre la panorámica borrada; la referencia es la misma vista de la panorámica original.
  - **Dónde:** una caja del tamaño típico de la clase (cada lado entre 0,8 y 1,25 veces) se apoya en el suelo del proxy bajo la máscara (contacto, como los muros de S.2) y se ajusta por IoU de silueta. El suelo del proxy y no un plano: con el plano, los coches de más de 20 m quedaban detrás del proxy (MoGe sube y acerca el suelo lejano). Se descartan ajustes con IoU < 0,7 (coches medio tapados). Un coche reflejado en una puerta de cristal se apoya en el cristal y no encaja: se descarta solo.
  - **Hacia dónde:** la silueta de una caja no distingue un coche de tres cuartos de uno de lado (ponía en fila los coches aparcados en batería) ni el frente de la trasera. Lo da **Orient Anything V2** (`sphere/steps/orient.py`, entorno de SAM 3 más `ORIANY_*` en `.env`) sobre el recorte de cada coche. Su acimut se calibró con renders de rumbo conocido: mediana de 5° de error. La caja gira solo ±15° alrededor de su rumbo.
  - **El frente de cada asset** está donde lo puso su autor: el mismo modelo sobre un primer render dice cuáles salen al revés (4 de 9 en corner_cars) y se giran 180°.
- [x] Medidas (`replica/measure.json`):

| Esfera | Vistas | IoU mediana (render frente a real) | Sombra de contacto real / réplica | Desenfoque que iguala bordes | Ruido añadido |
|---|---|---|---|---|---|
| corner_cars | 13 | 0,67 | 0,47 / 0,89 | 1,0 px | 0 |
| townhouses | 11 | 0,66 | 0,77 / 0,74 | 1,25 px | 0 |

  Court y park no tienen coches.
- [ ] Revisión del VLM de la réplica: pendiente; la comparación numérica ya cubre posición, rumbo y cámara.
- Check: `data/sphere_test/sheet_s5_*.jpg` (corte real, réplica, réplica con la cámara medida).

Hallazgos:
- **Cámara del móvil:** los bordes de los objetos renderizados tienen casi el doble de detalle fino que los reales (0,52 frente a 0,30). Con 1-1,25 px de desenfoque gaussiano se igualan (`replica.match_objects`). El ruido de la panorámica es casi nulo (el móvil la limpia): no hace falta añadirlo. Esto va al perfil `generic` en S.6.
- **Sombra de contacto:** igual en townhouses, más débil en la réplica junto a los coches aparcados en batería de corner_cars. Un domo de cielo cubierto CIE (tres veces más luz en el cenit, `render_plan.overcast_sky`) no la cambia (0,87 frente a 0,88), así que el domo uniforme se queda. Lo que la decide es la forma del asset: con pickups elevados y un Hummer entraba luz bajo la carrocería, y con turismos bajos mejora. Queda abierta hasta tener assets revisados.
- **El brillo de los objetos no se puede comparar:** la pintura del asset no es la del coche real. Se informa, no se ajusta.
- **Assets:** el catálogo local de coches es pobre (antiguos, de dibujos, uno sin texturas). La clase `small_vehicle` entera mete pickups y todoterrenos donde había turismos: la hoja usa solo turismos, taxis y coches de policía de LVIS. En el producto lo decide `qa_assets`. Se añadieron a `data/usd/exclude.txt` los tres assets rotos vistos en S.4.
- `s4_sheet` sumaba `yaw_fix_deg` al rumbo, pero los envoltorios ya llevan ese giro aplicado: dos coches salían de lado. Corregido.
- Isaac referenciaba cada asset una vez: dos objetos con el mismo asset en una vista eran el mismo prim y solo se veía uno. Ahora cada objeto tiene su prim (`key`).

### S.6 Extrapolación (1 día)

- [x] Assets de los objetivos (2026-10-17, `experiments/sphere/car_assets.py`): las etapas de assets del producto (descarga, conversión, vistas previas y QA) sobre `specs/benchmarks/sphere_cars.json`, 160 turismos de Objaverse-LVIS (coche, taxi, policía, monovolumen, descapotable). Las etapas están apagadas en el producto desde que se retiró la plantilla; aquí se fuerzan.
  - Opción nueva `qa.realistic`: el QA descarta modelos de dibujos, de juguete, de videojuego, sin textura o de otra época, porque van sobre fotos reales. Sus veredictos se guardan bajo su propio concepto (`asset_db.review_concept`) y no pisan los del QA normal, que acepta modelos estilizados para cámaras de baja resolución.
  - El VLM local (Cosmos Reason 2 8B) es blando: se quedó 96 de 159 y llamó "actuales" a los coches de 1900. Encima va un filtro de SigLIP 2 por parejas con la frase de la clase ("coche moderno" frente a "coche antiguo", "autobús" y "carrito de golf"): quita 16 más. "Juguete" y "dibujo" ganan en todos los renders por igual y no sirven de filtro. Quedan 78.
  - El frente de cada asset se mide con Orient Anything V2 sobre un render a rumbo conocido: 34 de 69 salen al revés y se giran.
- [x] Vistas (`sphere/extrapolate.py`): rumbo al azar (o alrededor del de la cámara del usuario si lo da), inclinación del spec ±4° acotada para que la vista no salga de la banda capturada (la photosphere llega a unos 28-31° bajo el horizonte; sin el límite se veía el negro), y la cámara ±0,2 m en altura.
- [x] De 1 a 5 objetivos por vista, sobre el suelo medido (huella libre de `render_plan.Ground`), con el 80 % de la huella sobre los tipos de suelo que pisaban los objetivos borrados (si no había, los pavimentados) y a 0,4 m de los demás.
- [x] Orientación: la del objetivo borrado más cercano si está a menos de 6 m (con su frente de S.5, ±8°, y un 30 % girados 180°), si no a lo largo del muro más cercano, si no al azar.
- [x] Etiquetas del simulador: mapa de instancias de Replicator por objeto (`inst_NAME.png`) y cajas ajustadas a lo visible. Un objeto con menos del 30 % de la silueta de su caja visible no se etiqueta.
- [x] Cámara: desenfoque de los objetos medido en S.5 (1-1,25 px) y después el perfil del móvil.
- [x] Filtro de parecido (SigLIP 2, `likeness.py`) contra las fotos sueltas del usuario.
- [x] Revisión del VLM por vista (`sphere/review.py`, Cosmos Reason 2 local, nota 0-10, fuera por debajo de 3 como `bg_review`). Las vistas sin coches no se puntúan: son negativos, y el modelo les ponía 0 a todas por no tener coche. Sus notas son casi binarias (0 o 10), y los ceros caen donde se ve: el todoterreno negro sin textura, el coche de madera, los taxis de aspecto de juguete. Pasan 15, 12, 14 y 14 de 20.
- Check: `data/sphere_test/sheet_s6_*.jpg` (20 vistas por esfera con las cajas, la similitud y la nota del VLM).

| Esfera | Coches colocados | Etiquetados | Vistas que pasan el filtro de parecido |
|---|---|---|---|
| corner_cars | 34 | 28 | 10 de 20 |
| court | 58 | 43 | 3 de 20 |
| townhouses | 53 | 41 | 1 de 20 |
| park | 30 | 23 | 1 de 20 |

Hallazgos:
- Colocación, escala y orientación se ven bien: los coches siguen la calle o la fila de aparcados.
- **Filtro de parecido:** el umbral sale de las fotos del usuario entre sí (la menor similitud con su vecina). Las fotos sueltas del 16 son ráfagas casi iguales y el umbral sube a 0,80: tira casi todo, incluidas vistas sin coches de sitios que salen en las fotos. Con ráfagas, el umbral tiene que salir de fotos distintas (una por sitio) o de otra regla. Pendiente de decidir.
- **Assets:** siguen colándose algunos raros (un coche de madera, un todoterreno negro sin textura, un coche quemado). Los coches salen más vivos de color que el fondo de anochecer.
- **Court es una cancha de baloncesto:** sin coches borrados, los pavimentados permiten ponerlos en la cancha. Es correcto geométricamente, pero no es un sitio de coches; el usuario elegirá esferas de su caso.

### S.6b Contenedores con las mismas esferas (2026-10-17)

Decisión del usuario (2026-10-17): las esferas también para el dataset de basuras, y casi mejor que para coches. Fotos: `data/real/container` (40, móvil, 2026-10-09). Trabajo `bins_count` (`experiments/sphere/bins_job.py`): contar por color (`blue_bin`, `yellow_bin`, `brown_bin`, `gray_bin`, `green_bin`), etiquetas de SAM 3 sin revisar, como C.0. Hoja: `data/sphere_test/bins/sheet_bins_labels.jpg`.

- **Qué hay en las fotos:** cubos con ruedas junto a portales, iglús de vidrio en la calzada, papeleras en poste y, sobre todo, los cubos marrones de 0,4 m del puerta a puerta, el 62 % colgados de paneles. SAM 3 los encuentra bien (55 marcas). Ninguna de las cuatro esferas tiene contenedores.
- **Assets:** buscar por color no da modelos (0 azules, 1 amarillo). Un solo pool de contenedores (`specs/benchmarks/sphere_bins.json`, 92 encontrados, 57 pasan la QA realista) y el color de la clase como pintura al renderizar.
  - **Filtro por las fotos del usuario** (`car_assets.photo_gate`): SigLIP 2 en gris entre los renders y los recortes de sus contenedores; se queda la mitad más parecida (28). Arriba cubos con ruedas y papeleras; en medio dumpsters; abajo cápsulas y bidones.
  - **Altura real de cada modelo** del VLM local (cubo con ruedas 1,05 m, dumpster 1,1 a 1,5 m). Sirve para emparejar forma y tamaño.
- **Contexto medido en las fotos** (`sphere/context.py`, Dvornik et al., ECCV 2018: pegar por contexto ayuda y al azar empeora): MoGe-2 y SAM 3 en cada foto dan el plano del suelo, la base y la altura de cada contenedor sobre él (colgado si la base está a más de 0,2 m), el suelo bajo él, la distancia a la pared más cercana, la distancia a la cámara y el color. Cosmos se probó antes para esto y falló (decía que ningún cubo colgaba).

| Clase | Marcas | Colgados | Suelo (de pie) | Contra pared | Altura mediana |
|---|---|---|---|---|---|
| brown_bin | 29 | 62 % | acera | 75 % | 0,41 m |
| gray_bin | 11 | 50 % | calzada, césped, acera | 0 % | 1,09 m |
| green_bin | 9 | 25 % | acera, calzada | 17 % | 1,42 m |
| blue_bin | 4 | 25 % | acera | 100 % | 1,01 m |
| yellow_bin | 2 | 0 % | acera | 50 % | 0,37 m |

- **Colocación** (`extrapolate.place_by_context`): clase sorteada según cuántas hay de pie en las fotos; altura, una de las medidas; modelo con la altura más parecida entre 4; contra una pared del proxy tantas veces como en las fotos (frente hacia fuera, a la distancia medida del muro); si no hay pared a mano, libre sobre sus tipos de suelo; distancias a la cámara, las de todas las fotos juntas (2,4 a 13,6 m: es cómo encuadra el usuario, no la clase).
- **Colgados fuera por ahora:** la ruta pone objetos en el suelo. Colgarlos de las paredes del proxy queda para después.
- Orient Anything da frente a 1 de 27 contenedores (los ve simétricos): el resto se gira al azar o con el frente por defecto contra la pared.
- Segunda pasada: en el borde de su suelo tantas veces como en las fotos (todas, a menos de 0,6 m de un bordillo, muro o borde del césped), y la pintura como proporción de brillo respecto al suelo junto a él (idea del usuario de la fotometría por proporciones), con el tono del propio contenedor, sobre el albedo del proxy de la esfera. Hojas: `data/sphere_test/sheet_s6_*_bins2.jpg`. Pasan la revisión del VLM 16, 18, 15 y 14 de 20.
- **S.8 con contenedores: las esferas empeoran** (solo fotos 0,361 de media en 4 ejecuciones, con esferas 0,291 en 3; `experiments/NOTES.md`). Solo mejora el verde. Lo que falta: los cubos colgados, la luz del sol y las formas de los cubos pequeños.

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
