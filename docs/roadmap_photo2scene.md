# Roadmap: photo2scene completo en el producto (2026-10-11)

## Por qué

El objetivo del sintético (usuario, desde el principio): **escenas que se parezcan a las fotos del usuario, y solo un pequeño porcentaje inventado**. Hoy el producto hace lo contrario:

- El sintético sale casi todo de la plantilla genérica (modelos de internet sobre suelos de Poly Haven), que solo toma números sueltos de las fotos (ángulo, suelo, color). Nadie compara el resultado con las fotos.
- De photo2scene solo está integrada la ruta de soporte, y solo en tareas de contar.
- Caso que lo destapó: `web_local/jobs/grass_border_no_grass` (roi_classification), 38 % (0,353 ± 0,134). Lo que se vio:
  - cielos, troncos y campos con árboles, aunque las fotos miran al suelo;
  - imágenes de un solo color, porque la cámara quedaba dentro de un modelo;
  - césped como un cuadrado sobre otro suelo;
  - una clase "empty" inventada, con el 60 % de los recortes sintéticos;
  - ningún sintético de border.

## Estado para retomar (2026-10-11)

- Hecho: decisiones 0.1 a 0.4 y M.1, M.2; A.0 (medida); A.1 (código y test, sin probar aún en un trabajo nuevo); A.2 (plantilla y etapas de modelos 3D apagadas, fotos obligatorias, motivo en informe y web; hoja `compare/A2_sin_plantilla.html`); A.3 (filtro de parecido en render, `sim2chip/likeness.py`; hoja `compare/A3_filtro_parecido.html`); A.4 (re-medida con 3 ejecuciones por brazo, `scripts/arms_a4.sh`; hoja `compare/A4_remedida.html`). B.1 (análisis de fotos, hoja `compare/B1_analisis_fotos.html`). B.2 (cámara constante, hoja `compare/B2_camara_constante.html`). B.3 (foto más limpia, hoja `compare/B3_foto_limpia.html`, falta revisión del usuario). B.4 (soporte al clasificar, hoja `compare/B4_soporte_clasificar.html`). B.6 (botellas re-medidas, hoja `compare/B6_botellas.html`). M.3 decidido. B.5 hecho y apagado por defecto (no ayuda: 0,771 frente a 0,814, hoja `compare/B5_superficies.html`). C.0 (calle: fotos de roads, contar coches, solo fotos F1 0,651; hoja `compare/C0_calle.html`). C.1 y C.2 (pasos en el paquete, hoja `compare/C1_calle_producto.html`). C.3 (ruta en la etapa). C.4 (control aceptado por ahora; textura localizada pendiente). C.5 (clasificar en calle y habitación, hoja `compare/C5_fase3_clases.html`, falta revisión a ojo). C.6 medido (la calle no aporta: el filtro de parecido descarta sus vistas) y ruta de calle apagada por defecto. Siguiente: fase D. Medir siempre con 3 ejecuciones por brazo (usuario, 2026-10-11) por el ruido de PicoDet (±0,03 a 0,05 con la misma entrada).
- El interruptor es `stages/base.py: TEMPLATE` (False). Los tests de la maquinaria que aún usan la plantilla como ejemplo la encienden con el fixture `template_on`. Ruido entre ejecuciones medido en A.2: unos 0,03 de F1 en las botellas (mismo trabajo, 0,672 y 0,705).
- Pruebas: home `runs/p2s_test` (`/media/lenena-iker/Crucial/runs/eyeforge/p2s_test`). Trabajos `jobs/grass` (copia del de la web local, para no tocar el original), `jobs/bottles_scene` (24 fotos) y `jobs/bottles_few` (8). Resultados en `arms.log`. Los scripts de brazos (`arms.sh`, `arms_grass.sh`) cambian `generation.frames`, `photo_scenes` y `training.seeds` en el spec del trabajo y lanzan `python -m sim2chip --home <home> run <spec> --job <job>` con el Python de `edge-sdg`; copias en `runs/p2s_test/scripts/`.
- Hojas de comparación para el usuario: `runs/p2s_test/compare/` (A.0 en `A0_cesped.html`).
- Sin desplegar: `600d872`, `afb9c86`, `ff6a9a9`, `46b74a0`, `ec640c1` y los de este roadmap. Deploy solo con el visto bueno del usuario.
- Cambios sin commitear que no son de esta línea de trabajo (etapas de modelos 3D, `server.py`, `export.py`, `workspace.py`, web, `firmware/`, `TODO.md`, `.gitignore`, `experiments/e2e/drive.py`, tests de server y de etapas): no mezclarlos en los commits de este roadmap.

## Definiciones

- **photo2scene** son las tres fases del experimento de calles (`experiments/NOTES.md`, 2026-10-07):
  1. **Extracción**: todo lo necesario para imitar la escena (ruta, cámara, máscaras, borrado, profundidad, objetos como modelos 3D, luz).
  2. **Imitación**: el lugar de la foto reconstruido y comprobado contra ella.
  3. **Extrapolación**: el lugar extendido a una escena completa, poblado con las clases del usuario.
- **Ruta de soporte**: un atajo para una superficie vista desde arriba. Cubre la fase 1 entera y una fase 2 simplificada (la foto borrada se queda de fondo y los objetos se recolocan encima). No tiene fase 3.
- **Plantilla genérica**: no es photo2scene. Deja de ser relleno del sintético (decisión 0.2).

## Decisiones

- [x] **0.1 Reparto del sintético** (usuario, 2026-10-11): 90 % imitación (fase 2) y 10 % extrapolación (fase 3). La fase D comprueba que la medida no lo contradice.
- [x] **0.2 Sin ruta no hay sintético** (usuario, 2026-10-11): si ninguna ruta de photo2scene encaja con las fotos, el trabajo entrena solo con ellas. El informe y la web lo dicen y explican por qué. **La plantilla genérica se retira**.
- [x] **0.3 Escribir 0.1 y 0.2 en `CLAUDE.md`** como decisión del usuario, y corregir `docs/plan_photo2scene.md`, que llama "photo2scene en el producto" a la ruta de soporte.

- [x] **0.4 Comparaciones tras cada punto importante** (usuario, 2026-10-11): al cerrar cada punto, una hoja de comparación (antes y después, imágenes y notas) para que el usuario la valore, con su ruta.

## Referencias de medida

Con estos números se compara cada fase. Mismo test y 5 semillas, como en la prueba de las botellas.

| Caso | Solo fotos | Hoy |
|---|---|---|
| Botellas, 24 fotos (contar) | 0,832 | escena de soporte: 0,801; plantilla: 0,777 |
| Botellas, 8 fotos (contar) | 0,672 | escena de soporte: 0,702; plantilla: 0,570 |
| Césped grass/border/no_grass (regiones) | 0,724 ± 0,024 | plantilla: 0,329 ± 0,143 (5 semillas) |
| Calle (por elegir, C.0) | por medir | sin datos |

A.4 (3 ejecuciones por brazo, la referencia buena desde ahora): botellas 24 fotos, solo fotos 0,779 ± 0,011 y escena 0,764 ± 0,031; botellas 8 fotos, solo fotos 0,697 ± 0,013 y escena 0,691 ± 0,023; césped tras A.1, solo fotos 0,814 ± 0,015.

---

## Mapa: qué se quita y qué se conecta

Cómo se retira: **apagar ahora** (deja de ejecutarse, el código sigue) y **borrar tras la fase C** (cuando se sepa qué necesita la extrapolación). Decisiones del usuario del 2026-10-11.

### Quitar

| Pieza | Dónde | Apagar | Borrar |
|---|---|---|---|
| Plantilla road | `sim2chip/scenes/road.py` | [x] (A.2) | [ ] |
| Plantilla generic | `sim2chip/scenes/generic.py`, `skies.py`, `scatter.py` | [x] (A.2) | [ ] |
| Plantilla places (fachada, balcón, habitación) | `sim2chip/scenes/places*.py` | [x] (A.2) | [ ] |
| Render de la plantilla en Isaac (render solo recoge los frames de photo2scene) | `stages/render.py` | [x] (A.2) | [ ] |
| Cielos y texturas de Poly Haven para la plantilla | `fetch_assets.scene_assets`, `polyhaven.py` | [x] (A.2, también el control de etiquetas de Poly Haven de la entrevista) | [ ] |
| Generados desde fotos con TRELLIS v1 (lo sustituye SAM 3D, ver Conectar) | `stages/generate_assets.py`, entorno y kit `trellis` | [x] (A.2; la entrevista ya no pide fotos para 3D) | [ ] |
| Listas manuales de revisión y modos `none`/`manual` (queda solo `auto`) | `QASpec`, `data/usd/exclude.txt`, 4 specs de `specs/` | [ ] | [ ] |
| `sim2chip match` y el ajuste de aspecto (`spec.look`) | `match.py`, `look.py`, `points.py`, `crops.py` | [ ] | [ ] |
| "Solo sintético": sin fotos del usuario no se entrena | entrevista, `spec.py` (`no_photos`, fuera de `validate` para que los trabajos viejos sigan cargando) | [x] (A.2) | n/a |
| Clase de fondo inventada al clasificar con etiquetas del usuario | `interview._drop_background_class` | [x] (A.1) | n/a |
| Campos del spec que solo usa la plantilla (`scene.template`, `weather`, `sky_tags`, `wall_tags`, `railings`, `ground_patches`, `edge_tags`, `distractors`, `generation.frames`, `empty_ratio`, `camera.target_fill`): revisar uno a uno cuáles siguen haciendo falta | `spec.py`, prompt de la entrevista | [ ] | [ ] |

Check de cada apagado: un trabajo de principio a fin sin esa pieza en el registro, y los tests pasan.

### Conectar

| Pieza | De dónde a dónde | Hecho |
|---|---|---|
| Objetos de internet (búsqueda, VLM) | de las etapas de modelos 3D a photo2scene: mezclados con los de la foto en la imitación (M.1) y como sustitutos en las copias de la extrapolación (M.2); la revisión compara con los objetos del usuario | [ ] |
| SAM 3D para fotos del objeto solo | fotos que sube el usuario a objetos de su biblioteca (en lugar de TRELLIS) | [ ] |
| photo2scene en todas las tareas | `scene_from_photo` también para clasificar y regiones, con etiquetas por tarea (B.4) | [x] (ruta de soporte; superficies en B.5) |
| Ruta de superficie como clase | nueva ruta de photo2scene (B.5) | [x] (apagada por defecto: no ayuda todavía) |
| Rutas de calle y habitación | del experimento al paquete del producto (C.1 a C.5); la extrapolación copia en espejo y cambia superficies y objetos (M.2) | [ ] |
| Elección de ruta | del análisis de fotos de la entrevista a photo2scene (C.3) | [ ] |
| Reparto 90/10 | imitación y extrapolación en render (D) | [ ] |
| Filtro de parecido | entre el sintético y el entrenamiento (A.3) | [x] (dentro de render, `likeness.py`) |
| Aviso "sin sintético: motivo" | del trabajo al informe y a la web (A.2, F.2) | [ ] (A.2 hecho: `render.no_synthetic`, `report.json` y aviso en la página; falta F.2) |
| Kits de Nebius | entornos y pesos de photo2scene (E) | [ ] |

### Decisiones abiertas del mapa

- [x] **M.1** (usuario, 2026-10-11) En la imitación entran los objetos de la foto del usuario **y modelos de internet parecidos** de la misma clase (la revisión mira que se parezcan a los suyos, no solo que sean de la clase).
- [x] **M.2** (usuario, 2026-10-11) La extrapolación parte de la escena imitada y la amplía, sobre todo con copias en espejo: una esquina pasa a cruce y un lado de calle a calle completa. En la copia se cambian las superficies (materiales y colores) para que no sea un espejo perfecto, y se pueden cambiar objetos por modelos parecidos de internet.
- [x] **M.3** Superficie como clase (césped): en principio una ruta de photo2scene, pero sus medidas y herramientas (tamaño de objetos, mallas, colocación) puede que no apliquen. Decisiones del usuario (2026-10-11): la imitación es un **reencuadre 3D** (cada foto como suelo 3D con su profundidad, renderizado desde cámaras nuevas dentro del rango medido, sin objetos), y la etiqueta de cada vista sale de las **máscaras** de SAM 3 (regla fijada con las etiquetas a mano del césped como verdad).

## Fase A: dejar de hacer daño (~1 día)

- [x] **A.0 Línea base del césped** (hecho: 0,724 frente a 0,329; hoja en `runs/p2s_test/compare/A0_cesped.html`): el mismo trabajo, solo con fotos, con 5 semillas.
  - Check: queda una cifra anotada en la tabla de arriba.
- [x] **A.1 Sin clase de fondo inventada** (`interview._drop_background_class`; comprobado en A.4 en el trabajo del césped) cuando las clases del usuario cubren toda la foto (clasificar y regiones). Es la regla de `CLAUDE.md` del 2026-09-30.
  - Check: test que demuestra que una spec de regiones con clases de superficie no tiene `background_class`.
  - [x] Check: el trabajo del césped no tiene recortes "empty" (exporta grass, border y no_grass). Efecto lateral: el profesor reetiqueta las fotos y coincide más con las etiquetas a mano del usuario (56 de 69 frente a 51).
- [x] **A.2 La plantilla deja de ser relleno** (decisión 0.2): sin escena válida, no entra sintético de plantilla. Tanto el informe como la web dicen "sin sintético: motivo". Las etapas de modelos 3D de internet también se apagan hasta que M.1 las conecte (usuario, 2026-10-11). Resultado: césped 0,735 ± 0,026 (solo fotos, motivo en la página), botellas 0,778 y 0,672 (sin cambio fuera del ruido). Hoja: `runs/p2s_test/compare/A2_sin_plantilla.html`.
  - [x] Check: test de render con escena desactivada y fotos del usuario, que no debe dar frames de plantilla (`tests/test_template_off.py`).
  - [x] Check: la página del trabajo muestra el motivo (captura sin cabeza en la hoja; falta que el usuario lo vea en su navegador).
- [x] **A.3 Filtro final de parecido**, igual para todas las tareas: cada frame sintético se compara con las fotos de train del usuario mediante embeddings de SigLIP 2 (el profesor que ya existe).
  - Lo que se aleja de la distribución de las fotos no entra en el entrenamiento.
  - El umbral sale de las propias fotos: la distancia que hay entre fotos reales de train.
  - [x] Check: en el césped elimina los cielos, los interiores de modelos y los colores lisos (revisión a ojo con una hoja de aceptados y rechazados). Sobre los 333 frames de plantilla del A.0 deja 1 (umbral 0,783); quedan fuera también céspedes de textura plausibles (0,75-0,77). Falta tu revisión a ojo de la hoja.
  - [ ] Check: en las botellas mantiene la mayoría de los frames de escena (sí: 20 de 20 en ambos), y la nota no baja de 0,801 / 0,702 (no comprobable así: con la misma entrada que A.2 dio 0,779 y 0,659; las ejecuciones repetidas de 8 fotos dan 0,672, 0,705 y 0,659, y 0,801 / 0,702 eran una sola ejecución con 1 frame de plantilla más). Medido en A.4 con 3 ejecuciones: 0,764 y 0,691, lo mismo que solo fotos dentro del ruido.
  - [x] Check: el informe dice cuántos frames entraron y cuántos se quedaron fuera (`report.json`: `synthetic.kept`, `synthetic.left_out`; `render/likeness.json` con la similitud de cada frame).
- [x] **A.4 Re-medir el césped y las botellas** con A.1 a A.3. 3 ejecuciones por brazo de 5 semillas cada una (media ± desviación de las 3 notas). Hoja: `runs/p2s_test/compare/A4_remedida.html`.
  - Botellas, 24 fotos: escena 0,764 ± 0,031 frente a solo fotos 0,779 ± 0,011.
  - Botellas, 8 fotos: escena 0,691 ± 0,023 frente a solo fotos 0,697 ± 0,013.
  - Conclusión: los 20 frames de la ruta de soporte no ayudan ni perjudican de forma medible. Las referencias de una sola ejecución (0,832 y 0,801) eran ruido.
  - [x] Check: el césped no queda por debajo de solo fotos (A.0). Es solo fotos por construcción (sin ruta para regiones): 0,814 ± 0,015. No es comparable con el 0,724 del A.0 porque A.1 cambió las etiquetas del profesor, y con ellas train y test.

## Fase B: ruta de soporte para todas las tareas y superficie como clase (~2 días)

- [x] **B.1 Análisis de fotos, clases correctas**: el VLM mide las clases del usuario y no lo más numeroso de la foto. En el césped midió las hojas: "35 por foto, 0,08-0,75 m".
  - Causa: un dataset de la biblioteca se describe antes de elegir las clases (`scene_profile.GENERAL`) y la entrevista tomaba el tamaño y el número de "lo que destaque" como los de la clase.
  - Arreglo (`scene_profile.py`): en la descripción general, tamaño y número solo por cosa vista (`seen`, ahora con `size_m`); la entrevista sabe que una clase toma el tamaño de una cosa vista solo si es esa clase, y que las superficies no tienen tamaño. Con clases, el VLM responde null para superficies.
  - [x] Check: con el césped, el perfil no propone el tamaño de las hojas para grass (VLM real, 16 fotos: antes 0,08-0,22 m y 30 por foto, ahora nada; las botellas conservan 0,12-0,31 m). Hoja: `runs/p2s_test/compare/B1_analisis_fotos.html`.
- [x] **B.2 Cámara según lo constante que sea** (idea del usuario): se mide cuánto varían el ángulo y la distancia entre fotos.
  - Si varían poco (cámara montada), el rango es estrecho y no se inventan ángulos nuevos.
  - Si varían, se usa el rango medido.
  - Regla (`scene_profile._camera`): montada si al menos el 80 % de las fotos está a ±5° del ángulo mediano y a ±15 % de la altura mediana, las dos cosas. Entonces ángulo mediana ±5°, altura y distancia mediana ±10 %. El 80 % es elección mía: una regla por cuartiles daba por fijo el césped hecho a mano, porque el VLM responde en pasos de 5°.
  - [x] Check: test con descripciones de ángulo constante, que debe dar un rango de ±5° o menos (`test_a_steady_camera_gets_a_narrow_range`: 65-75° con una foto atípica de 50°).
  - [x] Check: el césped hecho a mano conserva su rango (respuestas reales del VLM: césped y botellas sin cambio). Hoja: `runs/p2s_test/compare/B2_camara_constante.html`. Falta probarlo con un dataset real de cámara fija.
- [x] **B.3 Elegir la foto más limpia** para la escena: el VLM puntúa las candidatas (objetos enteros, sin solaparse, poco desorden). Ya no vale la más cercana a la mediana sin más.
  - `SceneFromPhoto._pick` (versión 2 de la etapa): las 6 primeras del orden de antes, puntuadas de 0 a 10 por el modelo de escena; el empate lo rompe el orden de antes. Notas en la caché (`cache/scene_pick/`) por foto y clases. Sin respuesta, se queda el orden de antes.
  - [ ] Check: con las botellas elige una foto que pase la revisión a ojo. 8 fotos: la 41 (7/10), claramente buena. 24 fotos: la 19 (4/10) en vez de la 13 (3/10, botellas pegadas); empata con la 22 y la 24, y la 24 parece algo más limpia. Falta la revisión del usuario en la hoja `runs/p2s_test/compare/B3_foto_limpia.html`.
  - [x] Check: la elección queda en el registro con su motivo ("picked photo 19 (19.jpg): 4/10, ...").
- [x] **B.4 La ruta de soporte en clasificar y regiones**: quitar el límite de solo contar (`scene_from_photo.py:55`). Las etiquetas salen del tipo de tarea:
  - una etiqueta por imagen al clasificar;
  - regiones o puntos según la tarea.
  - Decisiones del usuario (2026-10-11):
    - Objeto o superficie lo dice el VLM. Se pregunta al puntuar las fotos candidatas (B.3, campo `things`), porque un dataset de la biblioteca se describe en la entrevista antes de elegir las clases. Si la mayoría dice superficie, no hay escena y el motivo es "your classes are surfaces or areas...".
    - Al clasificar, una clase por vista (`bg_build.py --groups`, y vistas sin clase si hay clase de fondo, en la proporción `generation.empty_ratio`). Al contar puede haber varias.
    - Prueba de principio a fin con las fotos de botellas, etiquetadas por foto por SigLIP (crema solar frente a botella de agua).
  - Consecuencia mía de la segunda decisión: al clasificar se construye una escena por clase, desde una foto de esa clase. Con una sola escena, las clases que no salen en la foto no tendrían vistas.
  - Fallos encontrados por el camino y corregidos:
    - El profesor llamó crema solar a una foto con solo una figurita, y el VLM le dio 9/10. Ahora cada clase se puntúa con sus palabras, una foto sin la clase va al final, y si una foto no da escena se prueba la siguiente de su clase (`BUILD_TRIES = 2`).
    - Con una sola botella en una mesa grande, las vistas la dejaban cortada o fuera del encuadre, y la revisión rechazó la escena (nota 2). Ahora `bg_build.py` elige primero la cámara y coloca cada objeto donde esa cámara lo ve entero. Esto cambia también las vistas al contar: se mide en B.6.
    - Al parar una ejecución, los procesos de entrenamiento quedan huérfanos con memoria de GPU y la siguiente se queda sin memoria. Pendiente: que `run_stage` cierre su pool al recibir la señal.
  - Visto a ojo y sin arreglar: en la escena de botellas de agua hay cajas sobre cosas que no son botellas (base de una cafetera, un embudo). Es de la construcción de la escena, también al contar.
  - [x] Check: tests de las salidas de cada tarea (`tests/test_scene_pick.py`: vistas de una clase, vista sin clase como fondo o fuera, superficies sin escena, reintento; `test_template_off.py`).
  - [x] Check: un trabajo de clasificar con objetos sobre una mesa llega de principio a fin (`jobs/bottles_classify`, 2 escenas, 40 vistas, 19 y 20 recortes). 3 ejecuciones por brazo: solo fotos 0,925 ± 0,008, con escenas 0,914 ± 0,010, igual dentro del ruido (24 fotos de test, etiquetas del profesor). El césped (regiones) queda como superficie en 17 de 17 fotos y da el motivo. Hoja: `runs/p2s_test/compare/B4_soporte_clasificar.html`.
- [ ] **B.5 Ruta nueva, superficie como clase** (el caso del césped):
  - Imitación: nuevos encuadres y mezclas a partir de las superficies del propio usuario. La fase 1 segmenta cada clase de superficie en sus fotos, y la fase 2 compone vistas nuevas con esas texturas, la luz y la cámara medidas.
  - Extrapolación: bordes nuevos entre las superficies del usuario para border, y proporciones nuevas.
  - Sin objetos de internet.
  - Decisiones del usuario (2026-10-11): reencuadre 3D para la imitación; suelo 3D mezclado para la extrapolación; la ruta **apagada por defecto mientras no ayude** (`scene_from_photo.SURFACE_ROUTE`, `SIM2CHIP_SURFACE_ROUTE=1` la enciende para medir); y nada hecho por caso: las vistas las etiqueta el profesor con lo que aprendió de las correcciones del usuario (la etapa de etiquetado guarda sus centros en `label/teacher.json`).
  - Hecho: `photo2scene/surface.py` con `surf_views.py` (cada foto como suelo 3D de MoGe, cámaras nuevas: giro de hasta 25°, inclinación ±10° dentro del rango medido, altura relativa porque la escala de MoGe no cuadra) y `surf_mix.py` (dos fotos de clases distintas unidas en un suelo por una línea ondulada al azar, con espejo en los bordes). Un reencuadre entra si el profesor le da la clase de su foto; una unión toma la clase del profesor si su confianza llega a `label_min_confidence`. Las uniones son el 10 % (decisión 0.1).
  - Descartado por el camino: una regla aprendida de máscaras de SAM 3 para etiquetar (reconocía border en 1 de 5 fotos de train, y era una pieza hecha para el caso).
  - Resultado (3 ejecuciones por brazo, `scripts/arms_b5.sh`, `arms_b5b.sh`): reencuadres con la regla de máscaras 0,788 ± 0,021 (solo fotos 0,812); forzando la receta de última capa 0,771; reencuadres y bordes con el profesor 0,771 ± 0,012 frente a 0,814 ± 0,015. Ninguna variante ayuda. Lectura: los reencuadres son casi copias que repiten el desequilibrio; el profesor ve como grass muchos reencuadres de fotos de border (border baja a 13 imágenes). Hoja: `runs/p2s_test/compare/B5_superficies.html`.
  - [ ] Check: hoja de muestras revisada a ojo por el usuario (pendiente).
  - [ ] Check: el césped igual o mejor que solo fotos (A.0). No se cumple: 0,771 frente a 0,814; la ruta queda apagada.
- [x] **B.6 Re-medir las botellas** con B.1 a B.4. 3 ejecuciones por brazo (`scripts/arms_b6.sh`). Hoja: `runs/p2s_test/compare/B6_botellas.html`.
  - Botellas, 24 fotos: escena 0,764 ± 0,022 (foto 19, elegida por el VLM) frente a solo fotos 0,795 ± 0,024.
  - Botellas, 8 fotos: escena 0,695 ± 0,017 (foto 41) frente a solo fotos 0,661 ± 0,014.
  - El brazo de solo fotos, con la misma entrada que en A.4, se movió de 0,779 a 0,795 y de 0,697 a 0,661: entre tandas el ruido también llega a 0,03. Ninguna diferencia es firme; con pocas fotos la escena parece ayudar algo y con más no.
  - Visto a ojo: algún objeto sale aún cortado por el borde (la comprobación de B.4 mira base y altura, no el ancho), y hay cajas sobre cosas que no son botellas (cafeteras, vasos negros). Esto último es lo siguiente que más pesa en la calidad de las escenas al contar.
  - [x] Check: no queda por debajo de A.4 (0,764 igual; 0,695 frente a 0,691).

## Fase C: calle e interior completas en el producto (~3-4 días)

- [x] **C.0 Dataset de calle con test real**: elegir uno de un solo dataset (sin mezclar).
  - [x] Check: dataset con fotos de train y de test, y su línea base solo con fotos. Las 74 fotos de `roads` del usuario (en bici, 2026-10-02), contar coches con etiquetas de SAM 3 sin revisar: train 54 fotos (142 coches), test 20 (65). Solo fotos: F1 0,651 (0,638, 0,636 y 0,680 en 3 ejecuciones de 5 semillas), error 1,7 coches por foto. Trabajo `p2s_test/jobs/roads_cars`, hoja `compare/C0_calle.html`.
- [x] **C.1 Llevar las fases 1 a 3 del experimento** a `sim2chip/photo2scene/`, con el mismo sistema de pasos que la ruta de soporte. El producto no importa nada de `experiments/`.
  - Los pasos son `flow_observe`, `flow_route`, `flow_segment`, `flow_check_masks`, `flow_mira_case`, `flow_refine`, `flow_check_meshes`, `piece_surfaces`, `piece_build`, `env_add`, `flow_gates`, `flow_light`, `flow_photometry`, `flow_compare`, `crossroad`, `street_template`, `room_template`, `flow_check_views` y `flow_populate`.
  - Hecho: los pasos se movieron (no copiados) a `sim2chip/photo2scene/steps/` junto con `flow_camera`, `flow_things` y `piece_isaac`, que el flujo también usa; el driver del experimento los llama ahí. `flow_mira_case` y `flow_ccm` se quedan en `experiments/` (solo la variante Mira, que no es la de por defecto). El flujo del producto está en `sim2chip/photo2scene/place.py` (`observe`, `build`, `compose`). Kimi-K3 dejó de aceptar imágenes (2026-10-11): la observación pasa a Qwen3.8-27B (horizonte a 0,06 de MoGe de media en las 9 fotos del banco y home_office; K2.6 a 0,19 y el control de cámara lo rechaza en 6 de 7).
  - [x] Check: las fotos del banco de calle (corner_timber, terrace_row) y home_office dan, desde el paquete del producto, los mismos resultados que en el experimento. corner_timber y home_office construyen y pueblan como en el experimento (vistas que pasan 14 de 16 y 14 de 16, frente a 13 y 15). terrace_row con la observación de K3 del experimento pasa fases 1 y 2 como allí y se para en las vistas de la plantilla (53 %, el experimento dio 59 % y 76 % en dos pasadas); con la observación de Qwen la malla de la casa sale mala y la comparación falla. Las diferencias vienen del VLM entre pasadas, no del código portado (C.4). Hoja `compare/C1_calle_producto.html`.
- [x] **C.2 Entornos**: llevar la geometría del entorno `trellis` al de mallas (no crear un quinto entorno), o justificar por qué no se puede.
  - [x] Check: `missing_pythons()` lista los entornos reales y los tests de los pasos pasan. El entorno de mallas tiene open3d, pyrender, trimesh, scipy y cv2; los pasos de geometría corren ahí (refinado de la casa de terrace_row idéntico en los dos entornos). Tests en `tests/test_place.py`.
- [x] **C.3 Ruta decidida en el producto** a partir del análisis de fotos: soporte, superficie, calle, habitación o ninguna. Una foto fuera de alcance (noche, plaza sin plantilla) queda como fallida con su motivo y el trabajo sigue según 0.2.
  - Hecho: `scene_from_photo` (versión 5) observa la foto elegida (`place.observe`) y la manda a soporte, calle (`street`) o habitación (`room`); fuera de alcance queda `none`, fallida con su motivo, y se prueba la siguiente. Al contar, las fotos a nivel ya tienen ruta (`no_route` no exige cámara hacia abajo); al clasificar, solo soporte y superficie hasta C.5.
  - [x] Check: test de enrutado con las 9 fotos del banco de calle (las 2 de noche quedan fuera). `tests/test_place.py` con las observaciones de Qwen3.8-27B guardadas en `tests/fixtures/street_bench_observe.json`: 7 de día por la ruta de calle con plantilla construida, las 2 de noche fuera.
- [x] **C.4 Calibrar el control de la fase 2** (`flow_compare.py`): hoy cambia de veredicto entre pasadas y aprobó la casa flotante. Se calibra con una hoja etiquetada a ojo.
  - Hoja para etiquetar: `compare/C4_control_imitacion.html` (14 casos, 9 del experimento y 5 del producto). Con 3 pasadas por caso (2026-10-12) el veredicto que bloquea fue estable (los 14 aprueban las 3 veces), pero aprueba el terrace_row sin casa (un bloque liso donde está la hilera): el problema es que aprueba de más. Falta la etiqueta del usuario.
  - Prueba de C.3 con el trabajo de C.0 (`compare/C3_ruta_calle.html`): la foto 47 queda fuera (atardecer) y la 21 va por calle; sus 7 vistas no pasan el filtro de parecido (árboles de bloques, coches montados, falta el muro): F1 0,624 frente a 0,651 solo fotos, sin sintético.
  - Revisión del usuario (2026-10-12): por ahora vale; varios casos no están nada bien y la textura copiada de la foto casi siempre queda descolocada (habrá que aplicarla de forma localizada u otra alternativa). Se deja para la mejora de calidad de los próximos días; el control no se recalibra ahora.
  - [x] Check: acierto anotado frente a la hoja (revisión global del usuario, sin etiqueta por caso).
  - [x] Check: el mismo veredicto en 3 pasadas sobre los mismos casos.
  - C.1 revisado por el usuario: las escenas quedan bien. C.3: flojo por los árboles (poco trabajados), se mejorará.
- [ ] **C.5 Fase 3 con las clases del usuario**: `flow_populate` coloca los objetos del usuario (sacados de sus fotos) y las etiquetas salen según el tipo de tarea.
  - Hecho (2026-10-12): al clasificar, calles y habitaciones también tienen ruta; `flow_populate --groups` pone una clase por vista (o ninguna en la parte de fondo), como la ruta de soporte (B.4). Arreglado un fallo de composición: populate renumera las vistas que pasan el control y la etapa filtraba con la numeración vieja (la foto 21 de roads daba 7 de sus 11 vistas). Hoja `compare/C5_fase3_clases.html`: la calle de la foto 21 con tus coches en dos semillas (11 y 11 vistas, 31 y 30 cajas) y home_office clasificando.
  - Check: hoja de vistas pobladas revisada a ojo (falta el usuario).
  - [x] Check: etiquetas comprobadas en 20 vistas: 22 vistas de calle, cada coche visible con su caja ajustada, ninguna caja de más; los tapados llevan la caja de lo visible. Fallos de colocación (no de etiqueta): coches en la acera y montados, un cielo quemado.
- [ ] **C.6 Medir con C.0**:
  - [x] Check: nota de solo fotos frente a nota con escenas: 0,651 (0,638 / 0,636 / 0,680) frente a 0,662 (0,634 / 0,670 / 0,681), 3 ejecuciones de 5 semillas (2026-10-12, `scripts/arms_c6.sh`, hoja `compare/C6_calle_medida.html`). Las 11 vistas de la calle de la foto 21 no pasan el filtro de parecido (0 de 11): los dos brazos entrenan con lo mismo y la diferencia es ruido.
  - [x] Check: si la calle no ayuda, queda escrito y la ruta se desactiva por defecto. Hoy ni ayuda ni empeora: apagada por defecto (usuario, 2026-10-12). Una foto de calle queda fallida con su motivo; `SIM2CHIP_STREET_ROUTE=1` la enciende para medir cuando mejore la calidad de las calles. La habitación sigue encendida (sin medir). Encendida por defecto en producción (usuario, 2026-10-13), sin haber mejorado aún; `SIM2CHIP_STREET_ROUTE=0` la apaga.

## Fase D: el reparto medido (~0,5 día)

- [ ] **D.1 Proporción de sintético**: probar varios repartos de imitación frente a extrapolación (100/0, 90/10 y 80/20) y varias dosis de sintético por foto real, con las botellas y con C.0.
  - Dosis medida (2026-10-12, hoja `compare/D1_dosis.html`): con 8 fotos el sintético ayuda con 10, 20 o 40 vistas (0,70 frente a 0,66); con 30 fotos, 10 vistas 0,812, 20 vistas 0,764 y solo fotos 0,795, dentro del ruido. Con 40 vistas la revisión del VLM rechazó el juego entero en bottles_scene. El reparto imitación/extrapolación no se puede medir hoy: soporte no tiene extrapolación, calle y superficie están apagadas. Pendiente de decisión del usuario.
  - [x] Check: tabla en `experiments/NOTES.md`.
  - Check: el valor por defecto queda fijado en el código con su porqué.

## Fase E: Nebius (~1-2 días)

- [x] **E.1 Kits**: entornos y pesos de cada paso (SAM 3, SAM 3D Objects, MoGe-2, Qwen-Image-Edit con Lightning y Object-Remover, Isaac) en el bucket de caché, con el sistema actual (tar zstd). Hecho 2026-10-12: `scripts/scene_kits.py` y `sim2chip/photo2scene/cloud.py`.
- [x] **E.2 Qwen en la L40S**: bf16 si cabe en los 48 GB; si no, nf4.
  - Check: memoria y tiempo anotados. bf16 no cabe (32 GB de RAM del nodo); nf4: 31-33 s por foto, 19 GB de GPU (`experiments/NOTES.md`).
- [x] **E.3 De principio a fin en Nebius**: un trabajo con soporte y otro con calle.
  - Check: tiempo y coste por escena anotados, y la escena reutilizada en el segundo trabajo del mismo dataset. Soporte: 21,8 min y 0,34 $ al construir, 7,6 min y 0,20 $ al reutilizar. Calle sin probar (apagada, C.6). Hoja `p2s_test/compare/E_nebius.html`.

## Fase F: web (~1 día)

- [x] **F.1 Página Scenes**: la escena con sus objetos, la ruta y, si falló, por qué.
  - Hecho (2026-10-12, `9a03d99`, hoja `compare/F_web.html`): las fallidas salen con la foto en gris y el motivo. Arreglado de paso: la biblioteca seguía en la versión 5 y borraba la tabla de escenas cada vez que se abría.
- [ ] **F.2 En la página del trabajo**:
  - muestras del sintético que entró y del que el filtro dejó fuera;
  - reparto imitación / extrapolación;
  - "sin sintético: motivo" cuando corresponda.
  - Hecho (2026-10-12, `9a03d99`, hoja `compare/F_web.html`): panel "Synthetic images" con lo que entró, el reparto por fase (soporte siempre imitación; calle y habitación, extrapolación; superficie, imitación salvo los suelos unidos), lo que dejó fuera cada filtro con muestras y el motivo si no entró nada.
  - Check: lo revisa el usuario en el navegador.

## Fase V: vegetación con assets de NVIDIA (decisión del usuario, 2026-10-13)

Los árboles y arbustos salen de SAM 3D como bloques verdes y el filtro de parecido tira las vistas (roads, C.6). En vez de SAM 3D (caro, de 20 s a 2 min de GPU por árbol) o de tarjetas recortadas de la foto, se usan los assets de vegetación del bucket público de NVIDIA (`Assets/Vegetation`: 44 árboles y 37 arbustos; de 24.000 a 87.000 triángulos más ramas instanciadas; en cm y con Z arriba; materiales MDL con hojas recortadas; miniaturas en `.thumbs`). Comparación de opciones en la conversación del 2026-10-13: los assets ahorran la GPU de SAM 3D, se ven bien desde cualquier ángulo y dan variedad gratis en la extrapolación.

- [x] **V.1 Catálogo**: lista de árboles y arbustos con su miniatura, altura y grupo, cacheada; proxy ligero de cada uno (malla simplificada con colores) para los pasos que renderizan con pyrender (controles, etiquetas, mapa de alturas).
- [x] **V.2 Elección** (paso 1.5v, tras el control de máscaras): las máscaras de vegetación no van a SAM 3D; el VLM elige del catálogo, con la hoja de miniaturas y el recorte, la especie más parecida (forma de copa, color, tamaño).
- [x] **V.3 Colocación** (en `env_add`): base en el suelo bajo la máscara y altura medida con MoGe sobre el suelo (con la vertical de las paredes), giro al azar. En `meshes.npz` el proxy y la referencia al USD con su pose.
- [x] **V.4 Isaac**: `piece_isaac` referencia el USD en lugar de la malla, y espera a que cargue; las plantillas de la fase 3 copian la pose con cada copia; en la extrapolación se puede cambiar de especie.
- [x] **V.5 Controles**: `flow_gates` no pide coincidencia de silueta a la vegetación (no es su árbol); `flow_populate` no pone objetos sobre las copas.
- [x] **V.6 Medida** (2026-10-14, `runs/p2s_v6`): con 15 vistas de calle sin filtro, F1 0,651 / 0,623 / 0,636 (media 0,637) frente a ~0,65 solo fotos; el filtro deja pasar 0 de 15. No ayuda todavía. Plan original: hoja de antes y después en la foto 21 de roads y en la 24 de macetas; F1 de roads_cars con 3 ejecuciones (C.6: 0,651 solo fotos, 0,662 con calle).
- [x] **V.7 Nebius**: los USD se referencian por URL pública (Isaac los cachea); los proxies van a la caché del bucket. Hecho y medido el 2026-10-14 (`roads_cars_nb2`): hacía falta pasar a cada paso el fichero de certificados (los Python de los kits no tenían, `5f70663`) y bajar los cielos de Poly Haven a la caché (`28056bf`). Trabajo completo con escena de calle: 51,2 min, 1,32 $; la escena 37,9 min con la GPU al 4 % (descargas, carga de modelos, VLM, `env_add`); CPU media 16 % de 8 núcleos, RAM 27,2 de 32 GB (lo más justo); el filtro de parecido dejó 0 de 13 vistas.

### Hallazgos de la noche del 2026-10-14 (fotos 27 de macetas y 3, 12 y 29 de roads)

- **Velo en el suelo**: con el cielo ×3 el reflejo de Fresnel a ángulo rasante cubría la calzada de un gris uniforme (se perdían flecha y líneas). El suelo pasa a material mate.
- **Calzada quemada en la fase 3**: la fotometría solo oscurecía la ortofoto de la vista 0; la plantilla aplica ahora la misma ganancia (`baked_gain`) a calzada, aceras y césped.
- **Cámaras de la fase 3**: a altura de persona sobre la calzada mirando a lo largo, y solo vistas con algo delante (foto 27: de 62 % a 94 % de vistas válidas).
- **Parcelas**: laterales con la piedra del muro de la foto, descartando recortes con algo delante (un coche).
- **Límite conocido**: SAM 3D convierte un bloque largo que se aleja por la calle en una torre estrecha (roads foto 3), y no extrae pretiles ni barandillas.
- `env_add` tardó 8,5 min la primera vez en una calle arbolada (descarga y proxies de especies nuevas); con la caché caliente, 98 s, de ellos 65 s en las 6 rondas de ajuste con pyrender (sube todas las mallas en cada ronda: se puede acelerar). En Nebius la caché de vegetación de cada trabajo empieza vacía si no se sube al bucket.

### Hallazgos del 2026-10-14 por la tarde (cola de escenas, árboles, esquinas)

- **Cola de escenas** (`8863fba`, sin desplegar): las ejecuciones que renderizan van como tareas a una GPU caliente con Isaac compartida entre trabajos; los pasos pesados se turnan con el cerrojo (`7c283dd`). Medido en Nebius con `roads_cars_nb2` partido en tercios:
  - Intel `1gpu-8vcpu-32gb`, una construcción por máquina: escena 27-31 min con la GPU al 3-7 %, 1,02-1,15 $ por ejecución.
  - Intel con dos a la vez: la máquina se quedó sin latido al cargar los dos los 18 GB del borrador; en 32 GB cabe una.
  - AMD `1gpu-16vcpu-96gb` con tres a la vez: 1,53 $ los tres (0,51 $ cada uno, un 55 % menos), escenas de 31-39 min, RAM media 34 GB. Una vez Nebius no tuvo AMD en 7 min y el respaldo a Intel funcionó.
  - Decisión del usuario (2026-10-14, `679e142`): la máquina se elige al pedir la GPU de escenas; con una ejecución con sintético en marcha, Intel de 32 GB (1 plaza); con 2 o más, AMD de 96 GB (3 plazas), y si no hay AMD, Intel. Una ejecución que no cabe en las plazas libres pide su propia GPU en vez de esperar.
- **Árboles delante de la cámara** (`b32a69c`): la misma foto daba delta E de 24,6 a 43,1 según lo que viera el segmentador; un árbol de relleno quedaba en el camino de la cámara. Ahora se quita la planta de relleno que la foto contradice: las dos construcciones malas pasan a 19,5 y 20,4. Hoja: `p2s_test/compare/arboles_roads3_2026-10-14.jpg`.
- **dd42df6 (luz del suelo en la esquina)**: en la foto 24 de macetas no hace nada (la fotometría se revierte, no hay `baked_gain`); en `terrace_row` la calzada sale algo más clara y azulada, sin cambio claro (vistas que pasan: 41 % antes, 35 % después, ruido del VLM). No hace daño; se puede desplegar con lo demás. Hojas: `p2s_test/compare/esquina_dd42df6_photo24.jpg` y `esquina_dd42df6_terrace.jpg`.
- **Nuevo problema en la esquina**: con `terrace_row` las vistas de la plantilla tienen troncos delante de casi todas las cámaras y sombras muy oscuras; fallan por "floating" y "street_view", y `3.2_extend` tardó 696 s. Medido: los árboles quedan a 8 m o más de las cámaras, pero las cámaras apuntan a la mitad de cada esquina, donde la plantilla copia los mismos 4 árboles de la foto en las 4 manzanas (uno de 17,6 m); y sus proxies tienen hasta 1,8 millones de puntos (de ahí los 696 s). Pendiente: que la vista mire a lo que se ve de la esquina sin un tronco en el centro, y proxies de vegetación más ligeros.
- Arreglados de paso: `rebrick` con selección vacía (`339e82c`, una de las tres escenas de Nebius se perdió por él), escenas antiguas sin árboles de NVIDIA (`d12684c`), Python de photo2scene en la GPU caliente (`994a067`), GPU caliente muda cancelada a los 5 min (`96c08c4`).

## Fase J: Cosmos Reason como juez real frente a sintético (usuario, 2026-10-14)

- [ ] **J.1 Probar Cosmos Reason** (NVIDIA, modelo abierto pensado para razonar sobre física y escenas) como juez que valore el parecido entre la foto real y la escena sintética, y compararlo con los jueces actuales:
  - Jueces actuales: `flow_compare.py` (2.4, preguntas cerradas a Qwen3.8-27B en los dos órdenes), `flow_check_views.py` (3.4b, vistas de la plantilla), el delta E de `flow_photometry.py` y el filtro de parecido (`SIM2CHIP_LIKENESS`).
  - Disponibilidad: hoy no sale en nuestra lista de modelos de Token Factory (ni Cosmos Reason 2 ni Cosmos3-Super-Reasoner). Mirar primero si se puede pedir allí; si no, servirlo en la L40S (la versión de 8B cabe) solo para la prueba.
  - Banco de pruebas: escenas que ya tenemos con veredicto conocido (roads foto 3 antes y después del arreglo de árboles, `p2s_test/var3` y `par/`; la esquina de la foto 24; las 13 vistas de calle que el filtro rechazó en `roads_cars_nb2`; vistas buenas de botellas y HOPE) más la revisión a ojo del usuario como verdad.
  - Medir: acuerdo con el ojo del usuario, estabilidad entre ejecuciones (mismas preguntas varias veces), tiempo y coste por escena, frente a Qwen3.8-27B.
  - Se adopta solo si no empeora (regla de los patrocinadores) y con hoja de comparación para el usuario. Suma además como uso de un modelo de NVIDIA en la entrega.

## Fase G: cierre

- [ ] **G.1 Documentación**: `docs/plan_photo2scene.md`, `TODO.md` y `CLAUDE.md` al día.
- [ ] **G.2 Deploy**: todo lo pendiente junto, solo con el visto bueno del usuario. Están sin desplegar `600d872`, `afb9c86`, `ff6a9a9`, `46b74a0`, `ec640c1` y lo de este roadmap.
- [ ] **G.3 Prueba de punta a punta en producción** con una cuenta de prueba.

## Calendario orientativo

| Fechas | Fase |
|---|---|
| 12 oct | A |
| 13-14 oct | B |
| 15-18 oct | C |
| 19 oct | D |
| 20-21 oct | E |
| 22 oct | F y G |
| 23-30 oct | margen, hardware, vídeo y entrega |

Si C se alarga, se cierra con soporte y superficie (A, B, D, E, F) y la calle queda como experimento documentado.
