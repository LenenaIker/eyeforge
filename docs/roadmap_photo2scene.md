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

- Hecho: decisiones 0.1 a 0.4 y M.1, M.2; A.0 (medida); A.1 (código y test, sin probar aún en un trabajo nuevo); A.2 (plantilla y etapas de modelos 3D apagadas, fotos obligatorias, motivo en informe y web; hoja `compare/A2_sin_plantilla.html`); A.3 (filtro de parecido en render, `sim2chip/likeness.py`; hoja `compare/A3_filtro_parecido.html`); A.4 (re-medida con 3 ejecuciones por brazo, `scripts/arms_a4.sh`; hoja `compare/A4_remedida.html`). Siguiente: B.1. Medir siempre con 3 ejecuciones por brazo (usuario, 2026-10-11) por el ruido de PicoDet (±0,03 a 0,05 con la misma entrada).
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
| photo2scene en todas las tareas | `scene_from_photo` también para clasificar y regiones, con etiquetas por tarea (B.4) | [ ] |
| Ruta de superficie como clase | nueva ruta de photo2scene (B.5) | [ ] |
| Rutas de calle y habitación | del experimento al paquete del producto (C.1 a C.5); la extrapolación copia en espejo y cambia superficies y objetos (M.2) | [ ] |
| Elección de ruta | del análisis de fotos de la entrevista a photo2scene (C.3) | [ ] |
| Reparto 90/10 | imitación y extrapolación en render (D) | [ ] |
| Filtro de parecido | entre el sintético y el entrenamiento (A.3) | [x] (dentro de render, `likeness.py`) |
| Aviso "sin sintético: motivo" | del trabajo al informe y a la web (A.2, F.2) | [ ] (A.2 hecho: `render.no_synthetic`, `report.json` y aviso en la página; falta F.2) |
| Kits de Nebius | entornos y pesos de photo2scene (E) | [ ] |

### Decisiones abiertas del mapa

- [x] **M.1** (usuario, 2026-10-11) En la imitación entran los objetos de la foto del usuario **y modelos de internet parecidos** de la misma clase (la revisión mira que se parezcan a los suyos, no solo que sean de la clase).
- [x] **M.2** (usuario, 2026-10-11) La extrapolación parte de la escena imitada y la amplía, sobre todo con copias en espejo: una esquina pasa a cruce y un lado de calle a calle completa. En la copia se cambian las superficies (materiales y colores) para que no sea un espejo perfecto, y se pueden cambiar objetos por modelos parecidos de internet.
- [ ] **M.3** Superficie como clase (césped): en principio una ruta de photo2scene, pero sus medidas y herramientas (tamaño de objetos, mallas, colocación) puede que no apliquen. Pendiente de pensarlo con el usuario antes de B.5.

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

- [ ] **B.1 Análisis de fotos, clases correctas**: el VLM mide las clases del usuario y no lo más numeroso de la foto. En el césped midió las hojas: "35 por foto, 0,08-0,75 m".
  - Check: con el césped, el perfil no propone el tamaño de las hojas para grass.
- [ ] **B.2 Cámara según lo constante que sea** (idea del usuario): se mide cuánto varían el ángulo y la distancia entre fotos.
  - Si varían poco (cámara montada), el rango es estrecho y no se inventan ángulos nuevos.
  - Si varían, se usa el rango medido.
  - Check: test con descripciones de ángulo constante, que debe dar un rango de ±5° o menos.
  - Check: el césped hecho a mano conserva su rango.
- [ ] **B.3 Elegir la foto más limpia** para la escena: el VLM puntúa las candidatas (objetos enteros, sin solaparse, poco desorden). Ya no vale la más cercana a la mediana sin más.
  - Check: con las botellas elige una foto que pase la revisión a ojo.
  - Check: la elección queda en el registro con su motivo.
- [ ] **B.4 La ruta de soporte en clasificar y regiones**: quitar el límite de solo contar (`scene_from_photo.py:55`). Las etiquetas salen del tipo de tarea:
  - una etiqueta por imagen al clasificar;
  - regiones o puntos según la tarea.
  - Check: tests de las salidas de cada tarea.
  - Check: un trabajo de clasificar con objetos sobre una mesa llega de principio a fin.
- [ ] **B.5 Ruta nueva, superficie como clase** (el caso del césped):
  - Imitación: nuevos encuadres y mezclas a partir de las superficies del propio usuario. La fase 1 segmenta cada clase de superficie en sus fotos, y la fase 2 compone vistas nuevas con esas texturas, la luz y la cámara medidas.
  - Extrapolación: bordes nuevos entre las superficies del usuario para border, y proporciones nuevas.
  - Sin objetos de internet.
  - Check: hoja de muestras revisada a ojo por el usuario.
  - Check: el césped igual o mejor que solo fotos (A.0).
- [ ] **B.6 Re-medir las botellas** con B.1 a B.4.
  - Check: no queda por debajo de A.4.

## Fase C: calle e interior completas en el producto (~3-4 días)

- [ ] **C.0 Dataset de calle con test real**: elegir uno de un solo dataset (sin mezclar).
  - Check: dataset con fotos de train y de test, y su línea base solo con fotos.
- [ ] **C.1 Llevar las fases 1 a 3 del experimento** a `sim2chip/photo2scene/`, con el mismo sistema de pasos que la ruta de soporte. El producto no importa nada de `experiments/`.
  - Los pasos son `flow_observe`, `flow_route`, `flow_segment`, `flow_check_masks`, `flow_mira_case`, `flow_refine`, `flow_check_meshes`, `piece_surfaces`, `piece_build`, `env_add`, `flow_gates`, `flow_light`, `flow_photometry`, `flow_compare`, `crossroad`, `street_template`, `room_template`, `flow_check_views` y `flow_populate`.
  - Check: las fotos del banco de calle (corner_timber, terrace_row) y home_office dan, desde el paquete del producto, los mismos resultados que en el experimento.
- [ ] **C.2 Entornos**: llevar la geometría del entorno `trellis` al de mallas (no crear un quinto entorno), o justificar por qué no se puede.
  - Check: `missing_pythons()` lista los entornos reales y los tests de los pasos pasan.
- [ ] **C.3 Ruta decidida en el producto** a partir del análisis de fotos: soporte, superficie, calle, habitación o ninguna. Una foto fuera de alcance (noche, plaza sin plantilla) queda como fallida con su motivo y el trabajo sigue según 0.2.
  - Check: test de enrutado con las 9 fotos del banco de calle (las 2 de noche quedan fuera).
- [ ] **C.4 Calibrar el control de la fase 2** (`flow_compare.py`): hoy cambia de veredicto entre pasadas y aprobó la casa flotante. Se calibra con una hoja etiquetada a ojo.
  - Check: acierto anotado frente a la hoja.
  - Check: el mismo veredicto en 3 pasadas sobre los mismos casos.
- [ ] **C.5 Fase 3 con las clases del usuario**: `flow_populate` coloca los objetos del usuario (sacados de sus fotos) y las etiquetas salen según el tipo de tarea.
  - Check: hoja de vistas pobladas revisada a ojo.
  - Check: etiquetas comprobadas en 20 vistas.
- [ ] **C.6 Medir con C.0**:
  - Check: nota de solo fotos frente a nota con escenas.
  - Check: si la calle no ayuda, queda escrito y la ruta se desactiva por defecto.

## Fase D: el reparto medido (~0,5 día)

- [ ] **D.1 Proporción de sintético**: probar varios repartos de imitación frente a extrapolación (100/0, 90/10 y 80/20) y varias dosis de sintético por foto real, con las botellas y con C.0.
  - Check: tabla en `experiments/NOTES.md`.
  - Check: el valor por defecto queda fijado en el código con su porqué.

## Fase E: Nebius (~1-2 días)

- [ ] **E.1 Kits**: entornos y pesos de cada paso (SAM 3, SAM 3D Objects, MoGe-2, Qwen-Image-Edit con Lightning y Object-Remover, Isaac) en el bucket de caché, con el sistema actual (tar zstd).
- [ ] **E.2 Qwen en la L40S**: bf16 si cabe en los 48 GB; si no, nf4.
  - Check: memoria y tiempo anotados.
- [ ] **E.3 De principio a fin en Nebius**: un trabajo con soporte y otro con calle.
  - Check: tiempo y coste por escena anotados, y la escena reutilizada en el segundo trabajo del mismo dataset.

## Fase F: web (~1 día)

- [ ] **F.1 Página Scenes**: la escena con sus objetos, la ruta y, si falló, por qué.
- [ ] **F.2 En la página del trabajo**:
  - muestras del sintético que entró y del que el filtro dejó fuera;
  - reparto imitación / extrapolación;
  - "sin sintético: motivo" cuando corresponda.
  - Check: lo revisa el usuario en el navegador.

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
