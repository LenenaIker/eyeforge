# Plan: fallos de la primera ejecución en PROD (2026-10-28)

Ejecución `container_presence` en eyeforge.app, 00:41. Todo lo de aquí es obligatorio antes de la entrega (usuario, 2026-10-28). Datos leídos en el servidor, solo lectura.

## Qué pasó de verdad (corrige el punto 1 de `TODO.md`)

- El chat mandó `raw: ["library:7b2f…" (Trash, 40 fotos), "<subida del chat>" (40)]`. `adopt_library` metió las 40 fotos del chat **dentro de Trash** (ahora tiene 80) sin preguntar. Además siguen en "Chat photos 28 Oct", así que están duplicadas.
- El usuario pidió **Roads (74) + sus fotos**. El chat eligió Trash por su cuenta, porque la regla "un dataset por modelo" (`interview.LIBRARY`, `adopt_library`) le impedía usar los dos. Roads quedó fuera.
- Profesor: SigLIP 2 aprendió de 12 fotos revisadas (5 con contenedor, 7 vacías) y marcó 14 de 80 como contenedor. SAM 3 contó contenedores en 33 de ellas. En una foto de calle el contenedor ocupa poco espacio, y el embedding de la foto entera describe la calle, no el contenedor.
- Reparto: con 80 fotos, la mitad va a test (`ingest.split`). Por eso 33+33 vacías y 7+7 contenedor. Las 12 revisadas van primero a test (decisión del 2026-10-10), así que el alumno no entrenó con ninguna corrección del usuario, y 28 de las 40 de test solo tenían la marca del profesor. Por eso el 0,917: el test hereda el error del profesor.
- Espera: dos GPU calientes se encendieron al subir fotos y se apagaron sin trabajo (unos 0,31 $). La tercera tardó 248 s en cola. El chat de calibración no tiene botón de parar (`Job.tsx`, `Calibration`, solo un spinner).
- Escenas: PROD no tiene ninguna. `library.for_chat` ya lista las escenas listas por nombre, pero sin escenas no dice nada, y el prompt no explica qué hacer en ese caso. El modelo pidió ids 8 veces.

## Orden de trabajo

El orden sigue las dependencias. Primero va lo que hace que un modelo salga bien, y al final lo más caro y arriesgado.

### A. Profesor de presencia con SAM 3 (unas 3 h)

- **Regla, decidida en código desde el spec (sin ajustes por clase):**
  - clasificar con `background_class: "empty"` (son objetos en la foto) usa SAM 3: la foto es de la clase cuyo mejor objeto supera el umbral, y si ninguno lo supera, `empty`;
  - con `background_class: null` (estados o superficies que llenan la vista) sigue SigLIP 2.
- **Cambios:**
  - `stages/label.py`: `_classify` llama a un camino de presencia que reutiliza `_detections`, `_answer_files` y el umbral de `_count`. El umbral sale de las fotos revisadas (con contenedor contra vacías) y de `spec.errors`. Con varias clases, `_tell_apart` decide cuál es cada objeto.
  - Subir `Label.version`. Ajustar `gpu_needed` y `prefetch`, que hoy salen si la tarea no cuenta.
  - `teacher.py`: las frases de cada clase, con `phrases_of`, igual que en el conteo.
- **Prueba en local antes de desplegar:** un dataset abierto y etiquetado de contenedores en la calle (regla: datasets abiertos para las pruebas). Comparar la precisión de SigLIP con la de SAM 3 contra la verdad, con 12 revisadas. Si no hay uno a mano, Trash con verdad que revise el usuario.

### B. Entrevista honesta y unión de datasets (unas 4 h, incluye lo apuntado para mañana)

- **Se quita "un dataset por modelo".** Con varias fuentes (datasets o fotos del chat), `adopt_library` crea un **dataset nuevo** que las une. Se llama, por ejemplo, "Roads + Trash + chat 28 Oct". Las fotos son enlaces duros, sin espacio extra, y las duplicadas se quitan por hash. Nunca se modifica un dataset existente salvo que el usuario lo pida ("añádelas a Roads").
- **Acción nueva del chat** `{"merge": {"datasets": [...], "name": "..."}}`: crea el dataset unido sin entrenar, para cuando el usuario solo pide unirlos.
- **Etiquetas:** el etiquetado nuevo arranca vacío. Las revisiones del usuario de los datasets de origen pasan a él, por foto (hash) y por clave de clase, cuando la clave coincide. Lo demás lo marca el profesor. Así no se pierde el trabajo del usuario y no se mezclan clases que no casan. *(Pendiente de que lo confirme el usuario: ver Decisiones.)*
- **La verdad la escribe el código, no el modelo:** `summary()` y `_reuse_line` añaden la línea "Photos: Roads (74) + your 40 photos = 114, new dataset …", sacada del spec. El texto del modelo ya no puede contradecirla.
- **Prompt (`LIBRARY`):** reglas nuevas para unir y añadir; nunca decir "no se puede" a unir.
- **Pruebas:**
  - `tests/` de `adopt_library`: dos datasets más una subida dan un dataset nuevo, Trash queda intacto y las revisiones se copian;
  - el banco de la entrevista con el chat de esta noche como caso.

### C. Escenas en el chat (unas 2 h, más la subida)

- **Subir a PROD las 3 escenas públicas** (unos 4,2 GB; quedan 12 GB libres), con `sim2chip scene add <bundle> --owner <admin> --public --name …` en el contenedor. Es escritura en PROD: necesita tu visto bueno.
- **Prompt y código:**
  - el chat nunca pide un id. Recibe las escenas por nombre, con lo que muestran (`scene.json`: contexto, luz);
  - si el usuario dice "elige tú", elige la más parecida a sus fotos según el perfil de Nemotron, y lo dice;
  - sin escenas listas, la línea de biblioteca dice "no scenes yet" y el chat ofrece seguir sin sintético o crear una (punto F).
- `_walk_scenes_problem` acepta el nombre de la escena y lo traduce a su id. `summary()` muestra nombres, no ids.
- Comprobar en el servidor que `SIM2CHIP_IMAGE_TAG` es la etiqueta del 27 (`4b1ac9c4e6d1` o posterior). Sin ella, el render en Nebius instala todo al arrancar.

### D. Profesor sin esperas: precalcular y guardar (unas 4 h)

- **Al subir fotos o abrir un dataset en el chat:** la GPU caliente recibe una tarea real, los embeddings de SigLIP de esas fotos, que no dependen de las clases. Hoy se enciende y espera vacía.
  - Necesita un tipo de tarea por dataset, sin trabajo (`warm.py`, `warm_worker.py`, `nebius.prefetch_warm`).
  - El resultado vuelve a la caché del servidor (`cache/labels/siglip`), por contenido de foto: cualquier trabajo posterior lo encuentra.
- **En cuanto el chat propone un spec (antes de "Teach it"):** prefetch de SAM 3 con las frases de las clases sobre todas las fotos. Usa un trabajo borrador de la sesión; la caché compartida por contenido hace que el trabajo real lo encuentre.
- Con esas respuestas guardadas, la calibración corre en el servidor sin GPU (`_answers_here`). Pasa de minutos a segundos.
- La GPU solo se enciende cuando hay una tarea que mandarle: nunca más "keep-alive" sin trabajo al subir.
- **Medir:** tiempo desde "Teach it" hasta ver las fotos (hoy unos 4 a 5 min, objetivo menos de 10 s), y GPU ociosa en `warm_usage.jsonl`.

### E. Web (unas 2 h)

- **Botón Stop** en `Calibration` (mientras marca) y en `Labeling`:
  - el `cancel` del servidor ya para el proceso y la tarea de la GPU caliente;
  - el estado de calibración pasa a "stopped", no a "failed", con "Try again".
- **Aviso "No synthetic images"** (`Job.tsx:80`): tono informativo nuevo en `Notice` (neutro, sin ámbar ni rojo). Texto: "Trained on your photos only: no scene was chosen."
- **"Reading your photos"** (`Job.tsx:18`, etapa `label`) pasa a "The teacher labels your photos (SAM 3)", y en clasificar por estado "(SigLIP 2)". Así no se confunde con Nemotron.
- En la tarjeta de calibración: qué profesor, cuántas fotos y el tiempo que lleva.

### F. Crear escena desde el chat (unos 1,5 días, el punto 2d del plan F3c)

- **Mover el trabajo de escena a la VM de Nebius**, para que el servidor de Hetzner pueda lanzarlo:
  - entornos de SAM 3, Qwen con el LoRA 360, MapAnything, gsplat, Fixer y 3DGRUT en el disco de pesos;
  - `scene_job.unavailable()` deja de fallar en PROD.
- **Herramienta del chat "crear escena":** ofrece las propuestas de `scene_pick` (Nemotron VL describe, las reglas puntúan, SigLIP quita repetidas) por nombre de foto, con miniatura en el chat. Avisa de unas 2 h y 1 $, y lanza solo tras un sí.
  - El entrenamiento no la espera: el chat avisa cuando la escena está lista.
- **Riesgo:** es lo más grande y lo último. Si no llega al 30, sale marcado "experimental" y el vídeo usa las escenas públicas (CLAUDE.md, regla 3).

### G. Verificación en PROD (unos 3 $ de tope, con tu visto bueno)

1. Desplegar A a E.
2. Repetir el caso de esta noche: Roads + las 40 fotos del chat, "¿hay contenedor?", con "elige tú la escena".
3. **Medir:**
   - tiempos por etapa y espera del profesor;
   - GPU ociosa;
   - precisión del profesor frente a tus revisiones;
   - fotos de test revisadas;
   - F1 del alumno y coste.
4. Informe en `docs/prod_2026-10-29.md`, con los mismos campos que el de esta noche para comparar.

## Decisiones del usuario

1. **Fotos revisadas: ¿test o entrenamiento?** Hoy van todas a test, para tener una nota honesta, y el alumno no aprende de ellas.
   - Propuesta: la mitad de cada clase a test y la otra mitad a entrenamiento.
   - Con pocas revisadas, la nota avisa de que es poco fiable, como ahora.
2. **Etiquetas al unir datasets:** la propuesta de B (copiar tus revisiones cuando la clase coincide y que el profesor marque el resto). ¿Vale?
3. **Fotos adjuntas con un dataset nombrado:** dataset nuevo por defecto, y añadirlas al existente solo si lo pides. ¿Vale? ¿Y deshago lo de Trash (quitar las 40 fotos del chat que se le añadieron)? Esto es escritura en PROD.
4. **Escenas a subir:** las 3 públicas, y con qué cuenta de admin.
5. **Tope de Nebius para G:** 3 $. Lo de F se presupuesta aparte (VM más una escena, unos 3 a 5 $).

## Calendario (entrega el 30 a las 10:00 PDT, 18:00 en España)

| Cuándo | Qué |
|---|---|
| 29 mañana | A y B, con sus pruebas locales |
| 29 tarde | C, D y E; despliegue y G |
| 29 noche y 30 mañana | F; si no llega, sale como experimental. Vídeo en paralelo |

## Estado (2026-10-28, tarde)

Respuestas del usuario a las decisiones:
- **1:** sí a repartir las revisadas entre test y entrenamiento. Además, la ronda de revisión pasa a enseñar 15 fotos como mínimo: el 10 % desde 150 fotos, con un máximo de 40.
- **2 y 4:** sí.
- **3:** "borra directamente los datasets de PROD" (pendiente de confirmar cuáles).
- **5:** tope de 3 $ aprobado.

Hecho (con pruebas; 329 tests en verde):
- **A.**
  - `tasks.finds_objects`: clasificar con "nada" usa SAM 3 (`Label._presence`, `teacher.choose_presence`); los estados que llenan la vista siguen con SigLIP 2.
  - Prueba real en local (Street bins, 40 fotos, sin correcciones): SAM 3 ve contenedor en 27 y SigLIP 2 en 1. En la hoja, SAM 3 acierta en casi todas.
- **Reparto:** la mitad de las revisadas va a entrenamiento (`ingest._share_reviewed`).
- **Ronda de revisión:** `job_service.calibration_photos` (15 mínimo).
- **B.**
  - `library.join_datasets` y `carry_reviews`; `adopt_library` ya no rechaza varios datasets ni cambia uno sin pedirlo (`user_data.add_to`).
  - Acción `join` del chat.
  - La línea "Photos: …" la escribe el código.
- **C (código).**
  - Escenas por nombre; "none yet" cuando no hay ninguna.
  - El chat elige él si el usuario no nombra una y nunca pide ids; el resumen dice nombres (también en la web).
- **D.**
  - `JobService.prefetch_draft`: al subir fotos o abrir un dataset, embeddings de SigLIP 2; al proponer el spec, respuestas de SAM 3.
  - Todo en una carpeta borrador por dueño (`drafts/`); la revisión las lee de la caché.
- **E.**
  - Botón Stop en la revisión y en el etiquetado (estado "stopped", no "failed").
  - Aviso informativo cuando no se pidió sintético.
  - "The teacher labels your photos".
- **F (chat).** Acción `scene_photos`: el chat enseña las fotos propuestas y el usuario crea la escena con un clic. En PROD dice que aún no está disponible mientras falte la VM.
- **Despliegue:** A a E desplegados en eyeforge.app el 28 a las 14:40 (sin commit: el árbol tenía cambios del usuario).

- **Escenas en PROD:** las 3 públicas subidas el 28 a las 15:30, con la cuenta del usuario (`81ef0990522ab527` "Street with a bike lane", `70bb533ff5f57f66` "Country road by a hedge", `1e1a869b293e05ab` "Crossroads at dusk"). Despliegue de nuevo con la acción de escena del chat.

Pendiente:
- F en la VM (necesita presupuesto y decisión);
- G con el usuario;
- qué datasets de PROD se borran;
- commit.
