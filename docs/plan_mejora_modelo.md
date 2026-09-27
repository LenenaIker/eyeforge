# Plan: mejorar un modelo ya entrenado (fotos nuevas, captura en vivo, versiones, retocar el spec)

## Contexto

El primer modelo de un usuario probablemente no será bueno o será "decente pero no suficiente". Ejemplo: el robot de GreatScott! falla en las calvas secas del césped. El flujo buscado es:

1. Ver dónde falla, en la prueba en vivo o en la revisión.
2. Sacar fotos de eso.
3. Reentrenar sin repetir el sintético, que ya es bueno.
4. Ver "antes X %, ahora Y %" medido sobre las mismas fotos.

También debe poderse volver a la entrevista para cambiar el spec y relanzar como versión nueva.

Decisiones del usuario (2026-09-27):
- Entran las cuatro piezas: añadir fotos, capturar desde la prueba en vivo, versiones con antes/después y retocar el spec.
- Las fotos nuevas se reparten mitad y mitad entre entreno y test. El test que ya existía queda fijo.

Hallazgo que lo condiciona: `Pipeline.plan` (`sim2chip/pipeline.py`) trata `--from X` como "X y todas las etapas que vienen después en la lista".
- La revisión ya relanza con `--from label` (`review.apply` devuelve "label").
- Por eso hoy cada "Apply and retrain" repite fetch, convert, qa, render y crops: unos 12 min de L40S que sobran.
- Arreglarlo es la base de todo lo demás.

## Fase 1. Relanzar solo lo que depende de verdad (pipeline)

`sim2chip/pipeline.py`, `plan()`:
1. **`start` fuerza la etapa X y sus descendientes en el grafo, no el resto de la lista.**
   - Los descendientes se sacan invirtiendo `after`. Ya existe `_deps()` para el sentido contrario: añadir `_dependents()`.
   - Las etapas que no dependen de X se saltan si su huella está al día, como siempre.
   - `force` sigue forzándolo todo.
2. **Etapas que no se arrastran.** Nuevo atributo `Stage.follows_upstream = True` (en `sim2chip/stages/base.py`), que en `GenerateAssets` vale False.
   - Un relanzamiento de `label` no fuerza ni generate_assets ni lo que va detrás de él (fetch, convert, qa, render...).
   - Motivo: sus modelos 3D están en caché por contenido de foto, y las versiones deben usar el mismo sintético para poder compararse.
   - Solo se rehace si cambia su propia huella o si se pide por nombre.
   - Las huellas no cambian, así que los trabajos existentes no se invalidan.
3. Resultado:
   - Corregir marcas o añadir fotos relanza label, ingest, train, evaluate y export.
   - Decidir sobre assets relanza qa_assets y lo que depende de él. `ingest` y `label` quedan fuera.

Tests en `tests/test_pipeline*.py`, con etapas falsas:
- `--from label` no relanza render;
- con generate_assets activo, tampoco;
- `--from qa_assets` no relanza ingest;
- `force` lo relanza todo.

## Fase 2. Reparto estable entre entreno y test (ingest)

`sim2chip/stages/ingest.py`:
- **Reparto guardado.** Nueva salida `ingest/split.json` con `{content_key: "train" | "test"}`, más `rereads = ("split.json",)`. Es el mismo mecanismo que `label` usa para `verdicts.json`; `cloud_sync` ya lo respeta.
- **Fotos ya vistas:** conservan su lado.
- **Fotos nuevas:**
  - se reparten entre ellas por el hash de contenido que ya usa `_count`: la mitad a test, dentro del tope `test_per_class`;
  - las fotos de las que salieron modelos 3D (`held_out`) siguen fuera de test.
- **Tope de entreno.** Las fotos añadidas nunca se quedan fuera por `train_per_class`, que vale 50 por defecto y cortaría justo las fotos de calvas. El tope solo recorta el reparto inicial.
- **Alcance.** Vale para conteo (`_count`) y para clasificación. La clasificación hoy baraja con `random.Random(0)` sobre la lista entera, así que añadir un archivo mueve todo: se pasa al mismo reparto por hash.
- Se sube `version` de ingest. Los trabajos viejos rehacen ingest una vez; es barato, porque las respuestas del profesor están en caché.

Tests en `tests/test_ingest*.py`:
- añadir fotos no mueve las que ya estaban;
- la mitad de las nuevas va a test;
- el tope de entreno no las recorta.

## Fase 3. Versiones y antes/después

**Carpeta de versiones.** `sim2chip/workspace.py` gana `Job.versions` (`job/versions/`) y la función `snapshot(job, reason)`, que copia en `versions/v<n>/`:
- de `export/`: `model_int8.tflite`, `model.tflite`, `labels.txt`, `model_data.h` y `report.json`;
- además, `spec.json` y `meta.json` (fecha, motivo, titular F1 o acierto).

**Cuándo se guarda:**
- La llama el servidor antes de cualquier relanzamiento sobre un modelo terminado: correcciones, fotos nuevas y spec retocado.
- Copia además el modelo actual a `input/previous/model_int8.tflite`, con su `labels.txt`. `input/` ya viaja a la nube con `reads_input`.

**Comparación.** `sim2chip/stages/evaluate.py` gana `reads_input = True`:
- si existe `input/previous/`, puntúa también ese modelo sobre el test actual, con el mismo `tu.predict_int8` o `predict_int8_grid` y el mismo `counting.decode`;
- el resultado va a `report["previous"] = {version, real_test}`, y así la comparación es sobre las mismas fotos.
- Si el spec cambió de hardware o de clases (entrada o etiquetas distintas), no se compara, y el informe lo dice.

**API.** `status()` en `sim2chip/server.py` devuelve `versions: [{n, created, reason, headline}]`, y hay descarga por versión: `GET /api/jobs/{name}/versions/{n}/files/{file}`, con la misma lista blanca `DOWNLOADS`.

**Web.** `web/src/pages/Job.tsx`, `Result`:
- una línea "v2 · before 19 % → now 45 % on the same 24 test photos";
- una lista plegable de versiones con su descarga.

**Tests:** snapshot más `input/previous`, `status.versions` y `report.previous` (con modelos de fixture como en los tests de evaluate existentes).

## Fase 4. Añadir fotos a un trabajo terminado

**API.** Nuevo `POST /api/jobs/{name}/photos` (multipart), junto a `corrections` en `sim2chip/server.py`:
- Comprobaciones, las mismas que en correcciones y en `create_job`:
  - que el trabajo no esté corriendo (409);
  - `budget_left()`;
  - el tope `MAX_PHOTOS` en total en modo público;
  - solo extensiones `IMAGE_EXT`.
- Guarda las fotos en `input/photos/` y pone `user_data.raw = "photos"` si el spec no lo tenía.
- Llama a `snapshot(job, "added N photos: <nota>")` y a `start_run(job, req, "label")`. Con la fase 1, eso solo relanza la rama de fotos reales.
- Devuelve `{rerun_from, photos}`.
- Se reutiliza la escritura de fotos de `create_job`, sacada a una función común.

**Web:**
- `web/src/api.ts`: `addPhotos(job, files, note)`.
- En `Result` (`web/src/pages/Job.tsx`), una tarjeta "Make it better" con:
  - "What does it get wrong?", un texto opcional que queda como motivo de la versión;
  - un selector de fotos (cámara o galería, con `shrinkImage` de `web/src/images.ts`);
  - las capturas pendientes de la prueba en vivo (fase 5);
  - el botón "Add N photos and retrain", que al lanzar vuelve a la vista de progreso, como `onRerun` en Review.
- Texto que explique que se reutiliza el sintético y cuánto tarda y cuesta, con la estimación de `typical_seconds`.

**Test:** el endpoint guarda, crea la versión y lanza con "label"; rechaza un trabajo en marcha; aplica el tope público.

## Fase 5. Capturar desde la prueba en vivo

Nuevo `web/src/live/captures.ts`:
- Guarda JPEGs por trabajo en IndexedDB, para que sobrevivan a la navegación y funcionen sin conexión: `addCapture(job, blob)`, `listCaptures(job)`, `clearCaptures(job)`.
- Cada lectura y escritura va en try/catch.

`web/src/pages/Live.tsx`:
- Un botón flotante semitransparente, "This one is wrong", que guarda el fotograma actual a resolución completa.
- La foto se guarda ya enderezada con el `turn` actual, girando el canvas: el profesor la verá como en entreno.
- Un contador en el botón y un pequeño aviso de confirmación.
- Al salir con capturas pendientes, un botón "Improve with N photos" lleva a la tarjeta de la fase 4, que las muestra y las sube junto con las demás. Al subirlas se vacían.
- Fuera de línea, las capturas se quedan guardadas hasta que haya red.

**Test:** vitest del enderezado de la captura, reutilizando `uprightRgb` o su equivalente en canvas, y del almacén con `fake-indexeddb` si ya está a mano; si no, solo la lógica pura.

## Fase 6. Retocar el spec desde la entrevista

**Servidor (`sim2chip/server.py`):**
- `POST /api/interview?job=<name>`:
  - comprueba el dueño con `job_of`;
  - crea la `Session` y la siembra con el spec del trabajo.
- Nuevo método `Session.revise(spec)` en `sim2chip/interview.py`:
  - pone `proposed`;
  - añade al historial un mensaje del asistente con el resumen del spec;
  - la primera pregunta es "What would you like to change?".
- A partir de ahí el flujo de confirmación es el mismo, con `Turn.intro`.
- Nuevo `POST /api/jobs/{name}/revise` con `spec` y fotos opcionales:
  - valida igual que `create_job` (rutas solo de uploads en público, `adopt_uploads`);
  - llama a `snapshot(job, "spec changed: <diff corto de secciones>")`;
  - escribe `request.json` y llama a `start_run(job, req)` sin `start`: las huellas deciden qué se rehace. Cambiar el hardware rehace crops, train y export; cambiar clases, casi todo.
- La lógica común con `create_job` se saca a una función.

**Web:**
- En `Result`, un botón "Change the setup" lleva a `/app?revise=<job>`.
- `web/src/pages/Interview.tsx` arranca en modo revisión y, al confirmar, llama a `revise` en lugar de `createJob`.
- Antes de lanzar, un aviso con lo que se rehará y su coste aproximado: el servidor devuelve el plan con `Pipeline.plan` en un `dry` del mismo endpoint.

**Tests:**
- la sesión sembrada propone el spec sin preguntar desde cero (cliente LLM falso, como en los tests de la entrevista);
- `revise` crea una versión y lanza;
- el aviso de etapas es correcto para un cambio de hardware.

## Fuera de este plan

- La tarea por zonas (`grid`) y el flasheo desde la web: van aparte en TODO.md 2c-ter.
- Dejar que las fotos añadidas se usen para generar modelos 3D: generate_assets no se arrastra a propósito; se rehace pidiéndolo.
- Revisión por el usuario de las marcas del profesor en las fotos nuevas antes de entrenar: ya existe la revisión después, con su reentreno barato gracias a la fase 1.

## Archivos críticos

- **Pipeline:** `sim2chip/pipeline.py`, `sim2chip/stages/base.py`, `sim2chip/stages/generate_assets.py`.
- **Etapas:** `sim2chip/stages/ingest.py`, `sim2chip/stages/evaluate.py`.
- **Servidor y datos:** `sim2chip/workspace.py`, `sim2chip/server.py`, `sim2chip/interview.py`.
- **Web:**
  - `web/src/api.ts`;
  - `web/src/pages/Job.tsx`, `Live.tsx` e `Interview.tsx`;
  - nuevo `web/src/live/captures.ts`.
- **Documentación:** `README.md` (flujo de mejora), `TODO.md` (marcar 2c-ter) y la memoria del proyecto.

## Verificación

1. `pytest` (hoy 127) y `npm test` (hoy 19), más los tests nuevos de cada fase.
2. **Local, sin gastar nube**, con el trabajo `exp_places` (conteo, ya entrenado):
   - `sim2chip run ... --dry-run` o `pipe.plan("label")` muestra que render se salta;
   - subir 6 fotos por la web, que relanza solo label, ingest, train, evaluate y export;
   - el resultado muestra v2 con antes/después sobre el mismo test;
   - las fotos viejas no cambiaron de lado (se compara `ingest/split.json`).
3. **Móvil real por USB** (`adb reverse tcp:8000 tcp:8000`):
   - capturar 3 fotogramas en vivo, uno de ellos con el móvil girado;
   - comprobar que la tarjeta los muestra derechos;
   - subirlos y reentrenar;
   - la prueba en vivo carga el modelo nuevo (`freshReport`).
4. **Retocar el spec:**
   - cambiar el hardware desde la entrevista;
   - el aviso lista crops, train y export;
   - tras el relanzamiento hay v3 sin comparación, porque cambió la entrada.
5. **Nebius, una sola prueba corta y con permiso del usuario:** añadir fotos a un trabajo de la nube.
   - Comprobar que el worker se salta render y lee los recortes del bucket.
   - Anotar tiempo y coste; se espera menos de 10 min y unos 0.25 $, frente a los 22 min de la ejecución completa.
