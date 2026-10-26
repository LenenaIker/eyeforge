# Auditoría de diseño y Clean Code (2026-10-08)

Alcance: el paquete del producto `sim2chip/` (unas 13.300 líneas de Python) y, de pasada, `web/src`. Quedan fuera `experiments/` (bancos de pruebas, se espera que sea código de usar y tirar) y todo lo de Kubernetes, SkyPilot y Soperator. No se ha cambiado código: este documento solo propone.

Criterio para cada propuesta:

1. Solo se recomienda un patrón si resuelve un problema que ya existe o que el roadmap trae seguro (tercera tarea "seguir líneas", más perfiles de hardware). Un patrón "por si acaso" es over-engineering.
2. Nada puede costar rendimiento. Las partes caras (TensorFlow, Isaac Sim, SAM 3, PicoDet) no se tocan; una llamada a método más por etapa o por foto es irrelevante frente a eso.
3. Hay que respetar dos restricciones de la arquitectura actual: el núcleo (spec, pipeline, backends, llm) es solo stdlib para correr dentro del Python de Isaac Sim, y cada etapa importa sus librerías pesadas dentro de `run()`. Cualquier refactor debe mantener esas importaciones perezosas.
4. Deadline 2026-10-30. Cada propuesta lleva cuándo hacerla: antes de la entrega solo lo barato y de bajo riesgo.

## 1. Arquitectura actual

```
cli.py / server.py (FastAPI)          entrada: terminal o web
   |                 \
interview.py (Nemotron)   warm.py (GPU caliente compartida, cola en el bucket)
   |
spec.py  (Spec: dataclasses, validación, fingerprint)  <- única fuente de verdad del trabajo
   |
pipeline.py  Pipeline: plan (fingerprints, skip/run) + run(backend)
   |                 \
stages/*.py           backends/local.py   (subproceso con el Python de cada runtime)
 Stage.run(ctx)       backends/nebius.py  (un trabajo de Nebius por etapa o por ejecución)
   |                     |
teacher.py, student.py,  run_stage.py / cloud_worker.py / warm_worker.py
detector.py, counting.py
scenes/*.py (Isaac Sim)
workspace.py (Job, Cache, versiones)
```

Valoración general: **la arquitectura del pipeline ya es buena y ya usa patrones**, aunque no se llamen así. No hay que reescribirla. Los problemas están en tres sitios concretos: la bifurcación por tarea repartida por muchas etapas, `server.py` (una función de 704 líneas) y unas pocas funciones largas.

### Patrones que ya están (conviene nombrarlos en `docs/architecture.md`, ayuda al criterio "coherencia de diseño" del jurado)

| Patrón | Dónde | Comentario |
|---|---|---|
| Template Method | `Stage` (`stages/base.py`): `run`, `enabled`, `calls_llm`, `powered_by` con valores por defecto que cada etapa redefine. `OutdoorScene` en `scenes/common.py` con `build`, `randomize`, `place_target`, `place_distractors`. | Bien hecho. |
| Strategy | `Backend` (Protocol) con `LocalBackend` y `NebiusBackend`. | Bien hecho. |
| Bridge | `Stage` (qué se hace) desacoplado de `Backend` (dónde se ejecuta): cualquier etapa corre en cualquier backend. | Es el mayor acierto del diseño. |
| Factory (simple) y Registry | `backends.get(name)`, `scenes.load(name)` con importación perezosa, `stages.BY_NAME`, `hardware.PROFILES`. | Bien; la carga perezosa es necesaria por Isaac Sim. |
| Flyweight | `Hardware` (dataclass `frozen`) compartido por todos los trabajos. | Bien. |
| Prototype | `replace(base)` del `Client` en `server.default_interviewer`; `match.trial_spec` copia el spec. | Bien, con `dataclasses.replace`. |
| Memento | `workspace.snapshot` guarda versiones del modelo en `versions/v<n>/`. | Bien. |
| Adapter | `cloud_sync.S3Store` sobre boto3, `NebiusCLI` sobre el CLI `nebius`, `detector.py` sobre PaddleDetection. | Bien. |
| Proxy (de caché) | Respuestas del profesor cacheadas por contenido de foto (`cache/labels/...`), prompts y frases cacheados. | Bien. |
| Command | Tareas de la GPU caliente como JSON en el bucket (`warm.submit`), `Plan` del pipeline. | Bien; serializable, que es lo que importa. |
| Observer (ligero) | Callbacks `log`, `progress`, `timeline` pasados como funciones. | Suficiente; no hace falta un bus de eventos. |

## 2. Catálogo completo: dónde encajaría cada patrón

Leyenda: **Sí** (recomendado), **Quizás** (solo si se cumple una condición), **No** (sería pasarse o ya está cubierto).

### Creacionales

| Patrón | Candidato en el código | Veredicto |
|---|---|---|
| Abstract Factory | Una familia por hardware: constructor de modelo + cuantizador + exportador + formato de salida (TFLite Micro para ESP32, TFLite para móvil, en el futuro Edge TPU para Coral o TensorRT para Jetson). | **Quizás**. Hoy hay 4 perfiles y 2 formatos de export; no compensa. Pasa a **Sí** cuando entre el primer perfil con un compilador propio (Coral `edgetpu_compiler`, TensorRT). La propuesta A deja el camino preparado. |
| Builder | `Session.__init__` de la entrevista tiene 11 parámetros; `create_app` tiene 14. | **No** como Builder clásico. Basta con un dataclass de configuración (ver B). `spec.from_dict` ya hace de builder validado. |
| Factory Method | Elegir la receta de entrenamiento según tarea y hardware (`count` con PicoDet o rejilla, clasificar con cabeza o fine-tune). | **Sí**, como parte de la propuesta A (`recipe_for(spec, hw)`). |
| Prototype | Ya está con `dataclasses.replace`. | **No** hace falta más. |
| Singleton | Registros a nivel de módulo (`STAGES`, `PROFILES`, `PREFERENCES`). | **No**. En Python el módulo ya es el singleton. Cuidado con lo contrario: hoy `STAGES` contiene instancias compartidas y `Label` tiene un atributo mutable `sample` (ver C.4). |

### Estructurales

| Patrón | Candidato | Veredicto |
|---|---|---|
| Adapter | Ya en S3, CLI de Nebius y Paddle. Nuevo candidato: los tres "workers" en otro Python (`sam3_worker`, `siglip_worker`, `detector_worker`) que hablan por stdin/stdout. | **No** como patrón nuevo; ver la propuesta E (una función común), no una jerarquía. |
| Bridge | Stage x Backend ya lo es. | **No** hace falta más. |
| Composite | Etapas compuestas o escenas compuestas (`PlacesScene` sobre `GenericScene`). | **No**. El pipeline es una lista con dependencias, no un árbol; la herencia de escenas es corta (3 niveles) y clara. |
| Decorator | `llm.Client` registra uso, reintenta y mide latencia dentro de `_request`/`chat`. | **No** por ahora. Pasaría a **Quizás** si aparece un segundo proveedor de LLM: entonces separar "cliente" de "registro de uso" y "reintentos" como envoltorios. Hoy sería indirección sin beneficio. |
| Facade | `server.py` mezcla HTTP con lógica de negocio (calibración del profesor, prefetch en la GPU caliente, cobro de créditos, lanzar ejecuciones). | **Sí**, propuesta B: un servicio `JobService` que la web y el CLI usan igual. |
| Flyweight | `Hardware` ya lo es. | **No** hace falta más. |
| Proxy | Caché del profesor y del LLM. Candidato: unificar el cálculo de claves de caché (ver C.6). | **No** como clase; sí una función auxiliar. |

### De comportamiento

| Patrón | Candidato | Veredicto |
|---|---|---|
| Chain of Responsibility | Comprobaciones del spec propuesto en `interview._advance` (`problem = a or b or c or ...`) y `spec.validate` (105 líneas de `if`). | **Sí, en versión ligera**: una lista de funciones validadoras que devuelven el primer problema. Sin clases. Propuesta D. |
| Command | Acciones del JSON de Nemotron en `interview._advance` (`say`, `pick_photos`, `describe_scene`, `photos`, `research`, `search`, `spec`). | **Sí, en versión ligera**: tabla `acción -> manejador`. Propuesta C. |
| Interpreter | El spec es un pequeño lenguaje; `match.py` tiene "palancas" con rutas tipo `generation.max_targets`. | **No**. Con `getattr` y una tabla de palancas basta; un intérprete sería pasarse. |
| Iterator | Recorridos de fotos y frames. | **No**. Los generadores y `rglob` de Python ya son el patrón. |
| Mediator | `Pipeline` coordina etapas; el bucket coordina servidor y GPU caliente. | **No** hace falta más. |
| Memento | `snapshot` de versiones. | **No** hace falta más. |
| Observer | Progreso de la entrevista (`_step`, `_end`), logs, timeline. | **No**. Callbacks bastan. Solo si la web pasa a recibir eventos en vivo (SSE/WebSocket) tendría sentido un emisor de eventos. |
| State | Estados como cadenas: GPU caliente (`warm.task_state`), calibración (`calibration.json` con `labeling/ready/failed`), entrevista (`finished`, `proposed`). | **Quizás**, solo para la GPU caliente si crece el número de estados y transiciones. Hoy: basta con `Literal`/`StrEnum` para que los estados válidos estén escritos en un sitio (ver C.8). |
| Template Method | Ya en `Stage` y escenas. Nuevo candidato: `Train.run` y `Train._detector` repiten el bloque "N semillas, en paralelo con spawn si hay más de una". | **No** como patrón nuevo; una función `_map_seeds(fn, seeds, ...)` (C.5). |
| Visitor | `to_dict`, `_prune`, `_build` recorren los dataclasses del spec. | **No**. Son 30 líneas recursivas, claras. Un Visitor sería pasarse. |
| Strategy | **La bifurcación por tarea** (`spec.task == "count"`) aparece en `label`, `train`, `evaluate`, `export`, `render`, `ingest`, `points`, `interview`, `match`, `backends/nebius` y `server`. | **Sí, es la propuesta más valiosa** (A). |

## 3. Propuestas recomendadas, en detalle

### A. Strategy por tarea y receta (`sim2chip/tasks/`)  [prioridad alta, después de la entrega o en la primera semana libre]

**Problema.** Lo que cambia entre "clasificar" y "contar" está repartido en al menos 11 módulos con `if spec.task == "count"`, y dentro de contar hay otra bifurcación por `hardware.counter` (`grid` o `picodet_s`), y dentro de cada una otra por pocas fotos (`head`) o fine-tune. Ejemplos:

- `stages/train.py:51-61` y `_fit_seed` (`:267-301`): tres ramas con claves distintas en el resultado (`val_f1` o `val_acc`, `recipe`, `counter`).
- `stages/export.py:31-35`: deduce la receta mirando `summary[seed].get("counter", "grid") != "grid"`.
- `stages/evaluate.py:60` y `:113`: la misma deducción.
- `stages/label.py:50, 68, 85, 248`: profesor SAM 3 o SigLIP 2.
- `backends/nebius.py:256-260`: decide qué kits instalar replicando la misma lógica.

El roadmap ya trae una tercera tarea (seguir líneas con Molmo2 como profesor) y más perfiles de hardware. Con la estructura actual, añadirla obliga a tocar todos esos `if`, y es fácil olvidar uno (por ejemplo el export, que hoy infiere la receta por la ausencia de una clave).

**Propuesta.** Un objeto por tarea con lo que de verdad varía, y la receta como estrategia dentro:

```python
# sim2chip/tasks/base.py  (stdlib, sin imports pesados)
class Task:
    name: ClassVar[str]
    teacher: ClassVar[str]            # "SAM 3", "SigLIP 2", "Molmo2"
    metric: ClassVar[str]             # "val_f1" | "val_acc"
    def label(self, stage, ctx, files, keys, previous, sample, corrections): ...
    def recipe(self, spec, hw) -> "Recipe": ...      # Factory Method
    def evaluate(self, ctx) -> dict: ...

class Recipe:                         # cómo se entrena y se exporta un modelo concreto
    name: ClassVar[str]               # "picodet_s" | "grid_head" | "grid" | "head" | "fine_tune"
    kits: ClassVar[tuple[str, ...]]   # lo que el worker de Nebius debe instalar
    def train_seed(self, args, seed, data) -> dict: ...
    def export(self, src, out, hw) -> bytes: ...

TASKS = {"roi_classification": Classify(), "count": Count()}
```

Las etapas pasan a ser finas: `Train.run` hace `recipe = task_of(spec).recipe(spec, hw)` y delega; `Export` lee `summary["recipe"]` y llama a `RECIPES[name].export(...)` en vez de adivinar; `nebius.trains_picodet` pasa a ser `"picodet" in recipe.kits`.

**Pros.**
- Añadir "seguir líneas" es crear `tasks/line.py` y registrarla; el resto del pipeline no cambia. Eso es exactamente la "generalización que sale de la arquitectura" de `CLAUDE.md`.
- Un único sitio decide la receta: se acaban las deducciones por claves ausentes.
- Más fácil de testear: cada receta se prueba sola con datos falsos.
- Buen argumento para el jurado (coherencia de diseño): "una tarea nueva es un fichero".

**Contras y riesgos.**
- Refactor amplio (unos 6 ficheros de etapas y `nebius.py`). Riesgo de romper fingerprints: si cambia el contenido de `_done.json` o el `version` de las etapas, se reentrenan trabajos existentes. Mitigación: no tocar `uses`, `after` ni `version` en el mismo cambio.
- Las importaciones pesadas deben seguir dentro de los métodos (`train_seed`, `export`), no a nivel de módulo de `tasks/`.
- Con solo dos tareas, el beneficio aún es moderado; se vuelve claro con la tercera.

**Rendimiento.** Ninguno: es una llamada de método por etapa y por semilla.

**Cuándo.** No antes de la entrega salvo que "seguir líneas" entre en la demo. Si entra, hacer este refactor primero y la tarea nueva encima.

**Paso intermedio barato (antes de la entrega).** Sin crear clases: escribir la receta elegida de forma explícita en `train/summary.json` (`"recipe": "picodet_s"`, ya existe en algunas ramas) en todas las ramas, y que `export` y `evaluate` lean esa clave en lugar de inferirla. Una hora, quita la parte más frágil.

### B. Fachada de servicio para `server.py`  [prioridad alta, después de la entrega]

**Problema.** `create_app` (`server.py:87`) tiene 704 líneas, 14 parámetros y unas 40 funciones anidadas que comparten estado por clausura (`sessions`, `runs`, `calibrations`, `started`, `prefetching`). Dentro hay lógica de negocio que no es HTTP:

- `calibrate` (`:203-259`) decide si el profesor corre aquí, en la GPU caliente o en un trabajo de Nebius, lanza un subproceso del CLI y escribe estados.
- `prefetch` (`:261-284`), `start_run`, el cobro de créditos (`pay`), el control de presupuesto.

Consecuencias: no se puede reutilizar desde el CLI, los tests tienen que levantar la app entera, y cualquier cambio en la calibración obliga a leer 700 líneas.

**Propuesta.**

1. `ServerConfig` (dataclass) con `home, backend, max_usd, public, key, max_runs, daily_usd, contact`. Sustituye a los 14 parámetros.
2. `sim2chip/jobs_service.py` (Facade): clase `JobService` con el estado que hoy está en las clausuras y métodos `create_job`, `calibrate`, `prefetch`, `start_run`, `cancel`, `status`, `add_photos`, `revise`. Sin FastAPI dentro.
3. `server.py` queda como capa HTTP: middlewares (visitante, login, clave) y rutas que traducen petición a llamada al servicio. Si se quiere, separar rutas en `APIRouter` por área (`interview`, `jobs`, `auth`, `files`).

**Pros.** Lógica testeable sin HTTP; el CLI puede calibrar igual que la web; ficheros de 200 a 300 líneas en vez de una función de 700; los estados compartidos quedan en un objeto con nombre en lugar de diccionarios sueltos.

**Contras.** Es el cambio más grande del documento. `tests/test_server.py` y `test_server_phase.py` cubren buena parte y son la red de seguridad, pero habrá que adaptarlos. Hay concurrencia (hilos de calibración y prefetch): mover el estado a una clase obliga a pensar en un `threading.Lock` que hoy se da por supuesto (hoy tampoco hay lock; ver C.9).

**Rendimiento.** Ninguno.

**Cuándo.** Después de la entrega. Antes, como mucho, extraer `calibrate` y `prefetch` a funciones de módulo con parámetros explícitos (bajo riesgo).

### C. Tabla de acciones en la entrevista (Command ligero)  [prioridad media, cualquier momento]

**Problema.** `Session._advance` (`interview.py:702`, 107 líneas) es un bucle con una cadena de `if` por cada acción del JSON de Nemotron, y la rama `spec` tiene otros 40 líneas dentro. Añadir una acción (lo que pasó con `research`) obliga a meterse en medio del bucle.

**Propuesta.**

```python
ACTIONS = {"pick_photos": Session._on_pick, "describe_scene": Session._on_describe,
           "photos": Session._on_photos, "research": Session._on_research,
           "search": Session._on_search, "spec": Session._on_spec, "say": Session._on_say}
```

Cada manejador devuelve un `Turn` (termina el turno) o `None` (sigue el bucle). El bucle queda en unas 25 líneas: pedir respuesta, parsear, contar formato roto, despachar. Se mantiene el orden de prioridad actual (`spec` gana a `say`) recorriendo la tabla en un orden fijo.

**Pros.** Cada acción se lee y se prueba sola; añadir una es una entrada en la tabla; el prompt del sistema y la tabla pueden validarse en un test (toda acción que el prompt anuncia tiene manejador).

**Contras.** Hay que cuidar los contadores compartidos (`invalid`, `bad_format`): pasan a atributos o a un pequeño estado del turno.

**Rendimiento.** Ninguno (el coste está en la llamada a Nemotron).

### D. Validadores en lista (Chain of Responsibility ligera)  [prioridad media]

**Problema.** `spec.validate` (`spec.py:405`, 105 líneas de `if`) y la cadena `problem = problem or ...` de `_advance` (`interview.py:778-785`) son la misma idea escrita de dos maneras.

**Propuesta.**
- `spec.validate` dividido por sección: `_validate_classes`, `_validate_props`, `_validate_scene` (ya existen `_validate_contexts` y `_validate_ground_mix`, seguir ese estilo), `_validate_task`, `_validate_training`. Llamadas en orden desde `validate`.
- En la entrevista, `SPEC_CHECKS = [foreign_paths, scene_check, check_photos, sam3_names_problem, photo_models_problem]`, cada una `(session, spec) -> str | None`, y `next((p for c in SPEC_CHECKS if (p := c(self, s))), None)`.

**Pros.** Lectura por sección; añadir una regla no toca las demás; mensajes de error iguales que hoy.

**Contras.** Casi ninguno. No crear clases `Validator`: funciones bastan.

### E. Un único lanzador de "Python externo"  [prioridad baja]

`LocalBackend.run_stage` (`backends/local.py`), `teacher._worker` (`teacher.py:103`) y el entrenamiento de PicoDet montan cada uno su `env` con `PYTHONPATH`, `PYTHONUNBUFFERED`, `HF_HUB_CACHE`/`TORCH_HOME`, leen la salida línea a línea, filtran y guardan una cola para el error. Propuesta: `sim2chip/subproc.py` con `run_python(exe, module, args, stdin=None, env_extra=(), show=lambda line: bool, tail=40)`. No es un patrón, es quitar duplicación. Bajo riesgo, pero tampoco urgente: las tres copias funcionan.

### F. Detalles del contrato `Backend`  [prioridad baja, 15 minutos]

- `pipeline.py:85` usa `getattr(backend, "name", "local")`: añadir `name: str` al `Protocol` `Backend` y quitar el `getattr`.
- `backends.get` reparte `kw` solo a Nebius y el CLI decide qué `kw` pasar (`cli.py:41-45`): que cada backend acepte `**kw` y descarte lo que no usa, o un `BACKENDS = {"local": ..., "nebius": ...}` como en `scenes.TEMPLATES`.

## 4. Clean Code: hallazgos concretos

Ordenados por relación beneficio/coste. Ninguno afecta al rendimiento.

### C.1 Funciones largas

| Función | Líneas | Qué hacer |
|---|---|---|
| `server.create_app` | 704 | Propuesta B. |
| `stages/render.py:62 Render.run` | 200 | Extraer el cuerpo del bucle de frames: `_compose_frame` (elegir clase y grupo), `_place_on_ground`, `_place_in_context`, `_write_labels`. El estado de Replicator se pasa explícito. Ojo: es código que corre en Isaac Sim y solo se prueba renderizando; hacerlo con un render corto de referencia antes y después (mismo seed, mismas etiquetas). |
| `glb2usd._preview_surface` / `_omnipbr` | 112 / 94 | Conversión de materiales; dejar salvo que se toque. |
| `interview.Session._advance` | 107 | Propuesta C. |
| `spec.validate` | 105 | Propuesta D. |
| `preview_assets.run`, `match.run`, `cloud_worker.main`, `qa_assets.judge`, `warm_worker.main`, `cli.main` | 70 a 96 | Solo si se tocan por otro motivo. |

Regla práctica para el futuro: si una función pasa de unas 60 líneas o de 3 niveles de anidación, partirla al tocarla.

### C.2 Configuración dispersa

Hay 56 lecturas de `os.environ`/`config.env` repartidas por el paquete (`SIM2CHIP_*`, `S2C_*`, `NEBIUS_*`, `HF_*`). Varias se leen al importar el módulo (`warm.py:38-40`, `nebius.py:47-50`), lo que hace difícil cambiarlas en tests. Propuesta: `config.py` con un dataclass `Settings` y una función `settings()` cacheada que lee todo una vez, más documentar cada variable en un único sitio (hoy están en docstrings de varios módulos). Los módulos siguen pudiendo leer `settings().warm_idle_s`. Prioridad media; bajo riesgo si se hace variable a variable.

### C.3 Resultados como diccionarios sin tipo

Los resultados de entrenamiento (`_fit_seed`, `_detector_seed`), el `args` que viaja a los procesos de semillas y el `report.json` son `dict` con claves que dependen de la rama (`val_f1` o `val_acc`, `cv_curve` opcional, `counter` opcional). Es lo que obliga a `export` y `evaluate` a adivinar. Propuesta: `TypedDict` (o dataclass serializable) `SeedResult` y `SeedArgs`. Se puede hacer sin cambiar el JSON que se escribe, así que no rompe trabajos guardados. Encaja con la propuesta A pero vale sola.

### C.4 Estado mutable en una etapa registrada

`Label.sample` es un atributo de clase que el servidor cambia en una instancia nueva (`server.py:213, 237`), y además existe la variable `SIM2CHIP_LABEL_SAMPLE` para el mismo fin cuando la calibración va a la nube (`label.py:189`). Dos canales para el mismo parámetro. Propuesta: que la muestra viaje en un único sitio (por ejemplo `ctx` o un argumento del plan) y no en la instancia de etapa, porque `STAGES` contiene instancias compartidas y un día alguien hará `BY_NAME["label"].sample = 6` en un proceso largo.

### C.5 Duplicación

- `Train.run` y `Train._detector` repiten el bloque "semillas en serie o en `ProcessPoolExecutor` con `spawn`" (`train.py:72-90` y `:124-131`). Extraer `_map_seeds(fn, args, seeds, initializer=None)`.
- `Label.powered_by` (`label.py:47`) repite el cálculo de "where" de `Stage.powered_by`. Extraer `Stage.where(backend)`.
- `train.py:120` asigna una `lambda` con `# noqa: E731`: convertir en función normal.

### C.6 Claves de caché calculadas a mano

`label._prompts`, `label._phrases` y `label.answers_dir` hacen cada una `hashlib.sha256(json.dumps(...).encode()).hexdigest()[:12]`, mientras `spec.fingerprint` hace lo mismo con `[:16]` y `sort_keys=True`. Una sola función `cache_key(*parts, n=12)` en `workspace.py`. Importante: mantener la misma serialización exacta en cada sitio (con o sin `sort_keys`) para no invalidar las cachés existentes en el bucket.

### C.7 `except Exception` amplios

Hay 35 en el paquete. Muchos son deliberados y están comentados (una foto que falla no debe perder las demás, `llm.map_within`; el estado de calibración se muestra en la web). Otros tragan el error sin registro, por ejemplo `server.py:242` (`answers_here` devuelve `False` ante cualquier error) y `:272` (`except Exception: pass`). Propuesta: revisar uno a uno y, donde se mantenga, registrar al menos `type(e).__name__` con `say()`. No cambiar comportamiento.

### C.8 Cadenas mágicas

`task`, `runtime`, `gpu`, `counter`, `recipe`, estados de calibración y de la GPU caliente son `str` libres. Usar `Literal[...]` en las anotaciones (cero coste en ejecución, compatible con el JSON y con la validación actual de `spec.py`) o `StrEnum` donde haya comparaciones repetidas. Barato y sube la legibilidad.

### C.9 Concurrencia en el servidor

`runs`, `calibrations`, `prefetching` y `sessions` son diccionarios y conjuntos compartidos entre hilos (los de calibración y prefetch escriben, las rutas leen). En CPython las operaciones simples son atómicas, así que hoy no falla, pero `prefetch` hace "comprobar y luego añadir" (`server.py:267-274`), que sí puede duplicar un prefetch si llegan dos peticiones a la vez. Un `threading.Lock` alrededor de esas dos líneas. Si se hace la propuesta B, el lock vive en `JobService`.

### C.10 Herramientas

No hay configuración de linter ni de tipos en `pyproject.toml`. Propuesta mínima:

```toml
[tool.ruff]
line-length = 120
[tool.ruff.lint]
select = ["E", "F", "W", "B", "SIM", "UP", "I"]
```

`ruff check` tarda menos de un segundo y caza imports sin usar, variables muertas y errores comunes. `mypy` solo sobre el núcleo stdlib (`spec`, `pipeline`, `workspace`, `llm`, `hardware`), en modo no estricto, cuando se haga C.3. En `web/`, `tsc --noEmit` ya está implícito en el build; `api.ts` define `Spec` como `Record<string, any>`: tiparlo a partir del spec real si se toca la web.

### C.11 Frontera entre `experiments/` y el producto

`experiments/scene3d/` tiene unas 6.300 líneas y `photo2scene` va camino de ser parte del producto. Recomendación: cuando entre, que lo haga como etapa (`stages/scene_from_photo.py`) que llama a un módulo en `sim2chip/`, nunca importando de `experiments/`. Así `experiments/` puede seguir siendo desordenado sin contaminar el paquete. Mover solo lo que el producto usa.

### Lo que ya está bien y hay que conservar

- Comentarios que explican el porqué con fecha y medida (por ejemplo `hardware.py` sobre EfficientNetV2, `interview.py` sobre la temperatura): es lo más valioso del código para mantenerlo. No "limpiarlos".
- `version` de cada etapa con su historial: correcto; si la línea crece mucho (`train.py:41`), pasar el historial a un comentario multilínea encima.
- Fingerprints por sección del spec con poda de vacíos: diseño robusto que permite añadir campos sin invalidar trabajos.
- 25 ficheros de tests, incluidos pipeline, spec, servidor, backend de Nebius y warm. Son la red de seguridad de cualquier refactor de este documento.

## 5. Plan propuesto

Antes de la entrega (2026-10-30), solo lo de bajo riesgo:

| # | Cambio | Esfuerzo | Riesgo |
|---|---|---|---|
| 1 | Receta explícita en `train/summary.json` y lectura en `export`/`evaluate` (paso intermedio de A) | 1 h | Bajo |
| 2 | `name` en el `Protocol` `Backend` (F) | 15 min | Nulo |
| 3 | Lock en `prefetch` (C.9) y registro en los `except Exception` mudos (C.7) | 1 h | Bajo |
| 4 | `ruff` con reglas conservadoras (C.10), arreglar solo lo trivial | 1 h | Nulo |
| 5 | Nombrar los patrones existentes en `docs/architecture.md` (sección 1) | 30 min | Nulo |

Estado 2026-10-08: hechos los cinco (sin commit). La receta ya se guardaba en todas las ramas de `train`; `export` y `evaluate` la leen con `train.is_detector` (test en `tests/test_detector.py`). `ruff` configurado en `pyproject.toml` con reglas conservadoras e ignorados razonados (líneas largas, `zip` sin `strict`, `File()` de FastAPI, `l`, cierres en bucles); arreglados imports y variables sin usar y `raise ... from None` en las respuestas HTTP. Registro añadido en los `except` mudos de `server.py`, `teacher.py`, `backends/nebius.py` y `warm.py`. 218 tests en verde.

Después de la entrega, en este orden:

| # | Cambio | Esfuerzo | Condición |
|---|---|---|---|
| 6 | Tabla de acciones de la entrevista (C) y validadores en lista (D) | medio día | Ninguna |
| 7 | `TypedDict` de resultados (C.3), `cache_key` (C.6), `_map_seeds` (C.5), `Literal` (C.8) | medio día | Ninguna |
| 8 | `Task` y `Recipe` (A) | 2 a 3 días | Antes de la tercera tarea; antes de la entrega solo si "seguir líneas" entra en la demo |
| 9 | `JobService` y `ServerConfig` (B) | 2 días | Cuando haya que tocar la calibración o el servidor otra vez |
| 10 | `Settings` central (C.2), lanzador común (E), partir `Render.run` (C.1) | 1 a 2 días | Al tocar esas zonas |
| 11 | Abstract Factory por hardware | | Solo cuando entre un perfil con compilador propio (Coral, TensorRT) |

Estado de los pasos 6 a 10 (2026-10-08, un commit por paso, 228 tests en verde):

- **6, hecho** (`77c7956`). `Session.ACTIONS` con un método por acción, en el mismo orden de antes (el spec sigue ganando al texto); `SPEC_CHECKS` como lista de funciones; `spec.validate` partido por secciones con el orden de las reglas intacto.
- **7, hecho en parte** (`f2e354f`). `workspace.cache_key` (claves idénticas byte a byte, verificado), `Literal` en `Stage.runtime`, `Stage.gpu` y `Hardware.counter`, lambdas a funciones. `Literal` no se puso en `Spec`: su validación lee el tipo anotado como texto (`"str"`) y cambiarlo rompería `_build`. `_map_seeds` descartado: las dos ramas difieren en lo importante (datos en fichero o en memoria, descarga previa de pesos) y unificarlas costaba más que las ocho líneas repetidas.
- **8, hecho en versión intermedia** (`ca90459`). `sim2chip/tasks.py`: `Task` (métrica, clave de validación, profesor, si responde con puntos, qué detector entrena en cada hardware) y `Recipe` (si es detector, fichero que se exporta, cuantización), más `SeedResult` (`TypedDict`). `label`, `train`, `evaluate`, `export` y la elección de kits de Nebius preguntan ahí. El código de entrenar y puntuar sigue en las etapas: con dos tareas los métodos serían una apuesta; la tercera (seguir líneas) dirá cuáles. Test nuevo de export de un detector, que pasa también con el código anterior.
- **9, hecho** (`73f9f68`). `sim2chip/job_service.py` con `JobService` (calibración, prefetch con su lock, lanzar y cancelar, límite público, presupuesto). `create_app` baja de 704 a 581 líneas, todas rutas. `ServerConfig` descartado: los parámetros ya van por nombre y `serve` y los tests los usan así; no aportaba nada. Partir las rutas en `APIRouter` queda para cuando se toque la web.
- **10**. Lanzador común `subproc.run_module` y `python_env` en backend local, profesores y PicoDet (`39baf77`), con tests que arrancan un Python hijo real. `Settings` central descartado con motivo: unas 50 variables leídas donde se usan, algunas después de que los tests las cambien; lo que faltaba era encontrarlas, así que `.env.example` las lista todas con su valor por defecto. `Render.run` partido en `_new_stage`, `_build_scene`, `_camera`, `_load_models` y `_Shots` (una toma por tipo), verificado renderizando dos trabajos reales antes y después con la misma semilla.
- **C.4** (dos canales para la muestra de calibración) se queda: la variable de entorno es lo único que cruza al proceso del CLI que lanza la calibración en la nube.

Patrones descartados explícitamente para no volver a discutirlos: Singleton, Builder, Composite, Interpreter, Iterator, Mediator, Memento (ya está), Visitor, Observer con bus de eventos, Decorator del cliente LLM (hasta que haya un segundo proveedor).

## 6. Opinión

El proyecto no tiene un problema de patrones; tiene un núcleo bien diseñado (Stage x Backend, fingerprints, spec como única fuente de verdad) y dos zonas que han crecido por acumulación: la lógica por tarea y el servidor. Aplicar los 22 patrones sería claramente pasarse. Los que de verdad pagan su coste son tres: **Strategy/Factory Method para tareas y recetas** (porque el roadmap trae una tercera tarea y más hardware), **Facade para separar el servidor de la lógica** y **tablas de manejadores/validadores** en la entrevista y el spec. El resto de la deuda se resuelve con Clean Code básico (funciones más cortas, tipos en los resultados, configuración en un sitio), sin coste de rendimiento.

## 7. Segunda ronda (2026-10-27, sin ejecutar tests)

Revisión de calidad pedida por el usuario tras el audit de la ruta F3c. Cinco puntos; solo se ejecutaron comprobaciones estáticas (ruff, AST, py_compile) y la comparación numérica de `composite.py` contra el código anterior. Los tests nuevos están escritos pero sin pasar.

- **Servidor partido.** Las rutas de `create_app` pasan a `sim2chip/api/` (`interview`, `jobs`, `library`, `scenes`, `billing` y los helpers de spec en `specs`), cada módulo con `add(app, ctx)`; `ctx` es el estado compartido. `create_app` baja de 1675 a unas 400 líneas (middleware, quién pregunta, qué puede lanzar). Verificado por AST: las 68 rutas tienen el mismo cuerpo salvo tres que ahora usan `my_jobs`, y ningún par de rutas que se solapan (`/api/library/{kinds}/{item}` frente a `/api/library/labelings/{lid}`) cambia de orden. Los imports relativos de los cuerpos se movieron un nivel (`from ..`).
- **Comprobación estática.** ruff limpio en `sim2chip`, `tests` y la ruta de escena (que se lintea nombrándola: `experiments/` sigue excluido). ruff no habría cazado el `has_focal` que quedó en `scene_pick.py`: pyflakes no mira dentro de otros módulos. Lo caza `tests/test_imports.py`, que comprueba cada `from .x import y` contra el código de `x`. mypy descartado antes de la entrega: 210 avisos, casi todos de opcionales, ninguno real.
- **Configuración.** No es el `Settings` congelado descartado en la sección 5 (los tests cambian variables y se siguen leyendo donde se usan). `config.py` gana lectores con tipo (`env_int`, `env_float`) que nombran la variable si el valor no es un número, `.env` leído una vez mientras no cambie, y `KNOWN`, el registro de cada `SIM2CHIP_*` y `S2C_*` que lee el código (un test lo mantiene completo). `check()` avisa de nombres mal escritos, números que no lo son e intérpretes o carpetas que no existen, sin imprimir valores; corre al arrancar `sim2chip serve` y como `sim2chip config`. Fallo real corregido: `SIM2CHIP_PARALLEL_SEEDS`, `SIM2CHIP_LLM_WORKERS`, `SIM2CHIP_LLM_BASE_URL`, `SIM2CHIP_PICODET_STEPS`, `SIM2CHIP_PYTHON`, `SIM2CHIP_HOME` y los `SIM2CHIP_<ROL>_MODEL` venían en `.env.example` pero se leían solo del entorno: en `.env` no hacían nada.
- **Funciones largas.** `walkscene/isaac.py` partido en pasos con nombre (`new_stage`, `ground_plane`, `sky_dome`, `load_objects`, `camera`, `show`); la aritmética (prims por objeto, composición diferencial, cajas) pasa a `walkscene/composite.py`, sin Isaac, con tests en CPU (`tests/test_composite.py`). Comparado con el bucle anterior en 20 vistas aleatorias: idéntico. `splat.train` no se tocó: el otro agente lo tiene en su commit de Qwen.
- **Capturas genéricas.** Las que callaban el motivo lo dicen ahora en el log: `llm.map_within` (cuántos fallaron y el primer error; antes, con una clave caducada solo se veían respuestas vacías), los tamaños típicos de la entrevista, los prompts del profesor, el uso de GPU del caliente y el EXIF ilegible de una foto de escena. Las demás ya registraban o son reservas comentadas.
