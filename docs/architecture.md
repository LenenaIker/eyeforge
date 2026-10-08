# Arquitectura de EyeForge (motor `sim2chip`, 2026-09-24)

Estado: aprobada por el usuario e implementada en `sim2chip/` (semana 1). Sustituye a los scripts sueltos de `experiments/legacy/`, que se conservan como histórico.

Cambios respecto a la propuesta inicial, decididos al implementar:
- **`convert_assets` no depende de la clase**: solo endereza, centra y mide (largo, ancho, alto). La escala real del spec se aplica en `render`. Una reclasificación en QA no obliga a reconvertir y la caché sirve a cualquier trabajo. El USD normalizado referencia el crudo con ruta relativa, para que la caché se pueda montar en otro sitio (Nebius).
- **QA manual** (`qa.mode = "manual"`): lista de exclusión, reclasificaciones y modelos sin color, como entradas del trabajo (`--input review=...`). El modo automático con Nemotron VL llega en la semana 2.
- **Las rutas del spec son relativas a `input/` del trabajo**, que se rellena con `--input NOMBRE=RUTA` (enlace simbólico en local, subida en Nebius).
- **Selección de assets reproducible**: misma regla de muestreo que el spike; el spec de tráfico elige exactamente los mismos 274 modelos.

## Principios

1. **Un spec manda.** Todo lo que hoy está fijo en el código (clases, tamaños, distancias, escena, cámara, mapeo de fotos reales, modelo) sale de un `spec.json`. Hoy se escribe a mano como caso de prueba; en la semana 2 lo genera la entrevista con Nemotron. Ningún módulo conoce la palabra "vehículo".
2. **Etapas con interfaz común.** Cada paso del pipeline es una etapa que lee y escribe en la carpeta del trabajo. No se pasan datos en memoria entre etapas: así cualquier etapa puede correr en otra máquina.
3. **El dónde se decide aparte.** Un backend (local o Nebius) decide cómo y dónde se ejecuta cada etapa. Las etapas no saben dónde corren.
4. **Núcleo como librería.** El CLI, y más adelante la API y la web, son capas finas encima.

## Vista general

```
sim2chip run specs/examples/lora_alarm_traffic.json --backend local
        │
        ▼
   Pipeline  ── construye la lista de etapas a partir del spec (las opcionales solo si aplican)
        │
        ▼
   Backend   ── LocalBackend: subproceso con el intérprete adecuado (python.sh de Isaac o conda edge-sdg)
        │       NebiusBackend: un Serverless AI Job (L40S) que ejecuta todo el pipeline (cloud_worker.py)
        ▼
   Etapa.run(ctx)  ── lee y escribe en jobs/<id>/
```

## Etapas

| Etapa | Qué hace | Runtime | Viene de |
|---|---|---|---|
| `interview` | Conversación con Nemotron -> `spec.json` (web o terminal) | Token Factory | `interview.py` |
| `label` | Profesor (modelo de visión) que etiqueta fotos crudas; respuestas en caché por foto | Token Factory | nuevo |
| `generate_assets` | Modelos 3D desde fotos del usuario (TRELLIS, opcional, solo local) | trellis | nuevo |
| `ingest` | Fotos del usuario: normaliza, mapea sus clases a las del spec, reparte train/val/test | cpu | `prepare_crops.real` |
| `fetch_assets` | Busca y descarga modelos 3D por clase (búsqueda web, Objaverse-LVIS), cielos y texturas | cpu | `download_assets.py` |
| `convert_assets` | GLB -> USD con `glb2usd` (Asset Converter de Isaac solo de respaldo), endereza y mide | cpu | `convert_assets.py` |
| `preview_assets` | Cuatro vistas de cada modelo para el control de calidad | isaac | nuevo |
| `qa_assets` | Modelo de visión decide conservar o descartar cada modelo; base de conocimiento compartida | Token Factory | `exclude.txt` |
| `render` | Genera el dataset sintético con la plantilla de escena del spec | isaac (GPU RTX) | `generate.py` |
| `crops` / `points` | Recortes del tamaño de entrada (clasificación) o imágenes con puntos (conteo) | cpu | `prepare_crops.synth` |
| `train` | Entrena el modelo (sintético + fotos del usuario), N semillas en paralelo en la misma GPU | tf (GPU) | `train.py` |
| `evaluate` | Resultado del modelo int8 en el test real (precisión o F1 de conteo) | tf | `train.py` |
| `export` | `.tflite` int8, cabecera C para TFLite Micro, informe | cpu | nuevo |

Interfaz de una etapa (esbozo):

```python
class Stage:
    name: str
    runtime: str            # "cpu" | "isaac" | "tf" | "trellis"
    gpu: str | None         # None | "any" | "rtx"
    needs_llm: bool         # llama a Token Factory (calls_llm(spec) decide si de verdad lo hace)
    def enabled(self, spec) -> bool: ...   # p. ej. `label` solo si hay fotos crudas
    def run(self, ctx: Context) -> None: ...  # ctx.spec, ctx.job (rutas del trabajo), ctx.cache, ctx.log
```

Cada etapa escribe al terminar `jobs/<id>/<etapa>/_done.json` con un hash de la parte del spec que usa y de sus entradas. Si nada ha cambiado, al relanzar se salta. Así se puede cambiar solo el entrenamiento sin volver a renderizar. El CLI permite `--from render` y `--only train`.

## Por qué subprocesos

Isaac Sim tiene su propio Python (`python.sh`) y TensorFlow vive en el entorno conda `edge-sdg`: no pueden compartir proceso. Cada etapa se lanza como `python -m sim2chip.run_stage <nombre> --job <carpeta>` con el intérprete de su runtime. En Nebius, un solo job ejecuta el pipeline entero con el backend local dentro del contenedor (modo worker), sobre una de las dos imágenes propias: la ligera si no hay sintético, la de Isaac si se renderiza (`docs/imagenes.md`). Sin ellas, usa la imagen pública de Isaac Sim e instala TensorFlow y lo demás con `uv` al arrancar. El paquete `sim2chip` solo usa la biblioteca estándar en su núcleo (spec, pipeline, etapas base), para que funcione en los dos intérpretes sin instalar nada más en el de Isaac.

## Espacio de trabajo

```
jobs/<id>/
  spec.json
  input/            fotos y texto del usuario
  ingest/           fotos normalizadas y repartidas (train/val/test)
  assets/           manifiesto de modelos 3D elegidos para este trabajo
  render/           imágenes y bboxes sintéticos
  crops/
  train/seed_<n>/   modelo, historial
  evaluate/         results.json por semilla y resumen (media ± desviación)
  export/           model_int8.tflite, model_data.h, report.json
cache/
  objaverse/        descargas compartidas entre trabajos
  usd/<uid>_<hash>.usd   conversiones reutilizables
```

- **Local:** `jobs/` y `cache/` en `data/` (enlace a Crucial).
- **Nebius:** el worker copia del bucket del trabajo a una carpeta local con la misma estructura (solo lo que necesitan las etapas pendientes), ejecuta y sube el resultado; el portátil trae de vuelta lo que muestra la web. Las etapas ven carpetas normales en ambos casos: la diferencia la gestiona el backend, no las etapas.
- La caché de assets vive fuera del trabajo porque se reutiliza: un segundo cliente que pida coches no vuelve a descargar ni convertir.

## Spec (esbozo del caso de prueba de vehículos)

```json
{
  "name": "lora_alarm_traffic",
  "task": "roi_classification",
  "hardware": "esp32cam",
  "camera": {"height_m": [2, 6], "hfov_deg": 60, "resolution": [320, 240]},
  "scene": {"template": "road", "lighting": ["day", "dusk"], "distractors": ["tree", "person"]},
  "background_class": "empty",
  "classes": [
    {"id": "small_vehicle", "description": "cars, vans, pickups",
     "assets": {"source": "objaverse_lvis", "categories": ["car_(automobile)", "cab_(taxi)", "jeep", "minivan", "pickup_truck"]},
     "size_m": {"measure": "width", "range": [1.65, 1.95]}},
    {"id": "bicycle", "description": "bicycles",
     "assets": {"source": "objaverse_lvis", "categories": ["bicycle"]},
     "size_m": {"measure": "length", "range": [1.6, 1.9]}}
  ],
  "user_data": {"labeled": "input/mio_tcd", "class_map": {"car": "small_vehicle", "bicycle": "bicycle"}, "test_fraction": 0.5},
  "generation": {"frames": 2000, "empty_ratio": 0.2},
  "training": {"seeds": [0, 1, 2], "epochs": 25}
}
```

- **La distancia cámara-objeto ya no va por clase:** se deduce del tamaño real y del FOV para que el objeto ocupe un rango de píxeles razonable. Un parámetro menos que alguien tenga que inventarse.
- **El hardware es un perfil con nombre** (`esp32cam`, `esp32s3`, `jetson_nano`...) que fija tamaño de entrada, modelo, cuantización y memoria máxima. El spec solo lo nombra.
- MIO-TCD entra como "datos etiquetados del usuario" con su mapeo de clases: el mismo camino que usará cualquier cliente.

## Registros (puntos de extensión)

Mismo patrón para todo lo que crecerá: un diccionario nombre -> implementación, elegido desde el spec.

- **Plantillas de escena** (`scenes/road.py`; luego obra, campo, interior): construir la escena, aleatorizar entorno, colocar objetivo y cámara.
- **Fuentes de assets** (`objaverse_lvis`; luego búsqueda en Objaverse completo).
- **Perfiles de hardware** (`esp32cam`...).
- **Backends** (`local`, `nebius`).

Añadir un entorno o un hardware nuevo es añadir un archivo, sin tocar el resto.

## Patrones de diseño en uso (2026-10-08)

Nombres clásicos de lo que el código ya hace, para hablar de él con el mismo vocabulario. Auditoría completa, con lo que falta y lo que se descarta, en `docs/auditoria_diseno.md`.

- **Bridge**: `Stage` (qué se hace) y `Backend` (dónde corre) varían por separado; cualquier etapa corre en cualquier backend.
- **Strategy**: `Backend` (`Protocol` en `pipeline.py`) con `LocalBackend` y `NebiusBackend`.
- **Template Method**: `Stage` (`run`, `enabled`, `calls_llm`, `powered_by` con valores por defecto) y las escenas sobre `OutdoorScene` (`build`, `randomize`, `place_target`, `place_distractors`).
- **Registro con factoría perezosa**: `backends.get`, `scenes.load`, `stages.BY_NAME`, `hardware.PROFILES`; importan lo pesado solo al usarlo (Isaac Sim, TensorFlow).
- **Flyweight**: los perfiles `Hardware` son inmutables y compartidos por todos los trabajos.
- **Prototype**: copias con `dataclasses.replace` (cliente LLM por entrevista, spec de prueba en `match.py`).
- **Memento**: `workspace.snapshot` guarda cada modelo anterior en `versions/v<n>/`.
- **Adapter**: `cloud_sync.S3Store` (boto3), `NebiusCLI` (CLI `nebius`), `detector.py` (PaddleDetection).
- **Proxy de caché**: respuestas del profesor por contenido de foto (`cache/labels/`), prompts y frases traducidas.
- **Command**: tareas de la GPU caliente como JSON en el bucket (`warm.submit`) y el `Plan` del pipeline.
- **Observer ligero**: callbacks `log`, `progress` y `timeline` pasados como funciones.
- **Receta por semilla**: cada semilla deja su `recipe` en `train/summary.json` (`picodet_s`, `head`, `fine_tune`); `export` y `evaluate` la leen con `train.is_detector` en vez de deducirla.

## Estructura del repo

```
pyproject.toml
sim2chip/
  cli.py  spec.py  pipeline.py  workspace.py  hardware.py
  stages/    ingest.py fetch_assets.py convert_assets.py qa_assets.py render.py crops.py train.py evaluate.py export.py
  scenes/    road.py
  backends/  local.py  (nebius.py en semana 3)
specs/       examples/ (para usuarios) y benchmarks/ (pruebas internas)
tests/       spec, pipeline y caché (sin GPU)
experiments/ bancos de pruebas; legacy/ guarda la prueba de viabilidad
docs/
```

## Criterio de hecho de la semana 1

`sim2chip run specs/examples/lora_alarm_traffic.json --backend local` de principio a fin, sin pasos a mano, reproduce el híbrido del spike (77.3 ± 1.7 con 3 semillas, dentro del ruido).

## Fuera de la semana 1

Entrevista y profesor (semana 2), backend Nebius, API y web (semana 3), mejoras de realismo (semana 4).
