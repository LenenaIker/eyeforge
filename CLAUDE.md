# CLAUDE.md

Este archivo se carga automáticamente al inicio de cada sesión de Claude Code en este repo. Mantenlo corto y denso: hechos duraderos y decisiones tomadas, no estado de trabajo en curso.

## Reglas de interacción

1. **Nunca añadir a Claude como coautor en commits.** Nada de líneas `Co-Authored-By: Claude` ni menciones a Claude/IA en commits ni PRs de este repo.
2. **No usar em-dash (—) ni otros símbolos típicos de estilo IA** en código, commits, documentación ni respuestas dentro de este proyecto. Usar punto, coma o paréntesis en su lugar.

## Qué es esto

**Nombre: EyeForge** (dominio `eyeforge.app`, comprado en Porkbun el 2026-09-26 por un año; decisión del usuario 2026-09-26). "Sim2Chip" queda como nombre interno del motor: el paquete Python, el CLI `sim2chip`, las variables `SIM2CHIP_*` y los recursos de Nebius no se renombran (riesgo sin beneficio antes de la entrega). Tagline actual: "Describe what your camera should see. Get a tiny model ready to flash on a $10 chip." (revisar cuando existan perfiles de hardware más allá de la ESP32). Descripción de Devpost en `docs/devpost.md`.

Sistema polivalente para entrenar modelos de visión especialistas que corren en el propio dispositivo, sin nube. Los chips baratos (ESP32-CAM) siguen siendo el objetivo central y la demo, pero el target se amplía a cualquier usuario que necesite un especialista local (usuario, 2026-09-26): el hardware es un parámetro del spec (ESP32, Raspberry Pi o Coral, Jetson, GPU propia) y el alumno se dimensiona para él. El usuario describe qué quiere detectar, clasificar o contar, aporta fotos y conocimiento experto, y el sistema devuelve un modelo entrenado, cuantizado y listo para su hardware. Pitch: los modelos fundacionales (SAM 3, VLMs) saben un poco de todo pero necesitan un centro de datos y fallan en distinciones finas (especies casi idénticas); nosotros los usamos como profesores y destilamos un especialista del tamaño de tu hardware. Se vende como "pocas fotos reales (aunque sean crudas) + datos sintéticos", no como "cero fotos": el spike demostró que solo sintético no basta todavía.

**Lo más relevante del proyecto (usuario, 2026-09-24): la GENERALIZACIÓN.** Cualquiera (un biólogo que quiere saber si pasó el animal que estudia, un comprador de lora_alarm, una obra con excavadoras) aporta fotos y texto y el pipeline se adapta a su objeto y su hardware. El tráfico de lora_alarm es solo el primer caso de demo, no el producto. Es lo que más valorará el jurado, pero no es una tarea aparte: sale de la arquitectura (roadmap en `TODO.md`: primero construir el sistema, pulir el realismo al final). Debe funcionar al vuelo desde el spec, sin ajustes manuales por clase.

Proyecto para el **Nebius x NVIDIA Global AI Hackathon** (https://nebiusglobalaihackathon.devpost.com/). Track: **Physical AI** ("embodied and edge agents... robotics, IoT, and on-device intelligence").

## Pipeline previsto

Detalle, subprocesos opcionales y tareas abiertas en `TODO.md`.

1. **Entrevista inicial (Nemotron, Token Factory)**: preguntas al usuario para fijar el spec del dataset (clases, iluminación, tiempo, entorno, cámara, detalles como personas montadas en motos, hardware objetivo). Sale un spec estructurado.
2. **Datos propios del usuario (opcional)**: etiquetados se usan directos; crudos los etiqueta un profesor elegido por tarea (benchmark en `docs/profesor_modelos.md`): SigLIP 2 para clasificar (aprende el criterio del usuario de unas pocas correcciones, decisión 2026-09-30) y SAM 3 para contar (umbral según la preferencia de errores). Los modelos siempre eligen una de las clases del usuario, nunca "duda" ni una clase de fondo inventada si las clases cubren toda la foto (usuario, 2026-09-30). Se reparten en train, validación y test.
3. **Generación sintética**: Isaac Sim (Replicator, headless) guiado por el spec. Assets de Objaverse-LVIS. Opcional: Cosmos para realismo.
4. **Control de calidad**: Nemotron VL filtra assets y renders malos.
5. **Entrenamiento del modelo pequeño (alumno)** con sintético + datos del usuario, cuantizado int8.
6. **Evaluación en test real y export** para el hardware (ESP32-CAM: TFLite Micro).

Desarrollo en la RTX 3090 local; el producto final lanza los trabajos en **Nebius AI Cloud (L40S)**, que se apaga al terminar.

## Decisiones y restricciones

- **GPU en la nube: L40S, no H100/H200.** Isaac Sim necesita GPUs RTX (con RT cores) para renderizar; H100/H200 no los tienen.
- **Assets 3D son el riesgo principal.** "Detecta cualquier cosa" requiere modelos 3D de cualquier cosa. Alcance del hackathon: catálogo cerrado de 10-20 clases con assets existentes (NVIDIA SimReady, Objaverse) + modo híbrido con fotos reales para lo demás. No prometer "cualquier objeto".
- **La granularidad depende del hardware de cada usuario, no del pipeline.** En ESP32-CAM: pocas clases, baja resolución (p. ej. vehículo grande vs pequeño, no marcas). Con mejor cámara y cómputo (Jetson, GPU), distinciones finas. Se presenta como decisión de diseño por perfil, no se esconde.
- **Modelos abiertos primero** (usuario, 2026-09-27): por defecto, modelos abiertos en Nebius (Token Factory, NVIDIA). Los cerrados o de pago (Claude Opus/Fable por API, Hunyuan, Meshy, Rodin) solo para lo que con abiertos es imposible o de pésima calidad, medido antes y como opción, no por defecto.
- **Fuentes de assets 3D**: por defecto búsqueda en internet; opcionales modelos del usuario, generación desde sus fotos con modelo abierto (TRELLIS v1, MIT) o servicio de pago premium. Detalle en `TODO.md` 3.E.
- **Caso de demo: `~/work/multi-platform/lora_alarm`** (proyecto hermano e independiente: ESP32-CAM clasifica en el borde, destruye la imagen y manda unos pocos bits por LoRa a un gateway ESP32; pensado para cruces en carreras ciclistas). Este proyecto no modifica lora_alarm; solo le suministra modelos. La integración en lora_alarm la trabaja el usuario (2026-09-24): aquí no hay tareas de lora_alarm.

## Requisitos de entrega del hackathon

- **Licencia: AGPL-3.0-only** (decisión del usuario, 2026-09-28): las bases exigen repo público con licencia open source; AGPL protege el negocio (quien lo ofrezca como servicio debe publicar su código) y el autor puede vender licencias comerciales aparte. Se publica el proyecto entero, web incluida, cerca de la entrega. Pendiente confirmar con la organización que "such as Apache 2.0, MIT, or MPL 2.0" admite AGPL (si no, MPL 2.0). No aceptar contribuciones externas sin un acuerdo que permita esa doble licencia.

Deadline: **2026-10-30, 10:00 PDT**.

- Corre en Nebius Token Factory o AI Cloud y usa al menos un modelo open-source de NVIDIA (destacar Nemotron y dónde aceleró Token Factory).
- Vídeo ≤ 3 min; en Physical AI, al menos 1 min del hardware real funcionando.
- Repo público con licencia open-source, README con instrucciones de setup, descripción del proyecto.
- Criterios: implementación técnica, coherencia de diseño, impacto, calidad de la idea.
- Créditos: Nebius Builder Program (Token Factory, Tavily).

## Estado y dónde mirar

- Prueba de viabilidad hecha (2026-09-23): código en `experiments/legacy/`, hallazgos y resultados en `experiments/NOTES.md`; los bancos de pruebas actuales (FSC-147, profesor, densidad) en `experiments/`. Specs: `specs/examples/` para usuarios y `specs/benchmarks/` para pruebas internas.
- Entorno conda `edge-sdg` (objaverse, usd-core, TensorFlow 2.18 GPU). Scripts de Isaac con `/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh`.
- `data/` es un enlace simbólico a `/media/lenena-iker/Crucial/datasets/eyeforge/` (SSD secundario, gitignorado, ~13 GB): assets Objaverse y USD, HDRIs locales, renders, recortes, MIO-TCD. Todo lo pesado o de escritura masiva va ahí, nunca al SSD del sistema. Los scripts usan `ROOT/data` y las rutas absolutas guardadas (JSON, USD) pasan por el enlace: no quitarlo.
- `runs/` (modelos entrenados y resultados) es un enlace simbólico a `/media/lenena-iker/Crucial/runs/eyeforge/`.
- La caché de Isaac Sim (`~/.cache/ov`) se queda donde está: decisión del usuario, no moverla. `data/usd/exclude.txt` es la lista manual de assets descartados. MIO-TCD solo para evaluación (licencia de investigación, no redistribuir).
