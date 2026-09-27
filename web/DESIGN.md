# Diseño de la web de EyeForge

Plan general en `docs/plan_web.md`. Esto fija la identidad y las pantallas para que cualquier sesión construya igual.

## Identidad

- **Tono**: claro y amable (Stripe, Notion), para no expertos: un biólogo o un agricultor tiene que entenderlo todo sin saber qué es un F1. Frases cortas, verbos de acción, cifras explicadas.
- **Logo**: un ojo cuyo iris es un chip (cuadrado con patillas). SVG en `web/public/logo.svg`; el mismo dibujo sirve de icono de la PWA.
- **Color de marca**: índigo `#4f46e5` (acciones, enlaces, foco). Acento cálido ámbar `#f59e0b` solo para lo que "forja" (el chip, el progreso activo, la cifra estrella). Éxito verde `#16a34a`, error rojo `#dc2626`, aviso ámbar.
- **Neutros**: fondo `#fafaf9` (piedra muy clara), tarjetas blancas, texto `#1c1917`, secundario `#57534e`, bordes `#e7e5e4`.
- **Tipografía**: Inter variable (autoalojada con `@fontsource-variable/inter`, sin llamadas a Google: la PWA debe arrancar sin red). Titulares 600-700 con `tracking-tight`; cuerpo 16 px en móvil.
- **Forma**: radios 12 px (tarjetas) y 10 px (botones), sombras suaves (`shadow-sm`, una `shadow-lg` difusa solo en el héroe), mucho aire. Iconos `lucide-react`, trazo 1.75.
- **Movimiento**: transiciones de 150-250 ms; la etapa activa late suavemente; nada de animaciones de adorno. Respetar `prefers-reduced-motion`.

Los tokens viven en `web/src/index.css` (`@theme` de Tailwind v4): `--color-brand-*`, `--color-ember-*`, `--font-sans`, `--radius-*`.

## Pantallas

Rutas: `/` landing, `/app` crear modelo, `/app/job/:name` un trabajo (progreso, resultado, revisión).

### Landing `/`

1. Barra: logo, "How it works", "GitHub", botón "Try it".
2. Héroe: titular "Describe what your camera should see. Get a tiny model ready to flash on a $10 chip.", subtítulo de una línea, botón "Build a model" y a la derecha (debajo en móvil) una composición de tres tarjetas encadenadas: burbuja de chat ("Count the flower pots on balconies"), barra de progreso de etapas, y un chip con "3 pots".
3. "How it works" en tres pasos con icono: Describe (entrevista con Nemotron), Generate (Isaac Sim crea miles de imágenes a medida en Nebius), Deploy (modelo int8 de ~130 KB para tu dispositivo).
4. "Why a specialist": los modelos grandes saben de todo pero necesitan un centro de datos; los usamos de profesores y destilamos uno del tamaño de tu hardware. Tarjetas por perfil: ESP32-CAM, Raspberry Pi, Jetson, tu GPU.
5. "Built with": NVIDIA Nemotron, Isaac Sim, Nebius AI Cloud, Nebius Token Factory (texto, sin logos de terceros).
6. Cifras reales medidas (no inventar): una ejecución completa en Nebius, 22 min, 0,56 $ de GPU, modelo de 133 KB.
7. Pie: licencia MIT, repo, hackathon.

### Crear modelo `/app`

Cabecera de app: logo, pasos (Describe, Train, Result, Review) como migas, menú "Your models" con los trabajos del visitante.

- **Chat**: burbujas (bot a la izquierda en blanco, usuario a la derecha en índigo claro), indicador "thinking" con tres puntos. Barra de entrada fija abajo en móvil: botón de fotos (cámara o galería), campo de texto, enviar. Arrastrar fotos sobre el chat también sube.
- **Petición de fotos** (`kind: photos`): tarjeta dentro del chat con zona de soltar y botón grande "Add photos".
- **Elegir fotos para 3D** (`kind: pick`): cuadrícula de miniaturas seleccionables con su puntuación, botón "Use selected".
- **Confirmar** (`kind: confirm`): tarjeta de resumen en lenguaje llano (qué detecta o cuenta, clases con su descripción, dispositivo, escena, cuántas imágenes sintéticas, fotos propias), botones "Looks good, train it" y "Change something" (vuelve al campo de texto). El JSON, plegado en "Advanced".
- Avisos antes de entrenar: GPU ocupada o presupuesto del día agotado (de `/api/config`), con el botón desactivado y el motivo.

### Trabajo `/app/job/:name`

- **Entrenando**: tarjeta superior con la fase en palabras (`phase`), tiempo transcurrido, "about N min left" (`timing.eta_s`), coste en $ y botón "Stop". Debajo, la línea de tiempo vertical de etapas: icono de estado (hecho, activo con latido, pendiente, fallo), nombre amigable, "powered by" en gris y segundos al terminar. Log plegado en "Details".
- **Resultado**: cifra estrella grande (F1 o acierto en fotos reales) con una frase que la explica, tarjetas secundarias (error de conteo, tamaño del modelo, entrada, dispositivo), cuadrícula de 6-9 fotos de test con puntos del modelo y anillos de la verdad, y descargas (modelo, etiquetas, C array, informe). Tarjeta "Next: flash it on your chip" (informativa por ahora).
- **Revisión**: pestañas Your photos / 3D models / Synthetic images / Test. Tocar una foto abre el editor a pantalla completa en móvil: tocar para añadir marca, tocar una marca para quitarla. Barra inferior fija con "N corrections" y "Apply and retrain".

## Nombres amigables de etapas

label "Reading your photos", generate_assets "Building 3D models from your photos", ingest "Sorting your photos", fetch_assets "Finding 3D models", convert_assets "Preparing the models", preview_assets "Photographing the models", qa_assets "Checking the models", render "Generating synthetic images", points / crops "Preparing training data", train "Training", evaluate "Testing on your real photos", export "Packing the model".

## Estados

Vacío (sin trabajos), cargando (esqueletos grises), error de red (banda roja con reintento), presupuesto agotado, GPU ocupada, etapa fallida (qué pasó en una frase, log a mano).
