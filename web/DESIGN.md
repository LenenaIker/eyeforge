# Diseño de la web de EyeForge

Plan general en `docs/plan_web.md`. Esto fija la identidad y las pantallas para que cualquier sesión construya igual.

## Identidad

- **Tono**: claro y amable (Stripe, Notion), para no expertos: un biólogo o un agricultor tiene que entenderlo todo sin saber qué es un F1. Frases cortas, verbos de acción, cifras explicadas.
- **Logo**: un ojo cuyo iris es un chip (cuadrado con patillas). SVG en `web/public/logo.svg`; el mismo dibujo sirve de icono de la PWA.
- **Color de marca** (propuesta D, usuario 2026-10-08, estilo claro de nebius.com): índigo `#614efa` (acciones, enlaces, foco). Acento cálido ámbar `#f59e0b` solo para lo que "forja" (el chip, el progreso activo, la cifra estrella). Éxito verde `#16a34a`, error rojo `#dc2626`, aviso ámbar.
- **Neutros**: grises fríos. Fondo de la app `#f3f4f5` (las bandas de la landing), tarjetas blancas, texto `#0f1b24`, secundario `#55626c`, bordes `#e3e6e9`, pie oscuro `#0f1b2d`. Clases `gray-*` de Tailwind, no `stone-*`.
- **Tipografía**: Inter variable (autoalojada con `@fontsource-variable/inter`, sin llamadas a Google: la PWA debe arrancar sin red). Titulares 600-700 con `tracking-tight`; cuerpo 16 px en móvil.
- **Forma**: radios 16 px (tarjetas) y 10 px (botones), sombras suaves (`shadow-sm`, una `shadow-lg` difusa solo en el héroe), mucho aire. Iconos `lucide-react`, trazo 1.75.
- **Movimiento**: transiciones de 150-250 ms; la etapa activa late suavemente; nada de animaciones de adorno. Respetar `prefers-reduced-motion`.

Los tokens viven en `web/src/index.css` (`@theme` de Tailwind v4): `--color-brand-*`, `--color-ember-*`, `--font-sans`, `--radius-*`.

## Pantallas

Rutas: `/` landing; dentro de `/app`, con barra lateral (como la consola de Nebius): `/app` chat (crear modelo), `/app/models`, `/app/datasets`, `/app/assets/objects`, `/app/assets/scenes`, `/app/kits`, `/app/billing` y `/app/job/:name` un trabajo (progreso, resultado, revisión).

Biblioteca (usuario, 2026-10-08): toda foto real pertenece a un dataset (las que se adjuntan en un chat crean uno al momento); todo objeto 3D y escena se guarda con una foto para su tarjeta; de lo sintético solo se guardan las muestras que se enseñan; un kit no pesa, es una lista de referencias a datasets, objetos y escenas que se nombra en el chat ("usando el kit X, quiero..."). Código en `sim2chip/library.py`.

### Landing `/`

Propuesta D (usuario, 2026-10-08), en `src/pages/Landing.tsx` y `landing.css` (copiado del HTML aprobado y acotado a `.lp`). Los textos son del usuario: no reordenar ni reescribir sin permiso. Estructura: barra (How it works, Real numbers, Devices, GitHub, "Try it"); héroe "Describe it. Forge it. Spot it." con la estantería dibujada en CSS; cuatro promesas; cifras medidas en bloques de color; los tres pasos en filas alternas; dispositivos con precio y "Built with"; llamada final y pie oscuro. Pendiente: sustituir los dibujos por ejemplos reales del pipeline (etiquetado del profesor, filtro de tamaño, escena construida).

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
