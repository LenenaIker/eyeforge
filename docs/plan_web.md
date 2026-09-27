# Plan: web de EyeForge para el jurado (PWA) y lo que el sistema necesita para ella

## Contexto

La web actual (`sim2chip/web/index.html`, 590 líneas, una sola página sin compilación) funciona, pero parece una herramienta interna. Cuando un jurado la vea tiene que parecer un producto de startup moderna. Además, debe usarse cómodamente en el móvil e instalarse como app (PWA, sin tiendas). A futuro se añadirá un servidor MCP. El hardware (ESP32-CAM) empieza la semana que viene y va aparte.

Decisiones del usuario (2026-09-27):
- **Base técnica:** React + Vite + Tailwind, con componentes tipo shadcn/ui.
- **Tono visual:** claro y amable (estilo Notion o Stripe), pensado para no expertos.
- **Partes:** landing de presentación, la app para crear un modelo y la revisión de etiquetas. Sin galería de modelos.

Reparto de modelos: el diseño (fase 1) lo hace el modelo más capaz, Opus o Fable, en una sesión corta. La construcción (fases 2 a 5) la hace Sonnet 5, que sube a Opus solo si algo se atasca.

## Arquitectura

- **Frontend** en `web/` en la raíz del repo, con su propio `package.json`. `npm run build` deja el resultado en `sim2chip/web/dist/`, que va en `.gitignore` y se genera en el Dockerfile.
- **FastAPI sigue siendo el único servidor:**
  - sirve `dist/` como estáticos;
  - devuelve `index.html` en cualquier ruta que no empiece por `/api` (fallback de SPA);
  - la API REST no cambia de forma.
- **En desarrollo:**
  - `npm run dev` con un proxy de Vite a `/api` en `localhost:8000`;
  - `sim2chip serve` en paralelo.
- **Toda la lógica vive en la API.** La web solo pinta. El servidor MCP de más adelante reutilizará los mismos endpoints (ver la fase 6).
- **Transición:** la página actual se queda como `legacy.html` hasta que la nueva la iguale en funciones, y entonces se borra.

## Fase 0: cambios en la API que necesita la web nueva (Sonnet)

Archivo: `sim2chip/server.py`. Se reutilizan `status()`, `phase()`, `cloud_job()`, `budget_left()`, `review.gather` y `review.apply`.

1. **Servir estáticos y el fallback de SPA.**
   - Hoy `@app.get("/")` lee `WEB / "index.html"`.
   - Pasa a montar `dist/` con `StaticFiles` y a devolver `index.html` en las rutas desconocidas que no sean de la API.
2. **Progreso estructurado.** Hoy la web deduce las fases leyendo `log[-30:]`. Se añaden a `status()`:
   - `started_at`;
   - la duración de cada etapa, leída de `logs/timeline.jsonl`, que ya existe;
   - una estimación del tiempo restante, a partir de las duraciones de la última ejecución en Nebius (unos 22 min).
   - Es lo que permite una barra de progreso bonita. Sigue siendo polling cada 5 s, sin SSE por ahora.
3. **Resultados listos para pintar.** El informe (`report`) ya trae el F1 y el error de conteo. Faltan:
   - las URLs de 6-9 fotos de test con las marcas del modelo superpuestas, generadas en `evaluate` o al vuelo con el mismo dibujo que `review`;
   - el tamaño del modelo en KB;
   - el enlace de descarga.
4. **Privacidad en modo público.** Hay que comprobar que `GET /api/jobs` y `GET /api/jobs/{name}` solo muestran los trabajos de la sesión del visitante, no los de otros jurados. Si hoy lista todo, se filtra por sesión.
5. **`/api/config`** devuelve también:
   - el presupuesto restante del día;
   - si hay GPU libre (`--max-runs`);
   - `public`.
   - La web usa esto para avisar antes de pulsar Entrenar.

Tests: ampliar `tests/test_server*.py` con el fallback de SPA, los campos nuevos de `status` y el filtrado por sesión.

## Fase 1: diseño (Opus o Fable, sesión corta)

Resultado: `web/DESIGN.md`, corto, más los tokens de Tailwind (colores, tipografía, radios, sombras) en `web/tailwind.config` o en variables CSS.

1. **Identidad:**
   - logotipo de texto EyeForge y un icono sencillo, un ojo o un chip, en SVG, que también servirá para la PWA;
   - un color de marca;
   - tipografía Inter o Geist.
2. **Pantallas en móvil y escritorio**, descritas como estructura de componentes:
   - **Landing:**
     - un titular con el tagline;
     - "cómo funciona" en 3 pasos: describe, fotos, chip;
     - una franja de "hecho con NVIDIA Nemotron, Isaac Sim y Nebius";
     - un ejemplo real con cifras medidas;
     - el botón "Probar".
   - **Crear modelo:** un asistente de pasos en lugar de 4 secciones apiladas.
     - **Chat de entrevista:** burbujas, adjuntar fotos (en el móvil, abrir la cámara o la galería) y una tarjeta de resumen del spec en lenguaje llano con los botones "Entrenar" y "Cambiar algo". El JSON crudo, escondido tras "Avanzado".
     - **Entrenamiento en vivo:**
       - una línea de tiempo de etapas con iconos y "powered by";
       - el tiempo transcurrido y el restante;
       - el coste en $;
       - una vista previa de renders sintéticos en cuanto existan;
       - el botón "Parar".
     - **Resultado:**
       - la cifra grande (F1, error de conteo) explicada en lenguaje llano;
       - la cuadrícula de fotos con aciertos y fallos;
       - descargar el modelo;
       - la sección "Siguiente: flashear en tu chip", que por ahora es informativa.
     - **Revisión de etiquetas:** tocar para añadir o quitar marcas (táctil en el móvil) y un botón "Aplicar y reentrenar".
3. **Estados vacíos, errores y cargas:**
   - el presupuesto agotado;
   - la GPU ocupada;
   - un fallo de etapa con un mensaje comprensible.

El usuario revisa `DESIGN.md` antes de la fase 2.

## Fase 2: armazón de la app (Sonnet)

1. `web/`:
   - Vite, React y TypeScript, con Tailwind y shadcn/ui (solo los componentes que se usen);
   - React Router con las rutas `/`, `/app` y `/app/job/:name`;
   - un cliente de la API tipado en `web/src/api.ts`, que replica los endpoints de `server.py`.
2. La landing completa, que es estática y rápida de rematar.
3. Integración con el backend:
   - el Dockerfile gana una etapa `node:20-slim` que compila y copia `dist/`;
   - `pyproject.toml` incluye `sim2chip/web/dist` como package data;
   - el README explica el modo desarrollo.

## Fase 3: la app completa, igualando la actual (Sonnet)

Se porta la lógica de `index.html` pantalla a pantalla. Las funciones de referencia son `send`, `confirmSpec`, `showTurn`, `addPhotos`, `train`, `poll`, `showResult`, `loadReview`, `drawEditor` y `applyFixes`. Se mantienen las protecciones que ya se corrigieron:
- el bloqueo `starting` contra el doble lanzamiento;
- Entrenar desactivado mientras hay una ejecución;
- el aviso `already_running`.

Cuando esté todo, se borra `legacy.html`.

## Fase 4: PWA y móvil (Sonnet)

1. `vite-plugin-pwa`:
   - el manifest con el nombre, los colores y los iconos 192, 512 y maskable, generados desde el SVG de la fase 1;
   - `display: standalone`.
2. El service worker:
   - precachea solo la app (HTML, JS, CSS e iconos);
   - `/api` siempre va a la red, sin caché, para no mostrar un progreso viejo;
   - sin modo offline para los trabajos, solo un aviso de "sin conexión".
3. Detalles de móvil:
   - `<input accept="image/*" capture>` para la cámara;
   - las fotos se reducen en el navegador antes de subirlas (en torno a 1600 px), para que la subida sea rápida con datos móviles;
   - objetivos táctiles de 44 px;
   - `safe-area` en iOS.
4. Un botón "Instalar app":
   - con el evento `beforeinstallprompt` en Android;
   - en iOS, con instrucciones de "Compartir, Añadir a pantalla de inicio".

## Fase 4b: probar el modelo en vivo en el móvil (Sonnet, siguiente)

**Por qué** (usuario, 2026-09-27): cuando el modelo termina, quien lo ha entrenado (o un jurado) lo prueba en el acto con la cámara del móvil, dentro de la misma web o app. El modelo corre en el teléfono, sin red: es la misma idea que el chip de 10 $, demostrable sin hardware. De paso cubre la prueba en Android que estaba pendiente (sin Classifier Lens).

**Qué ya existe:**
- `export/model.tflite`: entrada float32 RGB 0-255, sin normalizar (el modelo normaliza), probabilidades a la salida.
- `export/report.json`: todo lo que la página necesita:
  - `task`, `labels` (el fondo primero) e `input` ([224, 224, 3]);
  - en conteo, `output.threshold` y `output.stride_px`.
- Los dos se descargan ya con `GET /api/jobs/{name}/files/{file}`, filtrado por visitante en modo público.
- La lectura de la rejilla está en `sim2chip/counting.py::decode`:
  - una celda es objeto si su mejor clase que no es fondo llega al umbral;
  - las celdas vecinas de la misma clase forman un objeto;
  - su centro, ponderado por probabilidad, es el punto, en 0..1 de la imagen.
- **Preproceso:** `ingest` estira la foto entera a 224x224, sin recortar ni añadir bandas. En vivo hay que hacer lo mismo con el fotograma de la cámara, o el modelo verá otra geometría.

**Motor en el navegador:** LiteRT.js (`@litertjs/core`, de Google; `@tensorflow/tfjs-tflite` está abandonado en alfa).
- Usa WebGPU si hay y, si no, WASM en CPU.
- Sus archivos `.wasm` se sirven desde `dist` y entran en la precaché de la PWA, para que funcione sin conexión.
- Primer paso de la fase: una prueba mínima que cargue `model.tflite` de `exp_places` y compare su salida con la de Python en la misma foto. Si LiteRT.js falla con este modelo, el plan B es TensorFlow.js con el modelo convertido en el export.

**Pantalla** `/app/job/:name/live`, abierta con el botón "Try it live" de la tarjeta de resultado:
1. **Arranque:** descarga `model.tflite` y `report.json` una vez y los guarda en la Cache API bajo el nombre del trabajo. A partir de ahí funciona sin red. Además, "Your models" marca qué modelos están guardados en el teléfono.
2. **Cámara trasera** (`getUserMedia` con `facingMode: environment`) a pantalla completa, con un canvas encima.
3. **Bucle:**
   - Con `requestAnimationFrame`, sin cola: si la inferencia anterior no ha terminado, se salta el fotograma.
   - Se dibuja el fotograma estirado a 224x224 en un canvas oculto y se pasan sus píxeles RGB en float32.
   - Se ejecuta el modelo y se lee la salida:
     - **conteo:** la rejilla se decodifica con el mismo algoritmo que en Python y los puntos se pintan sobre el vídeo;
     - **clasificación:** se muestra la clase y su confianza.
4. **Número grande** ("3 pots"), estabilizado con la mediana de los últimos 5 fotogramas para que no parpadee. También se ve cuánto tarda cada inferencia en milisegundos y el motor usado (WebGPU o CPU): es un buen dato para el jurado.
5. **Controles:**
   - cambiar de cámara;
   - "Use a photo", que carga una foto de la galería o del ordenador. Sirve para escritorio sin cámara y para las pruebas;
   - un deslizador de sensibilidad (el umbral), que parte del de `report.json`.
6. **Errores comprensibles:**
   - sin permiso de cámara;
   - navegador sin soporte;
   - página sin HTTPS: la cámara exige HTTPS o `localhost`.

**Una sola lógica en dos lenguajes:** `web/src/live/decode.ts` es la versión TypeScript de `counting.decode`. Para que no se separen:
- un script de Python (`scripts/decode_fixtures.py`) genera rejillas de ejemplo con sus puntos esperados (`web/src/live/decode.fixtures.json`);
- una prueba de Vitest exige que TypeScript dé los mismos puntos, al cuarto decimal.

Cualquier cambio futuro en `decode` (por ejemplo, el arreglo pendiente de la maceta alta contada dos veces) obliga a regenerar los fixtures, y la prueba falla si la web no se actualiza. Conviene hacer ese arreglo en Python antes de esta fase o justo después, con los fixtures regenerados.

**Fuera de esta fase:**
- compartir un modelo con alguien que no lo entrenó (un enlace público de solo lectura);
- el modelo int8 en el navegador;
- el flasheo del chip.

**Archivos:**
- nuevos: `web/src/pages/Live.tsx`, `web/src/live/{runtime,decode,camera}.ts`, `scripts/decode_fixtures.py`;
- modificados: `web/src/main.tsx` (ruta), `web/src/pages/Job.tsx` (botón), `web/vite.config.ts` (`.wasm` en la precaché), `web/package.json` (`@litertjs/core`, `vitest`).
- El servidor no cambia.

**Verificación:**
1. `npm test`: el decode de TypeScript igual al de Python.
2. **Prueba de paridad del modelo:** en Playwright, "Use a photo" con 5 fotos de test de `exp_places`. Los puntos deben coincidir con los de `model.tflite` en Python, con tolerancia de una celda. Se compara con el modelo float, no con el int8 de `evaluate`.
3. **En el móvil real,** con `adb reverse tcp:8000 tcp:8000` y `http://localhost:8000`:
   - instalar la app;
   - abrir "Try it live" sobre `exp_places` y apuntar a macetas;
   - activar el modo avión: debe seguir funcionando;
   - apuntar el tiempo por inferencia.
4. Lighthouse sin errores nuevos. La precaché sigue por debajo de 1,5 MB, incluidos los `.wasm`.

## Fase 5: pulido para el jurado (Sonnet, con Opus para el repaso visual)

- Animaciones suaves en la línea de tiempo y en los resultados, sin exagerar.
- Accesibilidad básica: contraste, foco y `alt`.
- Lighthouse en móvil: PWA instalable y Performance por encima de 90.
- Textos revisados en inglés, para el jurado.

## Fase 6 (futura, no ahora): servidor MCP

- Un módulo `sim2chip/mcp_server.py` con el SDK oficial de Python.
- Herramientas que llaman a la misma API REST o a las mismas funciones:
  - `start_interview` y `reply`;
  - `upload_photos`;
  - `train`;
  - `job_status`;
  - `get_model`.
- La clave de autenticación es la misma que la del modo público (`SIM2CHIP_WEB_KEY`).
- No cambia nada de la web. Por eso la fase 0 insiste en que la lógica quede en la API.

## Fuera de este plan

- El hardware y el flasheo desde la web (WebSerial/esptool-js): semana que viene, con su propio plan.
- Desplegar la URL pública: después de la fase 3, junto con la imagen propia de Nebius.
- La calidad del modelo (test fijo, agrupación de celdas): en paralelo, en otra sesión.

## Archivos críticos

- Modificar:
  - `sim2chip/server.py` (estáticos, fallback, `status()`, filtrado por sesión, config);
  - `Dockerfile` (etapa de node);
  - `pyproject.toml` (package data);
  - `.gitignore` (`sim2chip/web/dist`, `web/node_modules`);
  - `README.md`, `TODO.md`.
- Nuevo: `web/` (Vite, React y TypeScript).
- Se retira al final: `sim2chip/web/index.html`.

## Verificación

1. `pytest`: los 123 tests actuales más los nuevos de la fase 0.
2. `npm run build && sim2chip serve --backend local`:
   - abrir la landing;
   - hacer la entrevista con fotos;
   - entrenar con un spec pequeño en local;
   - ver el progreso, el resultado y la revisión.
3. Modo público: `SIM2CHIP_WEB_KEY=... sim2chip serve --public`. Comprobar:
   - que pide la clave;
   - que solo se ven los trabajos propios;
   - que los avisos de presupuesto y de GPU ocupada aparecen.
4. Móvil real:
   - abrir la web desde el móvil en la misma red (`--host 0.0.0.0`);
   - instalarla como app;
   - hacer una foto con la cámara y subirla;
   - usar la revisión táctil.
5. `docker build .` compila el frontend y la imagen arranca sirviendo la app.
6. Lighthouse móvil (Chrome DevTools): PWA instalable y Performance por encima de 90.
