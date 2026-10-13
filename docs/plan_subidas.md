# Plan: subida de fotos más rápida (2026-10-13)

Pedido por el usuario tras preguntar por qué tarda la subida. Aprobado el 2026-10-13: arreglos + fases 0, 1, 2 y 4; la 3 queda fuera; se mantiene 1600 px.

## Estado (2026-10-13, sin desplegar)

Hecho:

- **Arreglos 1-3.** Reducción de una en una con `<img>` de reserva; una foto ilegible se avisa y se deja fuera (`images.ts`). El chat sube las fotos antes del turno y, si la subida falla, devuelve texto y fotos a la caja sin esperar nada. El turno se reintenta una vez ante un error de red con el mismo `act`; el servidor contesta un `act` ya visto con su primera respuesta (`kept_answer`, `arrived` en `server.py`). `/activity` dice `arrived`; si a los 15 s no ha llegado, la web deja de esperar y devuelve el texto.
- **Fase 0.** Middleware `upload_timing`: log y cabecera `Server-Timing` (read, save, library, total). Banco en `experiments/upload_bench/` (resultados en `results.md`).
- **Fase 1.** La reducción vive en `sendPhotos` (`api.ts`), por donde pasan chat, "Make it better" y biblioteca.
- **Fase 2.** Lotes de 8 MB, 3 a la vez, 3 reintentos (1, 3 y 9 s) solo ante errores de red; el primer lote de un dataset nuevo va solo y el último de un modelo, al final. Progreso real con `XMLHttpRequest` ("Uploading 12 of 40 photos"). Si falla, dice cuántas no subieron; al reintentar solo suben esas. Un dataset ya creado en un intento fallido recibe el reintento.
- **Fase 4.** `POST /api/uploads/known` (chat, dataset o modelo): el navegador calcula el `content_key` y no manda lo que ya está. El chat guarda con `save_photos`: dos `image.jpg` distintas ya no se pisan. "Make it better" con etiquetas de la prueba en vivo siempre manda esas fotos.

Medido:

- Banco, 69 fotos de móvil a 10 Mbit/s: originales 337 MB en 270 s; reducidas 52 MB en 42,5 s (en serie) y 43,5 s (8 MB de 3 en 3). El servidor tarda menos de 1 s en guardar. Con el enlace saturado, el paralelo no acelera: lo que aporta es que un corte pierde 8 MB y se reintenta solo.
- Navegador real (Chromium, 10 Mbit/s), 40 fotos: biblioteca en 28 s con un lote cortado a propósito y reintentado, las 40 guardadas a 1600 px y bien orientadas; repetir la subida no manda nada (2,8 s). Chat: el turno cortado dos veces se avisa a los 45 s (30 de subida + 15) con el texto devuelto; al reenviar se contesta una vez.
- Las fotos reducidas pesan unos 800 KB (césped con mucho detalle), no 300-500 KB.

Pendiente: probar desde el Pixel 7a en producción tras desplegar (la memoria del móvil no se reproduce aquí).

## Cómo funciona hoy

- Camino: navegador → servidor de Hetzner (disco y `library.db`) → bucket de Nebius solo cuando arranca un trabajo (`cloud_sync.py`). La subida que espera el usuario es la primera parte.
- Chat (`Interview.tsx:205`) y "Make it better" (`Job.tsx:358`) reducen cada foto en el navegador a 1600 px de lado largo, JPEG 0,9 (`web/src/images.ts`). Una foto de móvil pasa de 3-12 MB a 300-500 KB.
- Datasets de la biblioteca (`Library.tsx:173-174` → `createDataset` / `addDatasetPhotos` en `api.ts`) **no reducen**: suben el original.
- Las fotos van en lotes de hasta 40 MB, uno detrás de otro (`BATCH_BYTES`, el nginx del servidor corta a 100 MB por petición). En el chat el último lote lleva el total y dispara el turno de la entrevista.
- En el chat las fotos adjuntas se suben al pulsar enviar, no al adjuntarlas.
- La biblioteca ya guarda cada foto una sola vez por contenido (`content_key`: sha256 de los bytes, 16 caracteres), pero solo lo sabe después de recibirla.
- Fallo conocido: `interview_photos` (`server.py:811`) guarda por nombre de archivo, así que dos fotos con el mismo nombre (`image.jpg` en iPhone) se pisan en la carpeta de la sesión. `save_photos` sí lo evita.

## Fallo visto en producción (2026-10-13)

Prueba del usuario en el chat desde el Pixel 7a. Eran **40 fotos**, y llegaron 11: el primer lote, de unos 36 MB, porque los lotes son de 40 MB. El segundo lote falló por red y no consta en nginx, así que se cortó antes (Cloudflare o el móvil). Por eso nunca se mandó el texto. Lo que sigue se escribió creyendo que eran 11.

- **Las fotos llegaron sin reducir** (4032x2268, 2-4,5 MB, 36 MB en total; 21 s de subida). `prepareImages` falló en silencio y mandó los originales. Sospecha sin confirmar: decodifica las 11 a la vez (`Promise.all`, unos 36 MB de memoria cada una) y el móvil se queda sin memoria.
- **El mensaje que va después de las fotos (el turno) nunca llegó al servidor:** no aparece ni en nginx ni en la app. El navegador dio un error de red ("Failed to fetch"). La web lo tomó por un corte de Cloudflare, esperó 10 minutos la respuesta de un turno que no existía y luego enseñó el error. Desde el servidor no se ve por qué falló la red.
- **No fue el límite de Cloudflare.** Cloudflare corta a los 100 s si el servidor no responde (en el plan gratuito no se puede cambiar), y la web ya se recupera de eso preguntando la respuesta al servidor. Nginx espera 600 s. Lo pendiente de Cloudflare en `TODO.md` es la regla de rate limiting, no el tiempo.
- La GPU caliente que se pidió con las fotos se apagó sola (0,16 $).

Arreglos propuestos:

1. Reducir las fotos de una en una. Si `createImageBitmap` falla, probar con `<img>`. Si aun así no se puede, avisar en vez de mandar el original sin decir nada.
2. Que `/activity` diga si el turno llegó al servidor. Si a los 15 s no ha llegado, la web deja de esperar y ofrece reintentar, conservando el texto. Las fotos ya están subidas y no se vuelven a mandar.
3. Reintentar una vez solo ante un error de red. Para que sea seguro, el servidor tiene que ignorar un turno con un `act` que ya ha visto (hoy no lo hace).

## Fases

### 0. Medir antes de tocar (medio día)

- Tiempos por petición en el servidor: bytes recibidos, tiempo de lectura, de guardado y de la biblioteca, en el log (y en la cabecera `Server-Timing` para verlo en el navegador).
- Banco de pruebas `experiments/upload_bench/`: sube un dataset real (las 74 fotos del césped, originales de móvil) al servidor local y al de producción con la conexión limitada a 10 Mbit/s (`curl --limit-rate`), con originales y reducidas, en serie y en paralelo. Sale una tabla de segundos por caso.
- Así sabemos cuánto es red y cuánto es servidor antes de elegir.

### 1. Reducir en todos los caminos (una hora)

- Mover `prepareImages` dentro de `api.ts`, en el único sitio por donde pasan todas las fotos (`sendInBatches`, `createDataset`, `addDatasetPhotos`), para que ningún camino nuevo se olvide.
- Se mantiene 1600 px. No bajar a la resolución del modelo: el profesor (SAM 3 a ~1000 px, SigLIP a 384-512), photo2scene, la revisión a ojo y un reentreno para otro hardware necesitan detalle.
- Ganancia esperada en datasets: ~10 veces menos datos.

### 2. Subidas en paralelo con reintento y progreso (un día)

- Lotes más pequeños (unos 8 MB) y 3 a la vez. En el chat, el lote que dispara el turno sale el último, cuando los demás han terminado; en la biblioteca, el primero crea el dataset y el resto van en paralelo.
- Cada lote que falla por red se reintenta hasta 3 veces con espera creciente; si sigue fallando, el usuario ve cuáles no subieron y puede reintentar solo esas.
- Barra de progreso real (bytes enviados, con `XMLHttpRequest`, porque `fetch` no informa de la subida).
- El servidor ya atiende las subidas en hilos (handlers síncronos); comprobar con el banco que 3 a la vez no lo saturan.

### 3. Empezar a subir al adjuntar (medio día)

- Como en WhatsApp o Claude: la foto empieza a subir en cuanto se adjunta en el chat, mientras el usuario escribe. Al enviar solo queda esperar lo que falte.
- Pide una ruta para quitar una foto ya subida si el usuario la quita del adjunto antes de enviar.
- La GPU caliente se pediría al adjuntar en vez de al enviar (hoy se pide al subir): adelanta el arranque, pero si el usuario se arrepiente se gasta. Decisión del usuario.

### 4. No subir lo que ya está (medio día)

- El navegador calcula el sha256 de cada foto ya reducida (`crypto.subtle`, mismo `content_key` que el servidor) y pregunta cuáles tiene ya ese dataset o esa sesión; solo sube las que faltan.
- Útil al reintentar tras un corte o al volver a subir una carpeta con fotos nuevas mezcladas.
- De paso se arregla el fallo de nombres repetidos: el chat guarda por contenido, como la biblioteca.

### 5. Después de la entrega (no ahora)

- **Subida directa al bucket** con URLs firmadas: quita el servidor de en medio. Hoy la biblioteca y las miniaturas viven en el disco de Hetzner, así que obliga a cambiar dónde se guardan las fotos. Demasiado riesgo antes del 30.
- **Subidas que se reanudan** (protocolo tus): con lotes pequeños y reintento (fase 2) y la deduplicación (fase 4) se cubre casi todo lo que aporta.
- **WebP o AVIF** en vez de JPEG (25-35 % menos): hay que comprobar que todo el pipeline los lee (Pillow, SAM 3, SigLIP, photo2scene, ingest).
- **Bajar a 1280 px y calidad 0,85** (otro 35-40 % menos): solo si el banco del profesor (`docs/profesor_modelos.md`) no empeora.

## Orden propuesto

0 → 1 → 2 → 4 → 3. La 1 es la que más ahorra y es casi gratis; la 0 dice si la 2 merece el día. La 3 cambia el comportamiento del chat y de la GPU, así que va la última y con el visto bueno del usuario.

## Decisiones del usuario

1. ¿Qué fases entran ahora? (propuesta: 0, 1, 2 y 4)
2. Fase 3: ¿subir al adjuntar, y pedir la GPU ya entonces o esperar al envío?
3. ¿Se mantiene 1600 px o se mide 1280 px?
