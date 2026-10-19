# Plan de pruebas: moverse dentro de la photosphere (2026-10-18)

## Objetivo

Decisión del usuario (2026-10-18): la cámara debe poder moverse en un área de **10 m de diámetro** (5 m en cualquier dirección desde el punto de captura). Se acepta que el modelo invente lo que la esfera no vio: el fin es entrenar alumnos de visión para móvil o menos, no reconstruir el sitio con fidelidad. Sustituye el tope de 0,3 m de P.6 (`docs/roadmap_photosphere.md`) si alguna rama pasa las pruebas.

## Punto de partida (medido hoy)

Malla de proxy de `PXL_20261016_172048699` texturizada con la panorámica borrada, sin luz, en Isaac. Hoja: `data/sphere_test/sheet_walk_PXL_20261016_172048699.jpg`.

- Suelo y fachadas planas aguantan hasta 4 m.
- Huecos detrás de árboles, postes y vallas y en escalones falsos de las fachadas: 3-6 % desde el centro, hasta 15 % a 4 m.
- La esfera de cielo como malla no se renderizó; el cielo se rellenó después desde la panorámica.

Scripts del experimento (scratchpad de la sesión, a pasar a `experiments/sphere_walk/` en W.0): `prep.py`, `render_walk.py`, `sheet.py`.

## Ramas

| Rama | Qué es | Dónde corre | Licencia |
|---|---|---|---|
| **A. Base** | La malla de proxy texturizada de hoy, con huecos | 3090 | Nuestra |
| **B. Relleno 2D por vista** | Render de A y los huecos (magenta) pintados en cada vista con LaMa y con Qwen-Image-Edit (el borrador que ya usamos). Los huecos se tratan como fondo lejano: los objetos insertados siempre van delante | 3090 | Apache 2.0 |
| **C. WorldGen** | Nuestra panorámica por la entrada interna `_generate_world(pano_image=...)` → profundidad 360 (DA-2) → splat o malla, con relleno de fondo (`inpaint_bg=True`, LaMa) | 3090 (~10 GB, `low_vram`) | Apache 2.0 (sin el paso de texto con FLUX) |
| **D. Matrix-3D** | Nuestra panorámica + trayectorias propias (`--json_path`, OpenCV) → vídeo panorámico (PanoVideoGen 720p-5B) → splat 3DGS por optimización | 3090 (12 GB en modo bajo de VRAM; ~1 h por vídeo en A800) | MIT (sin Text2PanoImage, que usa FLUX.1-dev) |
| **E. GEN3C** | Caché 3D = nuestra profundidad de MoGe sobre cortes de la esfera; trayectorias radiales de 5 m → vídeos de 121 fotogramas que rellenan lo no visto | L40S en Nebius (pico ~43 GB con offload): **solo con aprobación y estimación de coste** | Código Apache 2.0, pesos NVIDIA Open Model License |

Por qué estas y no otras: las demás que encajan (Lyra 2.0, HunyuanWorld, HY-World 2.0, Voyager, FlashWorld, TrajectoryCrafter) no permiten uso comercial o excluyen la UE. La búsqueda está resumida en la conversación del 2026-10-18 y en `memory` (`project_sphere_walk_test`).

Formato de salida y cómo entra en el pipeline:

- **A y B** producen las vistas directamente; los objetos se componen como hoy (render diferencial sobre la malla de proxy).
- **C y D** producen un splat (C también malla). Un splat entra en Isaac por NuRec/3DGUT, o se renderiza fuera (gsplat) con profundidad y se compone con los objetos de Isaac por profundidad. Hay que alinear su escala y su suelo con nuestro proxy (altura de cámara y plano del suelo de `geometry.json`).
- **E** produce vídeo, no escena. Dos usos posibles, a decidir tras ver resultados: cada fotograma como fondo con su profundidad de MoGe (como la ruta de soporte), o reproyectar los fotogramas sobre los huecos de la malla para rellenar su textura una vez.

## Pruebas

### W.0 Preparación
- Pasar los scripts de la prueba de hoy a `experiments/sphere_walk/` y añadir la parrilla nueva.
- Descargar pesos y entornos aparte (conda propio por modelo, en Crucial). Matrix-3D y GEN3C piden aceptar condiciones en Hugging Face: **lo hace el usuario**.
- Comprobar que el render de un splat en Isaac 5.1 (NuRec) funciona con un splat de prueba antes de depender de él.

### W.1 Parrilla común de vistas
- Dos sitios: `PXL_20261016_172048699` (calle con árboles, el de hoy) y `PXL_20261016_172729712` (otro tipo de sitio, para no ajustar a uno).
- 8 rumbos (cada 45°) × desplazamientos 0, 1, 2, 3, 5 m hacia delante, atrás y a los lados, más puntos sueltos dentro del círculo de 5 m. Misma altura e inclinación que el spec; FOV 68°.
- La misma parrilla para todas las ramas, para comparar vista a vista.

### W.2 Medidas automáticas (por rama, sitio y distancia)
- **Huecos**: % de píxeles sin contenido (A antes de rellenar; C, D y E por zonas vacías o negras del render).
- **Fidelidad en el centro**: a 0 m, PSNR y LPIPS contra el corte real de la panorámica (la rama no debe estropear lo que sí se vio).
- **Coherencia entre vistas**: el mismo punto del suelo visto desde dos posiciones a 1 m de distancia debe tener el mismo color (reproyección con la geometría de la rama). Un relleno que cambia en cada vista enseña al alumno ruido.
- **Suelo y escala**: altura del suelo bajo la cámara y en 5 puntos medidos frente al plano de `geometry.json` (error en cm). Sin esto los objetos flotan o se hunden.
- **Coste**: segundos y VRAM por esfera y por vista.

### W.3 Hoja para el usuario
Por sitio, una hoja JPG como la de hoy (foto real | 0 | 1 | 2 | 3 | 5 m) con una fila por rama y rumbo, en `data/sphere_test/sheet_walk_<rama>_<sitio>.jpg`. El usuario juzga a ojo y elige qué ramas siguen.

### W.4 Medida en el alumno (ramas elegidas en W.3)
- Bins S.8 con el test revisado a mano, 3 ejecuciones por brazo.
- Brazos: esfera actual (radio 0,3 m) frente a la rama ganadora con radio 2 m y con radio 5 m.
- Mismo número de imágenes sintéticas en todos los brazos: solo cambia de dónde salen las vistas.
- Si el radio 5 m no empeora frente a 0,3 m, P.6 pasa a 5 m.

## Orden y criterio de parada

1. W.0 y W.1 con A (ya casi hecho) y B: un día. B es la referencia barata; si llega a 5 m sin defectos graves en la hoja, puede bastar.
2. C y D en la 3090, en paralelo con lo anterior en segundo plano (D tarda ~1 h por trayectoria).
3. E solo si B, C y D se quedan cortos a 5 m, y tras aprobar el coste de Nebius.
4. W.4 solo con las ramas que el usuario apruebe en la hoja.

Una rama se descarta en cuanto falla de forma clara en W.2: suelo con más de ~20 cm de error a 5 m, coherencia mala entre vistas vecinas o más de 1 h por esfera en la 3090 sin forma de bajarlo.

## Fuentes
- GEN3C: https://github.com/nv-tlabs/GEN3C
- Matrix-3D: https://github.com/SkyworkAI/Matrix-3D
- WorldGen: https://github.com/ZiYang-xie/WorldGen
- Registro de licencias de modelos de mundo (leído 2026-09-29): https://github.com/rishsadh/world-model-registry

## Resultados W.1-W.3 (2026-10-18)

Código en `experiments/sphere_walk/` (`run_a.sh`, `run_bc.sh`; cada script dice cómo se lanza). Dos sitios: `PXL_20261016_172048699` (calle con árboles) y `PXL_20261016_172729712` (casas bajas). 136 vistas por sitio (8 rumbos, 0-5 m en cuatro direcciones).

Rama nueva **B_layers** (no estaba en el plan): la idea por capas de WorldGen sobre nuestra malla de MoGe. Lo que está más cerca que su entorno (distancia < 0,75 de la más lejana en ~28°, sin suelo) se pinta una vez por detrás con LaMa y se pone como segunda capa a la distancia del entorno. El suelo nunca visto bajo la cámara se pinta en la capa delantera. **B_layers_lama**: B_layers y LaMa por vista solo en lo poco que deja.

Medias de los dos sitios. Coherencia: diferencia media de color (0-255) del mismo punto visto desde dos posiciones vecinas, solo en lo inventado (los huecos de A). Menos es mejor; A, sin relleno, da ~1.

| Rama | Huecos a 5 m | PSNR centro | Coherencia de lo inventado (1 m / 5 m) |
|---|---|---|---|
| A (malla tal cual) | 18,8 % | 37,2 dB | (sin relleno) |
| B_lama (por vista) | 0 % | 36,4 dB | 9,2 / 12,8 |
| B_qwen (por vista) | 0 % | 37,6 dB | 11,1 / 14,7 |
| **B_layers** | 2,2 % | 36,5 dB | **2,0 / 1,4** |
| B_layers_lama | 0 % | 36,3 dB | 7,1 / 9,5 |
| C (WorldGen) | 5,8 % | 27,3 dB | 1,4 / 0,9 |

- Los rellenos por vista tapan todo, pero inventan algo distinto en cada vista: el alumno vería el mismo sitio cambiar al moverse 1 m.
- B_layers es lo mejor: coherente y fiel, y llega a 5 m con ~2 % de huecos (franjas en esquinas de casas). Tarda ~7 s por esfera además del render.
- C: DA-2 se desvía 38-46 cm en el suelo frente a lo medido; su detector de primer plano marca un 0,16 % en un sitio y un 42 % en el otro, donde su capa trasera quedaba delante de la cámara (se dejó solo donde queda detrás, y entonces desaparece). Sus huecos los tapa estirando la malla (por eso su coherencia es baja sin ser buena imagen). PSNR del centro bajo por la resolución de DA-2. Descartada.
- D (Matrix-3D) sin probar: pide ~40 GB de pesos y en Crucial quedan ~25. E (GEN3C) pendiente de aprobar el coste en Nebius.

Fallos de Isaac encontrados por el camino (en `render_walk.py`): RTX no emite por la cara trasera aunque la malla sea de doble cara (las caras se orientan hacia el punto de captura; dos sentidos a la vez se pisan en profundidad y salen rayas); una malla de ~2 M de caras con caras al revés salía casi negra (se parte en trozos de 50 000); la geometría muy lejana (esfera de cielo a 600-2000 m) no se ve (el cielo se pone en post desde la panorámica).

Hojas: `data/sphere_test/sheet_walk_arms_<sitio>.jpg` (ramas en filas) y `sheet_walk_<rama>_<sitio>.jpg`.

Siguiente propuesto: W.4 con B_layers (radio 0,3 frente a 2 y 5 m) en bins S.8, y subir el criterio de B_layers en las esquinas antes.

### Esferas con sol a 3, 5, 8 y 13 m (2026-10-18)

Seis photospheres del usuario (`data/real/photospheres-sunny/`, extraídas con `experiments/sphere/s3_sheet.py --target car --movable person --movable bicycle`, 10-25 min cada una), prueba con `experiments/sphere_walk/run_walk.sh` (ramas A, B_layers y B_layers_lama; WorldGen, LaMa y Qwen por vista ya descartados). Medias de las seis:

| Distancia | A huecos | B_layers huecos (máx.) | B_layers coherencia | B_layers_lama coherencia |
|---|---|---|---|---|
| 0 m | 7,9 % | 0,4 % (1,8) | | |
| 3 m | 11,0 % | 1,3 % (8,8) | 2,6 | 12,8 |
| 5 m | 12,8 % | 2,1 % (12,0) | 1,8 | 11,3 |
| 8 m | 15,5 % | 4,0 % (21,8) | 1,7 | 13,5 |
| 13 m | 22,4 % | 8,4 % (89,6) | 1,7 | 19,6 |

- Hasta 5 m B_layers se ve bien en todas. A 8 m aparecen estirados y huecos detrás de escalones, muretes y setos. A 13 m muchas vistas no tienen sentido: la cámara atraviesa muros, cuestas o contenedores (un sitio llega a 20,6 % de huecos de media y una vista al 90 %).
- Conclusión: el límite real no es un radio fijo sino el espacio libre de cada sitio. Hace falta un muestreador de posiciones que solo acepte puntos a los que se llega desde el punto de captura sin cruzar el proxy y a más de ~1 m de cualquier superficie.
- PSNR del centro de B_layers 30,8 dB frente a 34,1 de A con sol (en nublado 36,5 frente a 37,2): la capa trasera asoma por las grietas de la delantera en escenas de mucho contraste. Pendiente de mirar.

### Regla de espacio libre (usuario, 2026-10-18)

Una cámara movida solo vale si se llega a ella desde el punto de captura sin pasar a menos de 1 m de nada que esté de pie (puntos del proxy que no son suelo, entre 0,2 m y la altura de cámara + 0,3 m), con el punto final incluido (`prep.reachable`). Medidas y hojas solo cuentan las alcanzables (las otras salen como BLOCKED). Esferas con sol, B_layers: 3 m 1,0 % de huecos (156 de 192 posiciones), 5 m 1,7 % (148), 8 m 2,9 % (100), 13 m 3,2 % (56).

### D (Matrix-3D) y E (GEN3C), 2026-10-18

- **D, Matrix-3D** (`matrix3d_walk.py`, entorno `matrix3d` en Crucial, 3090 local, 5B en modo de poca memoria): 4 recorridos de 13 m en `PXL_20261018_130625084`, ~14 min cada uno. Sin huecos a ninguna distancia y escenas creíbles hasta 13 m, pero borroso (panorámica 1440×720: 25 dB en el centro frente a 31-33). No trae geometría que case con la nuestra (inventa la suya): para componer objetos haría falta su reconstrucción 3D o profundidad por fotograma. La coherencia medida con nuestra geometría (12-25) no es justa con D.
- **E, GEN3C** (`gen3c_walk.py`, Nebius L40S `gpu-l40s-d` 96 GB por T5-11B, imagen `nvcr.io/nvidia/pytorch:24.10-py3`): 3 vídeos de 121 fotogramas hacia delante a 13 m (plaza a 0° y 90°, `PXL_20261018_131415152` a 0°), 20-25 min cada uno. Su caché 3D sale de MoGe escalado a nuestra profundidad, así que la cámara va en metros y encuadra como las otras ramas. El más nítido y creíble de los tres; rellena soportales y fondos. Fallos: a 13 m a 90° pierde el muro de la izquierda, y en `131415152` el recorrido cruza un banco y la imagen apenas avanza a 8-13 m. Seis intentos hasta que funcionó (pip del contenedor de NVIDIA, restricciones de versiones, numpy de MoGe, OpenCV duplicado; todo resuelto en el script). Coste aproximado: ~4-5 $ en total con los intentos.
- Hoja comparativa de las tres: `data/sphere_test/sheet_walk_gen3c.jpg`.

Conclusión provisional: para moverse lejos (8-13 m) los modelos generativos (GEN3C mejor que Matrix-3D) dan imágenes mucho mejores que nuestras capas; hasta 5 m las capas bastan, son exactas y casi gratis. GEN3C cuesta ~0,75 $ por vídeo de una dirección en Nebius y no cabe en la 3090.

### Decisión y prototipo de producción (2026-10-19)

- **GEN3C es obligatorio en producción** (usuario, 2026-10-19). Las demás ramas (capas, Matrix-3D, LaMa) solo se usan para pruebas baratas.
- Gasto real de la prueba de GEN3C del día 18: **1,24 $** con los intentos fallidos incluidos (yo había estimado 4-5 $).
- Idea de producción: GEN3C se ejecuta una vez por sitio y en lote. Sus fotogramas se pintan sobre nuestra escena de dos capas, y después solo trabaja Replicator.
  - Lote: `gen3c_batch.py`. Hace 8 recorridos por sitio, uno cada 45°, cada uno recto hasta 5 m o hasta donde permita la regla de 1 m. Cada trabajo de Nebius hace varios recorridos para cargar el modelo una sola vez, y varios trabajos corren en paralelo.
  - El trabajo parchea GEN3C con dos variables: `GEN3C_TRAJ` (nuestro recorrido) y `GEN3C_HFOV` (nuestras intrínsecas en lugar de las que estima MoGe).
  - Fusión: `fuse_gen3c.py`. La geometría sigue siendo la nuestra, así que las vistas son coherentes entre sí. Solo los téxeles que nunca se vieron (la capa trasera pintada por LaMa y el suelo nunca visto) toman el color de GEN3C. Un téxel se pinta si algún fotograma lo ve sin oclusión (z-buffer de las dos capas) y si la profundidad de MoGe en ese fotograma, escalada a la nuestra, coincide a ±15 %. Pesan más los fotogramas cercanos.
  - Medida y hoja: `run_gen3c_fuse.sh <sitio>`, rama `GL` frente a `B_layers`, solo vistas alcanzables hasta 5 m.
- Estado: en la plaza (`PXL_20261018_130625084`) los 8 recorridos llegan a 5 m. Lote lanzado el 2026-10-19 (2 trabajos L40S de 4 clips).
- La fusión ya funciona con los vídeos del día 18, pero esos usaban la trayectoria y las intrínsecas de serie, así que las cifras todavía no valen.
- Regla de GPU (usuario, 2026-10-19), en `gen3c_batch.py --gpu auto`:
  - Un trabajo con 8 clips o más va a H100 (`gpu-h100-sxm`, `1gpu-16vcpu-200gb`). Con menos, a L40S.
  - Si no se puede crear la H100, se vuelve a la L40S.
  - El umbral de 8 es provisional hasta medir un clip en H100.

### Resultado del lote (noche del 2026-10-19)

- Lote: 8 clips de 5 m en la plaza, en 2 trabajos L40S de 4 clips cada uno. 20-26 min por clip; los trabajos terminaron a las 02:49 y a las 03:04. Unas 2 h de GPU (~3,6 $ a precio de lista; ayer la factura salió por debajo de la lista).
- **GL (color de GEN3C sobre nuestras capas): descartado.** Solo pintó el 1-2 % de los téxeles ocultos, por dos motivos:
  - La geometría de nuestra capa trasera (un plano a la distancia más lejana del entorno) no coincide con lo que GEN3C muestra detrás de las columnas.
  - Con recorridos radiales apenas se ve detrás de lo que tapa.
- **TS (geometría y color de GEN3C, fusión TSDF): la rama que se queda** (`tsdf_gen3c.py`). Funciona así:
  - Funde en un volumen TSDF de Open3D (vóxel de 4 cm) los 120 fotogramas de GEN3C, con su profundidad de MoGe escalada a nuestro proxy, y 36 cortes de la panorámica (12 rumbos × 3 inclinaciones) con nuestra propia profundidad y peso 4.
  - Las caras que la esfera vio (a menos del 4 % de la distancia medida) llevan la panorámica como textura a resolución completa. El resto lleva color por vértice, de GEN3C.
  - Detrás, como respaldo, va B_layers alejado un 8 %, que solo se ve donde la fusión no llega.
  - Tarda ~40 s en local.
- Resultados frente a B_layers, vistas alcanzables:

  | Medida | B_layers | TS |
  |---|---|---|
  | Huecos a 3 m | 0,4 % | 0,1 % |
  | Huecos a 5 m | 1,2 % | 0,3 % |
  | PSNR en el centro | 30,8 dB | 28,1 dB |

  - Detrás de las columnas aparece el soportal con su profundidad, y desaparecen los escalones cercanos.
  - El PSNR baja porque el color por vértice cada 4 cm es más blando.
  - Queda moteado oscuro en el suelo en algunas vistas laterales.
- Hojas: `data/sphere_test/sheet_walk_TS_PXL_20261018_130625084.jpg` y `sheet_walk_arms_PXL_20261018_130625084.jpg`.
- Escena para la cámara libre de Isaac: `data/sphere_test/PXL_20261018_130625084/walk/TS_scene/scene.usd` (~330 MB, 12 M caras).
  - Cielo en cúpula (`sky_dome.png`) con intensidad 1140. Una cúpula de intensidad 1 sale a 0,00088 en HdrColor.
  - La cúpula latlong de Isaac muestra la longitud L hacia nuestro rumbo L − 90, así que la panorámica se gira un cuarto de vuelta.
- Siguiente paso posible:
  - Hornear el color de GEN3C en textura en lugar de por vértice (más nítido).
  - Recorridos laterales, para ver mejor detrás de lo que tapa.
  - Medir un clip en H100.

## Revisión del usuario y prueba de tiempos (2026-10-19, mañana)

- El usuario abrió `TS_scene/scene.usd` en Isaac: bien en general, pero con arcos oscuros en el suelo al mover la cámara. Causa: la capa vista se orientaba hacia el punto de captura y el suelo fusionado ondula unos milímetros; desde otra altura se veían caras por detrás (negras). Arreglo: suelo aplanado (`GROUND_SNAP_M` 3 cm, caras dobladas fuera) y la capa vista conserva el sentido propio de la malla (`"winding": "own"` en `layers.json`). Comprobado desde 2 m más alto. Centro 27,8 dB (antes 28,1); huecos iguales.
- Auditoría de cobertura (`coverage.py`): 81 puntos cada metro dentro de 5 m, 8 direcciones, 648 vistas. Píxeles: foto real 71 %, GEN3C 13,5 %, respaldo 15 %, huecos 0,07 %. Por punto, la peor dirección tiene 21-49 % de respaldo (incluye lo lejano, más allá de 25 m). Mapa en `walk/TS_scene/coverage.png`. GEN3C solo cubre lo que ven sus 8 recorridos hacia delante.
- Tiempos de anoche por clip (~21 min): generación ~14 min, texto ~1,5 min, y el resto en recargar el modelo y T5 en cada clip.
- Prueba en marcha (`gen3c_probe.py`, un trabajo L40S): un solo proceso para todos los clips (modelo y texto una vez), 18 frente a 35 pasos, y caché de varias vistas (6 rumbos x 2 inclinaciones de la esfera con nuestra profundidad, selección de búferes por fotograma) con un recorrido que gira hasta mirar atrás a 4 m.
- Resultado de la prueba (trabajo de ~1 h, L40S, unos 1,8 $ a precio de lista). Carga del modelo una sola vez: 252 s. Por clip:

  | Clip | Caché | Generación | Calidad |
  |---|---|---|---|
  | 1 vista, 18 pasos | 18 s | 564 s (incluye texto, una vez) | igual que 35 pasos a la vista |
  | 12 vistas, 35 pasos | 376 s | 884 s | gira y mira atrás mostrando la escena real |
  | 12 vistas, 18 pasos | 365 s | 456 s | al girar, edificios lejanos borrosos y con fantasmas |

  Conclusiones: 18 pasos valen para recorridos rectos, no para giros; la caché de 12 vistas tarda 6 min porque renderiza todas por fotograma (se puede elegir por geometría antes). Hoja: `data/sphere_test/sheet_gen3c_probe_PXL_20261018_130625084.jpg`.

## Escena con el giro fusionado y diseño de la siguiente prueba (2026-10-19, tarde)

- `TS2_scene` (local, 1 min de fusión): los 8 recorridos rectos más el clip de 12 vistas y 35 pasos (`tsdf_gen3c.py --add P_probe/<clip> --out TS2_scene`). Cobertura: GEN3C 13,5 a 14,1 %, respaldo 15,1 a 15,7 %, huecos 0,06 %. El usuario la ve objetivamente mejor, pero al moverse 2 m en una dirección ya se nota lo que no se generó.
- Siguiente prueba (pendiente de saldo y de aprobación del usuario):
  1. Caché más rápida: elegir por geometría los recortes que miran hacia la cámara antes de proyectar, no proyectar los 12 por fotograma (de 6 min a ~1, mismo resultado).
  2. Recorridos elegidos por el vacío (idea del usuario): auditar con `coverage.py`, elegir los recorridos (origen, rumbo y giro) que más respaldo cercano limpian, generarlos, fusionar y volver a auditar. Por rondas de 3-4 recorridos en paralelo, cada uno sobre una zona distinta, hasta que no quede respaldo cercano o se llegue al tope de clips. Así cada sitio usa los clips que necesita, sin adivinar cuántos.
  3. Recorridos con 12 vistas y 35 pasos, un proceso por trabajo de L40S.
  4. Medida: respaldo cercano en todo el círculo de 5 m y el tiempo y coste reales por sitio (estimado de ~35 min y 4-8 $).

## Planificación de recorridos por next-best-view (2026-10-19, noche)

Decisiones del usuario:

- La photosphere es **opcional**. Se sube desde la biblioteca, no desde el chat, y se construye sola en segundo plano. Una vez lista, el usuario la elige en cualquier generación.
- Calidad por fases sin bloquear nunca al usuario:
  1. Extracción y `B_layers`: el sitio queda "básico".
  2. Vuelta por el perímetro: "mejorando".
  3. Rondas dirigidas: "completo".
  
  Una generación que coge el sitio a medias usa lo que haya.
- Coste estimado visible antes de lanzar, con tope de rondas y de dinero.
- Fecha límite: 2026-10-25 a las 23:59 (regla 3 de `CLAUDE.md`). Lo que no esté listo sale como experimental.

### Qué dice la literatura

- **Vistas elegidas por sus huecos, empezando en vista conocida** (Taming Video Diffusion Prior, 2503.05082): se renderizan poses candidatas con el modelo actual, se quedan las de más huecos y la trayectoria se interpola desde una vista de entrada hasta ellas. Así el modelo de vídeo empieza con contexto conocido.
- **El proxy geométrico decide adónde ir, con comprobación de choques** (2603.13910, NavCrafter 2604.02828, OmniCam 2610.09513). Hay que alinear la trayectoria con las coordenadas del modelo de vídeo.
- **Condicionar con renders de malla, no de nube de puntos** (Matrix-3D 2508.08086): la nube da moiré y oclusiones mal resueltas. Prueba posible: meter en la caché de GEN3C el render de la malla TSDF fusionada.
- **GEN3C ya está pensado para recorridos largos**: trozos solapados un fotograma, con la caché 3D actualizada con lo generado (suplemento de GEN3C).
- **Incertidumbre en vez de huecos** (FisherRF, Next Best Sense): elegir al azar desperdicia vistas solapadas. Queda para después de la entrega; contar huecos basta.

### Principio: lo que destapa es el paralaje

- Girar sin moverse no destapa nada: lo que se ve girando ya está en la photosphere. Por eso "mirar a todos lados" desde un punto es caro y no rinde.
- Hacia delante o hacia atrás el paralaje es casi cero en el centro de la imagen (como un zoom); solo cambian los bordes.
- De lado (perpendicular a la mirada) es lo que más destapa: lo que hay detrás de árboles, postes y coches.
- No hace falta mirar a todos lados en cada punto. Basta con ver cada hueco una vez desde una pose que lo destape.

### Planificador (determinista, sin GPU)

1. **Cobertura orientada a la tarea.** Los huecos que importan son los que verían las cámaras del alumno: la rejilla de 1 m, las alturas y los rumbos de entrenamiento (`coverage.py`), no toda la geometría.
2. **Grupos de huecos.** `coverage.py` agrupa el respaldo cercano por el objeto que lo tapa (detrás del árbol X, detrás del coche Y).
3. **Candidatos por grupo.** Desplazamiento lateral o arco alrededor del objeto que tapa, con la cámara apuntando al hueco y saliendo de una zona bien cubierta. FOV ancho (`GEN3C_HFOV`) para cubrir varios huecos por clip. Regla de 1 m de paso libre contra choques.
4. **Puntuación geométrica.** Se rasteriza el proxy desde los fotogramas de cada candidato y se cuentan los píxeles de hueco que destaparía. No se usa GPU de Nebius.
5. **Selección greedy por ganancia/coste.** Se descuenta lo que ya cubren los elegidos (solapamiento) y salen 3-4 clips por ronda. Si hace falta azar, con semilla fija.
6. **Bucle:** generar (12 vistas, 35 pasos, un proceso por trabajo de L40S), fusionar con `tsdf_gen3c.py --add`, volver a auditar y repetir. Se para con menos de un 1 % de huecos en el área de 10 m o al llegar al tope de rondas o de dinero.

**Vuelta por el perímetro (primera ronda).** Se camina por el borde mirando de lado. Por cada tramo, el planificador puntúa hacia dentro (destapa la parte de atrás de lo de dentro) y hacia fuera (destapa lo de detrás de los objetos del borde). Sospecha: gana hacia fuera, porque las cámaras del alumno estarán dentro mirando hacia fuera. Lo decide la puntuación, no nosotros. Se genera por trozos con la caché encadenada.

### Calendario

| Día | Qué |
|---|---|
| 20 | Test por sitio (¿ayudan las esferas?, punto 1 de `docs/sintetico_completar_2026-10-18.md`) y planificador geométrico, sin GPU |
| 21-22 | Perímetro y una ronda dirigida en un sitio, con hoja de comparación |
| 23-24 | Trabajo en segundo plano desde la biblioteca, por fases |
| 25 | Medida con el alumno; decidir si sale opcional o experimental |

Pendiente: saldo de Nebius (unos 5 $) para los días 21-22, o rondas locales con menos pasos.

### Fuentes

- [Taming Video Diffusion Prior with Scene-Grounding Guidance (2503.05082)](https://arxiv.org/pdf/2503.05082)
- [Scene Generation at Absolute Scale (2603.13910)](https://arxiv.org/pdf/2603.13910)
- [NavCrafter (2604.02828)](https://arxiv.org/pdf/2604.02828)
- [OmniCam (2610.09513)](https://arxiv.org/html/2610.09513)
- [Matrix-3D (2508.08086)](https://arxiv.org/html/2508.08086v1)
- [GEN3C, proyecto](https://research.nvidia.com/labs/toronto-ai/GEN3C/) y [suplemento](https://research.nvidia.com/labs/toronto-ai/GEN3C/supplment.pdf)
- [FisherRF](https://arxiv.org/html/2311.17874v1), [Next Best Sense](https://arxiv.org/html/2410.04680v4)

### Implementación (2026-10-19, noche; sin ejecutar)

Código en `experiments/sphere_walk/nbv.py`, en el entorno `SIM2CHIP_SCENE_PYTHON` (Open3D en CPU, sin GPU). Pasos:

```
nbv.py audit  --place <sitio> --scene TS2_scene             # huecos del alumno, en <escena>/audit.json
nbv.py plan   --place <sitio> --round 1 --scene TS2_scene   # ronda 1 = perímetro; walk/N01/plan.json, plan.png
nbv.py launch --place <sitio> --round 1 [--jobs 2]          # enseña coste; solo gasta con --confirm (tope --max-usd 5)
nbv.py fetch  --place <sitio> --round 1
nbv.py fuse   --place <sitio> --round 1                     # tsdf_gen3c con todas las rondas -> walk/N01_scene + auditoría
nbv.py plan   --place <sitio> --round 2                     # dirigida a los grupos de huecos, sobre N01_scene
```

Cómo funciona y qué sale de la investigación:

- **Huecos.** Un hueco es un rayo de la cámara del alumno que acaba en el respaldo (`far`, `farback`; en `BL_scene`, `back`) a menos de 20 m, o en nada por debajo del horizonte. Los puntos de respaldo se guardan como vóxeles de 25 cm, con peso igual a cuántos píxeles del alumno los ven. Un clip "destapa" un vóxel si alguno de sus fotogramas lo ve.
- **Candidatos.**
  - Perímetro: arcos de 4 m en los círculos de 5, 4 y 3 m (los menores si algo estorba en el borde), mirando de lado hacia fuera o hacia dentro y en los dos sentidos, con un clip por sector.
  - Rondas dirigidas: alrededor de los 6 grupos de huecos más pesados, barridos laterales y arcos de 3 m de ancho, a 3, 5 y 8 m del grupo y desde cada 30°.
  - Todos respetan la regla de 1 m de paso libre, una zona de 8 m y un inicio al que se llega andando.
- **Filtros de la literatura.**
  - El primer fotograma debe tener menos del 3 % de hueco. ViewCrafter y Taming empiezan siempre desde una vista conocida, porque el primer fotograma de GEN3C es nuestro render.
  - Ningún fotograma puede pasar del 35 % de hueco. ViewCrafter descarta por encima del 60 % y Taming por encima del 10 %; nuestro 35 % es una suposición pendiente de medir.
- **Selección.** Greedy por ganancia marginal: lo que ya ven los clips elegidos no vuelve a contar. Es cobertura submodular, así que el greedy garantiza al menos un 63 % del óptimo con k clips. El orden de los empates es determinista.
- **Caché de GEN3C.**
  - Renders de la malla actual en los fotogramas 0, 30, 60, 90 y 120 del propio recorrido, más el inicio girado ±60°, con los huecos a profundidad 0 (máscara) para que GEN3C solo pinte lo desconocido.
  - Es la idea de Matrix-3D: render de malla, no de nube.
  - Son 7 vistas, en vez de las 12 fijas de la prueba, así que la caché debería tardar menos.
  - El runner es el de `gen3c_probe.py`: el modelo se carga una vez por trabajo.
- **Cambio frente al plan:** clips independientes en vez de encadenados. Cada tramo del perímetro arranca de un render de la escena actual, en lugar de seguir desde el último fotograma generado del tramo anterior. Así se lanzan en paralelo, no hay que cambiar el runner y la selección greedy ya evita solapes. El encadenado queda como opción si la fusión muestra tramos que no casan.
- **Coste estimado** (de la prueba del 19): 8 clips de perímetro en 2 trabajos son ~1,5 h y ~5,3 $; 4 clips dirigidos en 2 trabajos, ~0,9 h y ~3,1 $. Supera el saldo actual, así que antes del 21 hay que recargar o recortar con `--budget`.
- `tsdf_gen3c.py` ya funciona en un sitio sin el lote recto (`G_gen3c`): fusiona solo los clips añadidos.

#### Primera ejecución (2026-10-19, `PXL_20261018_130625084`, `TS2_scene`)

Las dos órdenes funcionaron a la primera, sin tocar el código. `audit` tarda 8 s y `plan` 18 s, con un pico de 2,8 GB de RAM (la escena de ~15 M de triángulos carga en 1 s). Logs en `experiments/sphere_walk/logs/nbv_*.log`.

- **Auditoría frente a `coverage.py` (Isaac).** Nuestra cifra es 8,17 % de hueco (respaldo a menos de 20 m y nada bajo el horizonte), frente al 15,7 % del respaldo total de Isaac. Si quitamos el corte de 20 m, sale 15,79 % frente a 15,72 %: la diferencia media por vista es 0,07 puntos, la máxima 1,1 y la correlación entre las 648 vistas, 0,999. Los puntos peores son los mismos, (3, -4) y (4, -3), y el 0,05 % de rayos perdidos cuadra con el 0,06 % de huecos de Isaac. Por tanto, la mitad del respaldo que ve el alumno está a más de 20 m.
- **Plan de la ronda 1.** De 96 candidatos de perímetro, todos factibles, 22 empiezan con menos del 3 % de hueco. Salen 6 clips (de los 8 sectores posibles), todos mirando hacia dentro, y el hueco esperado pasa del 8,17 % al 2,21 %:

  | Clip | Ganancia | Hueco al inicio | Hueco máximo |
  |---|---|---|---|
  | 0 `per_r5_s04_in_cw` | 2,84 % | 2,4 % | 4,6 % |
  | 1 `per_r4_s07_in_ccw` | 1,46 % | 2,4 % | 2,9 % |
  | 2 `per_r3_s01_in_ccw` | 0,82 % | 2,9 % | 7,3 % |
  | 3 `per_r3_s06_in_cw` | 0,50 % | 3,0 % | 3,6 % |
  | 4 `per_r5_s05_in_ccw` | 0,17 % | 2,4 % | 2,6 % |
  | 5 `per_r3_s00_in_ccw` | 0,17 % | 2,9 % | 3,4 % |

- **Dentro o fuera.** Gana "dentro", pero es por el filtro, no por la puntuación. Las vistas hacia fuera empiezan con un 9-18 % de hueco (mediana 13,7 %), así que ninguna pasa la regla del 3 %. Sin filtros, 8 clips hacia fuera ganarían un 7,6 % y 8 hacia dentro un 6,0 %. Hacia fuera sería ir a pintar sin una vista conocida, que es justo lo que la literatura desaconseja.
- **Primeros fotogramas** (`walk/N01/*_first.jpg`, hoja `walk/N01/firsts_sheet.jpg`): parecen vistas reales y sin agujeros, salvo vetas finas en los bordes de pilares y árboles. Los clips 0 y 4 arrancan en el mismo punto (límite entre los sectores 4 y 5) y salen en sentidos opuestos. Los clips 1 y 3 miran al mismo soportal.
- **Coste** (`estimate()`; `launch` sin `--confirm` lo bloqueó el modo automático, aunque no gasta): 6 clips cuestan 4,15 $ en 2 trabajos (1,15 h) o 3,73 $ en 1 (2,1 h). Con `--budget 4` (ganancia 5,6 % de las 5,96 %), 3,05 $ en 2 trabajos o 2,63 $ en 1. No se ha lanzado nada.
- **Pegas para la ronda 2.**
  - El agrupado de huecos da un solo grupo con el 99 % del peso: el respaldo es una lámina continua vista desde arriba. Las rondas dirigidas necesitan otro agrupado, por ejemplo por dirección desde el centro o por la profundidad del oclusor.
  - En `plan.png`, todos los puntos del alumno salen rojos porque la escala satura al 10 %.
  - `plan.json` estima el coste con 1 trabajo y `launch` con 2.
  - La ganancia esperada supone que un clip "arregla" todo vóxel que ve: es optimista.

Las cifras de esta sección (8,17 %, 6 clips, 4,15 $) usan la regla de huecos antigua y quedan sustituidas por la siguiente.

#### Regla de huecos única y plan de los seis sitios soleados (2026-10-20)

**Por qué cambió la regla.** En el mismo sitio, la regla antigua daba 0,4 % de hueco en `BL_scene` y 8,17 % en `TS2_scene`, aunque TS2 es mejor:
- En TS2 contaba como hueco todo lo que GEN3C no fusionó, aunque la panorámica lo viera bien.
- En BL solo contaba la capa de detrás de los oclusores, y las fachadas vistas muy de lado pasaban por buenas.

Por eso 131415152 salía "completo". La regla nueva es la misma para los dos tipos de escena. Un rayo del alumno es hueco si acaba en:
- **una capa de detrás de los oclusores** (`back`, `farback`) a menos de 20 m: nadie la vio, la pintó LaMa;
- **la superficie de la panorámica** (`front`, `far`) en un punto que la panorámica nunca vio (`seen.png`; en estos seis sitios lo vio todo), o visto con más de 20° de paralaje. El paralaje es el ángulo entre el rayo del alumno y el rayo de la cámara de la esfera al mismo punto. El suelo plano queda exento: aguanta desde cualquier sitio, según la prueba del 18;
- **nada**, por debajo del horizonte.

Lo que GEN3C fusionó (`seen`, `fused`) cuenta como visto. Con la regla nueva, en el mismo sitio, TS2 da 1,10 % y BL 5,47 %: el orden correcto.

**Otros cambios.**
- Un clip cuenta como destapado todo vóxel que sus fotogramas ven, sea hueco para él o no, porque la fusión lo toma desde su punto de vista.
- Modo `mixed`: el perímetro y los grupos de huecos en la misma selección. Los sitios estrechos tienen pocos arcos libres en el borde (en 132135947 chocan 76 de 96).
- La caché de GEN3C rellena el cielo con la panorámica, según la dirección del rayo. Antes iba en negro y GEN3C lo habría tomado como dato.
- `launch` acepta varios sitios separados por comas y los reparte en los mismos trabajos: cada trabajo paga una sola vez la preparación y la carga del modelo.
- `plan` borra las entradas de un plan anterior no lanzado.

**Limitaciones.**
- Lo que borró el borrador (coches, personas) cuenta como visto: no hay máscara panorámica de lo borrado.
- Los 20° de paralaje son una suposición. A 5 m de lado equivalen a un punto a unos 14 m.
- El agrupado de huecos sigue sacando un grupo enorme en algunos sitios.

**Plan por sitio** (ronda 1, `--mode mixed`; escena: TS2 donde hay GEN3C, si no BL):

| Sitio | Escena | Hueco | Clips | Ganancia | Hueco esperado | Decisión |
|---|---|---|---|---|---|---|
| 130625084 | TS2 | 1,10 % | 1 | 0,95 | 0,15 % | no lanzar: ya casi completo |
| 131415152 | BL | 1,30 % | 2 | 0,52 + 0,16 | 0,62 % | no lanzar: poco que ganar |
| 132135947 | BL | 7,00 % | 2 (dirigidos) | 1,57 + 1,56 | 3,86 % | no lanzar todavía: los dos empiezan con un soporte de cubos estirado en primer plano |
| 132424956 | BL | 6,09 % | 1 (dirigido) | 1,52 | 4,58 % | **lanzar** |
| 134318253 | BL | 11,47 % | 2 (dirigidos) | 5,02 + 4,84 | 1,62 % | **lanzar** |
| 135805749 | BL | 11,18 % | 2 (perímetro, hacia dentro) | 6,43 + 3,43 | 1,32 % | **lanzar** |

Primeros fotogramas de los 7 clips planificados de los cuatro sitios con huecos: `data/sphere_test/nbv_N01_firsts_sunny.jpg`. Cada plan está en `walk/N01/plan.json` y `plan.png`.

**Para lanzar mañana** (5 clips; `estimate()`: 1 trabajo ~3,18 $ y 1,8 h, 2 trabajos ~3,60 $ y 1,15 h):

```
nbv.py launch --round 1 --jobs 1 --max-hours 2.5 --confirm \
  --place PXL_20261018_132424956,PXL_20261018_134318253,PXL_20261018_135805749
# al acabar, por sitio:
nbv.py fetch --place <sitio> --round 1
nbv.py fuse  --place <sitio> --round 1      # N01_scene + auditoría: comparar con el hueco esperado
```

Con 132135947 serían 7 clips, ~4,28 $ en 1 trabajo, que pide `--max-hours 3`.

Fuentes añadidas: [ViewCrafter (2409.02048), planificación NBV](https://arxiv.org/html/2409.02048v1); [Taming, detalles de trayectoria](https://arxiv.org/pdf/2503.05082); [NBV como maximización submodular (2007.02084)](https://arxiv.org/pdf/2007.02084); [GEN3C, suplemento](https://research.nvidia.com/labs/toronto-ai/GEN3C/supplment.pdf).
