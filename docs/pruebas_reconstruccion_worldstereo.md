# Pruebas de reconstrucción de la ruta WorldStereo

Ruta: foto, panorama de WorldGen, caminatas de WorldStereo, splat de gsplat, USDZ de NuRec. Caso de prueba: `photo21`.
Objetivo: un 3D tan bueno como los vídeos, también fuera de los caminos grabados.

Desde el 2026-10-26 los scripts `ws_*.py` que cita este documento están congelados en `experiments/worldstereo/closed/`
(se ejecutan con esa ruta). La receta elegida, F3c, vive en `experiments/worldstereo/wsroute/` con el driver
`experiments/worldstereo/ws_route.py`.

Regla de trazabilidad (usuario, 2026-10-22): cada prueba cambia **una sola cosa** respecto a la fila que cita como base.
Lo que mejora se queda como base de las siguientes; lo que no, se apunta y se quita.

## Tabla

Leyenda de columnas al final. "-" = no aplica, "pend." = pendiente, "n/m" = no medido.

| ID | Base | Cambio | Fase 1 | 360 al final | Fase 2 (NBV) | Guía fase 2 | Tope hueco/frame | Grupos de huecos | Reconstrucción | Huecos alumno | LPIPS ↓ | SSIM | PSNR visto / inventado | Gaussianas | Entreno | Nube | Estado y veredicto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | - | referencia | caminos | no | no | - | - | - | DefaultStrategy, 12k pasos | 21.1 % | 0.373 | 0.551 | 20.85 / 19.84 | 3.6M | 3 min | 0.93 $ (fase 1) | hecho: referencia |
| G3 | G0 | pérdida robusta (recorta el 20 % peor) | caminos | no | no | - | - | - | G0 + robusta | 21.5 % | 0.380 | 0.542 | 19.93 / 18.73 | 3.5M | 3 min | - | hecho: peor, los fantasmas siguen. Fuera |
| G5 | G0 | tope de anisotropía 10:1 | caminos | no | no | - | - | - | G0 + anti-agujas | 20.5 % | 0.378 | 0.548 | 20.63 / 19.80 | 3.6M | 3 min | - | hecho: sin efecto. Fuera |
| G1 | G0 | 30k pasos, splitting x2 | caminos | no | no | - | - | - | Default, 30k, grow 0.0003 | n/m | 0.315 | 0.563 | 21.03 / 19.70 | 13.8M | 23 min | - | hecho: mejor que G0 pero 4x de peso. Superado por G2 |
| **G2** | G0 | MCMC, 30k pasos, tope 4M | caminos | no | no | - | - | - | **MCMC 30k, 4M** | 19.5 % | **0.288** | **0.576** | 20.93 / 19.58 | 4.0M | 13.5 min | - | hecho: **mejor reconstrucción, base actual** |
| P2 | G2 | fase 2 NBV, 8 tramos | caminos | no | 8 tramos | agresiva (ángulo 20°) | 60 % | uno solo (fallo) | G2 | **5.0 %** | 0.332 | 0.572 | 20.86 / 19.73 | 4.0M | 14.6 min | ~1.0 $ | hecho: tapa huecos pero inventa cosas que contradicen la fase 1 (valla, otro SUV). No se queda tal cual |
| S1 | G2 | 360 en el sitio al final de cada caminata larga | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 | 15.7 % | 0.401 | 0.544 | 20.09 / 18.35 | 4.0M | 15.4 min | ~1.1 $ | hecho: peor. La guía de cada cuarto de vuelta está casi vacía (la nube del panorama vista desde 8 m) y WorldStereo inventa otro barrio (césped, verjas, portones) donde había setos y muro. Fuera tal cual; solo tiene sentido con una guía que tape lo visto (N2) |
| S2 | S1 | 360 en un círculo pequeño (0.5 a 1 m) | caminos | círculo | no | - | - | - | G2 | pend. | | | | | | ~1.3 $ | pendiente |
| S3 | S1 | guía de los 360 = render del splat G2, hueco solo donde no hay nada | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 | 15.3 % | 0.360 | 0.558 | 20.66 / 18.98 | 4.0M | 15.5 min | ~0.6 $ (VM, 19 min) | hecho: los giros ya obedecen (giro puro 57 a 78 %, 8.1 a 9.7° por cada 9°) y respetan la escena (muro, verja, SUV en su sitio). Mejor que S1, pero peor que G2 en los frames de la fase 1: copian las manchas del splat y su error de pose residual emborrona. A la vista, cerca de los finales de camino, parecido a G2 (más limpio en algún muro, más borroso en otros). Primera fila de la VM: generación normal |
| R1 | S3 | pesos iguales: los frames generados ya vistos pesan como el panorama (0.3 y 4.0 pasan a 1.0) | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 + pesos 1/1/1 | n/m | 0.359 | 0.577 | 20.70 / 18.58 | 4.0M | 17 min | - (local) | hecho: el coche de los giros sale más definido (hoja), pero lo que viene del panorama (muros, árboles, inicio de `reconstruct_0`) pierde la mitad de la nitidez. Desplazamiento mediano 2.1 a 1.6 px. Nitidez conservada en frames de entrenamiento 0.23 a 0.17. Los pesos no son la causa principal del borrón. Candidato a peso intermedio |
| R2 | S3 | ajuste de pose por frame generado (CameraOptModule de gsplat; recortes del panorama fijos) | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 + pose por frame | n/m | 0.373 | 0.513 | 20.49 / 18.72 | 4.0M | 16 min | - (local) | hecho: correcciones pequeñas (mediana 0.45°, 4.6 cm; p90 1.4°, 7.9 cm), nitidez igual que S3 (0.22) y los reservados peor. El desfase no es error de cámara sino contenido que no cuadra entre frames. Fuera |
| R3 | S3 | solo el primer 75 % de cada clip (el final deriva más, según el otro agente) | caminos | en el sitio, 4 x 90° | no | - | - | - | G2, 75 % de cada clip | n/m | 0.364 | 0.546 | 20.52 / 18.91 | 4.0M | 15.6 min | - (local) | hecho: algo peor que S3, y la consistencia R4 tampoco mejora. Fuera |
| R4 | - | métrica: consistencia entre generaciones con MapAnything (sin splat) | - | - | - | - | - | - | - | - | - | - | - | - | - | - (local) | hecho: ver "Consistencia entre generaciones" |
| R5 | S3 | fuente por zona: deja de enseñar el píxel que la mayoría de los otros tramos contradice | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 + máscara por mayoría | n/m | 0.379 | 0.535 | 20.22 / 18.59 | 4.0M | 16 min | - (local) | hecho: nitidez conservada 0.28 (la mejor; S3 0.23), pero más desplazamiento (2.4 px) y peor LPIPS/SSIM. La máscara borra coches enteros. Fuera, se queda R6 |
| R6 | R5 | fuente por zona: solo cuenta la contradicción de una vista más cercana (manda la cercana) | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 + máscara, manda la cercana | n/m | 0.363 | 0.551 | 20.51 / 18.72 | 4.0M | 16 min | - (local) | hecho: nitidez conservada 0.26 (S3 0.23), mismo desplazamiento (2.0 px), LPIPS igual que S3. Mejora pequeña, a juzgar en Isaac. USDZ en `splat/arms/usdz_R/` |
| R7 | R6 | una fuente por objeto: SAM 3 segmenta coches y árboles, MapAnything los asocia entre tramos y cada vóxel del objeto lo enseña solo el tramo mejor puntuado que lo ve (píxeles por nitidez) | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 + máscara R6 + reparto por objeto | n/m | 0.362 | 0.551 | 20.53 / 18.66 | 4.0M | 16 min | - (local) | hecho: "nitidez conservada" 0.39 (R6 0.26), pero es ruido de bordes, no detalle: los coches salen hechos un mosaico de trozos de tramos distintos, porque cada tramo pone el coche a unos decímetros de otro. Desplazamiento 3.8 px (R6 2.0). Descartado. Partir un objeto entre generaciones no funciona; o una sola fuente para el objeto entero, o generaciones coherentes. Hoja `splat/arms/sheet_diag_S3_R6_R7.jpg`, máscaras `depth_ma/sheet_select_views_s3.jpg` |
| K0 | S3 | control: los 4 giros del SUV (`spg_0_*`) otra vez con los ajustes de S3 y semilla nueva por tramo (mide cuánto cambia solo el ruido) | caminos | en el sitio, 4 x 90° (solo `spg_0_*`) | no | - | - | - | - | n/m | | | | | ~0.86 $ las tres K (VM, dos sesiones: 12.5 y 16 min) | hecho: frente a fase 1 0.322 (S3, los mismos 4 giros: 0.297), entre los 4 giros 0.258 (S3 0.213), deriva 26 cm. Solo cambiar la semilla mueve la consistencia de 0.03 a 0.05: es el margen de ruido. Hoja `sheet_knobs_spg_0_2.jpg` |
| K1 | K0 | memoria de WorldStereo con 20 frames por clip en vez de 8 (el máximo) (`align_nframe`) | caminos | en el sitio, 4 x 90° (solo `spg_0_*`) | no | - | - | - | - | n/m | | | | | | (en K0) | hecho: frente a fase 1 0.330, entre los giros 0.286 (K0 0.322 / 0.258), deriva 25 cm. Dentro del ruido: más memoria no cambia nada medible. Nota: los clips tienen 21 frames, el máximo es 20 (32 falló en la VM) |
| K2 | K0 | pie de foto neutro en los giros (sin izquierda ni derecha) | caminos | en el sitio, 4 x 90° (solo `spg_0_*`) | no | - | - | - | - | n/m | | | | | | (en K0) | hecho: frente a fase 1 0.368, entre los giros 0.291 (K0 0.322 / 0.258), deriva 28 cm. La única mejora, +0.046, pero del tamaño del ruido de semilla: prometedor, sin confirmar. A la vista inventa farolas y señales que no estaban |
| S4 | S1 | guía de los 360 = nube fusionada de todos los frames generados | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 | pend. | | | | | | ~0.8 $ (VM) | pendiente |
| S5 | S4 | guía de los 360 = nube fusionada más splat G2 | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 | pend. | | | | | | ~0.8 $ (VM) | pendiente |
| N1 | P2 | guía conservadora: hueco solo donde no hay nada | caminos | (lo que gane S) | 8 tramos | conservadora | 60 % | uno solo | G2 | pend. | | | | | | ~1.0 $ | pendiente |
| N2 | N1 | borrar antes de WorldGen en vez de en la panorámica: SAM 3 + borrador Qwen sobre la foto, descripción nueva (ya no nombra el SUV), panorámica WorldGen semilla 1, mismos caminos (caso `photo21_np`) | caminos | no | no | - | - | - | G2 | n/m | 0.202 | 0.713 | 23.42 / 22.53 | 4.0M | 14 min | 0.47 $ (fase 1, VM) | hecho 2026-10-23: sin coches y sin manchas de sombra; WorldGen no inventa coches. La mejor de las tres en LPIPS/SSIM/PSNR (escena distinta: WorldGen rehace el fondo y abre la calle detrás, menos seto cerca), nitidez de vistas nuevas más baja (333 frente a 551-615). Coherencia fase 1 0.430 (ruido). Entorno `panogen` rehecho (venv sobre eyeforge-qwenedit + nunchaku 1.2.0, stub de DA-2). Hojas `panogen_test/sheet_photo21_np.jpg`, `nocar_photo/sheet_photo_erase.jpg`, `photo21_np/sheet_G2_N1_N2.jpg`. USDZ `photo21_np/splat/arms/G2/splat.usdz` |
| NS | N2 | N2 más un 360 al final de cada camino (4 x 90°, guía = render del splat G2 de N2, como S3), todo junto en el splat (caso `photo21_np`) | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 | n/m | 0.274 (N2 0.225) | 0.679 (N2 0.700) | 21.90 / 19.67 (N2 22.03 / 19.82) | 4.0M | 15 min | ~0.45 $ (VM, 19 min) | hecho 2026-10-23: a la vista casi igual que N2 al final de los caminos; algo más nítido en algún muro, mismos huecos negros en el cielo. Peor en la prueba de la fase 1 (los 360 la contradicen un poco: coherencia 360 frente a fase 1 0.387, 360 entre sí 0.320, fase 1 0.430). Métricas con `ws_splat.py score` (21 frames de la fase 1, mismo render para los tres) |
| NSF | NS | NS con inventar una vez (`consensus --rule first` sobre fase 1 + 360) | caminos | en el sitio, 4 x 90° | no | - | - | - | G2 + máscara "primero" | n/m | 0.323 | 0.627 | 20.90 / 18.62 | 4.0M | 15 min | - (local) | hecho 2026-10-23: peor que NS: el asfalto sale con ondas y manchas. Fuera |
| N2 | P2 | guía de nube de puntos (como upstream) | caminos | (lo que gane S) | 8 tramos | nube de puntos | 60 % | uno solo | G2 | pend. | | | | | | ~1.0 $ | pendiente |
| N3 | P2 | tope de hueco por frame al 25 % | caminos | (lo que gane S) | más tramos | (lo que gane N1/N2) | 25 % | uno solo | G2 | pend. | | | | | | ~1.0 $ | pendiente |
| N4 | P2 | grupos de huecos por objeto | caminos | (lo que gane S) | 8 tramos | (lo que gane N1/N2) | 60 % | por objeto | G2 | pend. | | | | | | ~1.0 $ | pendiente |
| W1 | G2 | giro de vuelta más lento (90° por tramo) | caminos, vuelta lenta | no | no | - | - | - | G2 | pend. | | | | | | ~1.0 $ | pendiente |
| G6 | G2 | recortes del panorama a resolución completa | caminos | no | no | - | - | - | G2 + recortes HR | pend. | | | | | | - | pendiente (local) |
| G7 | G2 | inicialización con DA3 (Apache 2.0) | caminos | no | no | - | - | - | G2 + init DA3 | pend. | | | | | | - | pendiente (local) |
| G4 | G2 | 2DGS | caminos | no | no | - | - | - | 2DGS | pend. | | | | | | - | pendiente (local, expectativa baja) |
| T1 | G2 | malla TSDF texturizada | caminos | no | no | - | - | - | malla TSDF | pend. | | | | | | - | pendiente (local, expectativa baja) |
| F0 | G2 | NVIDIA Fixer (Cosmos 0.6B, Open Model License) como posproceso de los renders, sin reentrenar | caminos | no | no | - | - | - | G2 + Fixer al renderizar (t150 / t250 / t400) | n/m | 0.316 / 0.338 / 0.365 (G2 por el mismo camino 0.298) | 0.579 / 0.568 / 0.553 (0.583) | 20.93 / 19.74 (t150) | 4.0M | ~20 s por pasada (37 vistas, 3090) | - (local) | hecho: peor en todos los timestep. En vistas fuera del camino limpia los coches fantasma (yaw 180: el coche duplicado sale entero y limpio), pero alisa la textura (muro de piedra, seto) y apaga el color. Igualando el color el LPIPS apenas cambia (t150 0.315): lo que pierde es textura, no tono. Las rayas del seto pasan a borrón, no desaparecen. Como posproceso, fuera. Fidelidad (LPIPS crudo frente a corregido) 0.19 / 0.27 / 0.32. Hoja `splat/arms/F0/sheet.jpg`, script `ws_fixer.py`, imagen Docker `fixer-cosmos-env` (43 GB) |
| F1 | G2 | destilación Difix3D+ en una ronda: 120 vistas falsas (40 cámaras del alumno con huecos < 25 %, a 1/3, 2/3 y 3/3 desde el frame de entrenamiento más cercano), render de G2 corregido con Fixer t150, entran con peso 0.3, sin profundidad, receta G2 desde cero | caminos | no | no | - | - | - | G2 + vistas Fixer | n/m | 0.323 (G2 0.288) | 0.585 (0.576) | 21.21 / 20.11 (20.93 / 19.58) | 4.0M | 14 min | - (local) | hecho: peor en LPIPS y mejor en PSNR/SSIM, la firma de un splat más liso. Nitidez en vistas nuevas 359 (G2 615). A la vista casi igual que G2: algo más limpio alrededor de algún coche, sin arreglar las rayas del seto a 2,5 m (esas vistas tienen demasiados huecos para entrar como vistas falsas). Fixer alisa la textura y el splat la hereda. Fuera. Hoja `splat/arms/sheet_G2_F1.jpg` |
| F2 | F1 | F1 más Fixer al renderizar | caminos | no | no | - | - | - | F1 + Fixer al renderizar | n/m | | | | | - | - (local) | descartada: F0 ya muestra que corregir al renderizar empeora y F1 no mejora |
| C0 | - | control: la fase 1 otra vez, guías de siempre (solo nube del panorama), semilla fija 1024 por tramo | caminos | no | no | - | - | - | - | n/m | | | | | | (en C1) | hecho: coherencia C0 entre sí 0.584 (fase 1 original 0.439): solo la semilla mueve el conjunto 0.15 y un par de 0.11 a 0.81. Ruido grande con ~20 pares |
| C1 | C0 | memoria 3D que crece tramo a tramo (`ws_cache_job.py`): tras cada tramo, MapAnything saca su profundidad y sus píxeles hueco entran en la nube; la guía del siguiente tramo se renderiza de esa nube (el artículo de WorldStereo lo hace con WorldMirror; el código publicado no) | caminos | no | no | - | - | - | - | n/m | | | | | | ~1.2 $ las tres C (VM, una sesión) | hecho: coherencia entre sí 0.383 (C0 0.584), dentro del ruido de la semilla: no mejora visible. Ayuda donde debe (pares con `reconstruct_2/traj1` 0.13 a 0.42, 0.05 a 0.44) pero ese tramo choca ahora con `reconstruct_0/traj0` (0.33 a 0.07, 25 veces más solape). Los segundos tramos (58 a 81 % de hueco) siguen contradiciendo todo; con la misma semilla, 40k puntos de diferencia cambian una verja negra por una puerta roja. VM 30 min ~0.95 $. Hoja `cache/sheet_frames.jpg`. Ensayo local con la fase 1 vieja: huecos de la guía 0.70 a 0.25 (`reconstruct_2/traj1`), 0.59 a 0.31, 0.81 a 0.58; hoja `cache_dry/sheet.jpg` |
| C2 | C1 | orden: la vuelta (`wonder_0`) al final, tras los segundos tramos (el artículo genera las rotaciones al final) | caminos | no | no | - | - | - | - | n/m | | | | | | (en C1) | hecho: 0.372, igual que C1. Fuera |
| C3 | K0 | modelo completo (`worldstereo-memory`, 40 pasos UniPC, CFG 5) en vez de DMD (4 pasos); solo los dos primeros giros (`spg_0_0`, `spg_0_1`), fase 1 en memoria, semilla 1024 | caminos | en el sitio, 2 x 90° | no | - | - | - | - | n/m | | | | | | ~1.2 $ (VM, 38 min; ~16 min por tramo) | hecho: frente a la fase 1, igual o algo peor (0.290 / 0.312 por giro; K0 0.353 / 0.318). Entre los dos giros generados, mucho mejor: 0.645 (K0 0.178, S3 0.073), con un solo par, así que es indicio, no prueba (la semilla mueve un par de 0.11 a 0.81 en C0). A la vista, parecido a K0, algo más suave; el SUV fantasma junto al muro sigue en los tres. Hoja `full/sheet_frames.jpg`. Lectura: el modelo completo parece más coherente consigo mismo; contra una fase 1 hecha con DMD no gana. Probarlo de verdad pide la fase 1 también con el modelo completo (~9 tramos, ~2.5 h, ~4.5 $) |
| U1 | e2e photo27 | generación como el ejemplo upstream (`ws_upstream.py`): giros `view*` primero, tramos de 2 m encadenados hasta 8 m, traj1 = vistazo corto, cielo en la nube guía, memoria 3D con alineación sim3 | 30 tramos | no | no | - | guía 2-13 % de hueco con la memoria (e2e 50-88 %) | - | G2 | n/m | 0.283 (e2e 0.204, frames de prueba distintos) | 0.769 (0.824) | 21.91 / 23.28 (25.47 / 25.88) | 4.0M | 15 min | ~1.35 $ (VM 45 min) | hecho 2026-10-25: **no mejora**. Coherencia 0.238 (e2e 0.267); por tipo: view x view 0.31, step x step 0.007. Los tramos encadenados casi no avanzan (MapAnything mide un 12-42 % del recorrido pedido): WorldStereo se queda pegado al fotograma de partida, así que no alimentan la memoria y el splat sale más borroso que el e2e a 4 y 8 m. Cerca (0-2 m) parecido; giros nítidos y cielo coherente. La alineación sim3 sí funciona (desfase 0.14-0.55 m frente a 0.3-1.6 de C1). Hojas `u1/sheet_frames_u1_photo27.jpg`, `u1/sheet_compact_e2e_vs_u1_photo27.jpg` |
| A1 | G2 | inventar una vez: cada sitio lo enseña solo el primer tramo (orden de generación) que lo vio; los posteriores dejan de enseñar esos píxeles (`ws_depth.py consensus --rule first`: caminos pierden 37 %, la vuelta 74 %) | caminos | no | no | - | - | - | G2 + máscara "primero" | n/m | 0.307 | 0.549 | 21.13 / 20.85 | 4.0M | 27 min | - (local) | hecho 2026-10-23: casi idéntico a G2 en las vistas nuevas; el SUV sigue roto igual. Las métricas bajan porque los frames de prueba son de tramos que ya no enseñan. Con solo la fase 1, lo que rompe el SUV no es la contradicción entre tramos sino que nadie lo ve desde ese ángulo. Descartado. Hoja `splat/arms/sheet_G2_A1.jpg` |
| N1 | G2 | panorama sin coches: SAM 3 (car, van, truck, person) + borrador Qwen de la ruta de esfera, fase 1 generada de nuevo con los mismos caminos (caso `photo21_nocar`) | caminos | no | no | - | - | - | G2 | n/m | 0.246 | 0.614 | 22.66 / 22.22 | 4.0M | 14 min | 0.45 $ (fase 1, VM) | hecho 2026-10-23: los coches rotos desaparecen y WorldStereo no inventa coches nuevos; el resto igual que G2 (setos con agujas de cerca). Coherencia fase 1 0.410 (con coches 0.439, dentro del ruido). Queda la mancha oscura de la sombra borrada en el asfalto. Métricas no comparables uno a uno (escena más fácil). Los coches volverían como objetos 3D. Hojas `photo21_nocar/sheet_legs.jpg`, `photo21_nocar/sheet_G2_cars_nocar.jpg`, borrado `nocar/sheet_erase.jpg`. USDZ `photo21_nocar/splat/arms/G2/splat.usdz` |
| - | - | WorldMirror | - | - | - | - | - | - | - | - | - | - | - | - | - | - | descartado: la licencia excluye la UE, también para probar |

## Consistencia entre generaciones (R4)

`ws_depth.py` (MapAnything, pesos Apache 2.0 `facebook/map-anything-apache`). Cada tramo pasa por MapAnything **por separado** (un forward por tramo, nunca varios juntos: el modelo reconciliaría dos SUV distintos en uno intermedio y escondería el conflicto) con las cámaras e intrínsecas que pedimos. Luego los puntos de cada tramo se proyectan en las cámaras de los demás:

- coincide: profundidad a menos del 5 %.
- conflicto: el punto queda delante de la superficie que ve el otro (el otro ve a través de él). Detrás no cuenta (puede estar tapado).
- consistencia = coincide / (coincide + conflicto). Color: error medio (0-255) en lo que coincide, quitando el tono medio de cada frame.
- deriva de cámara: MapAnything toma nuestras cámaras como pista y devuelve las que muestra el vídeo; anclando el primer frame, cuánto se aparta cada tramo de lo pedido.

Escala: en un giro en el sitio no hay paralaje y MapAnything se inventa la escala (el mismo frame salió 1.7 veces más hondo en el forward del giro que en el de la caminata de la que parte). Los tramos que se mueven menos de 0.5 m se anclan a la nube del panorama (mediana en los píxeles que cubre); las caminatas no, porque sus cámaras métricas ya fijan la escala.

Control (cada tramo contra sí mismo con las poses de MapAnything): 0.96 a 0.98 en caminatas, 0.64 a 0.91 en giros.

| Conjunto | Consistencia (todo el clip) | 75 % del clip (R3) | 50 % del clip | Color | Deriva de cámara, mediana (peor) |
|---|---|---|---|---|---|
| fase 1 entre sí | 0.439 | 0.455 | 0.490 | 21.8 | 5 cm (9 cm) |
| S1 (giros, guía nube) frente a fase 1 | 0.154 | 0.153 | 0.160 | 36.8 | 118 cm (510 cm, `spin_1_3`) |
| S3 (giros, guía splat G2) frente a fase 1 | 0.290 | 0.264 | 0.261 | 29.4 | 24 cm (54 cm, `spg_2_3`) |
| P2 (fase 2 NBV) frente a fase 1 | 0.205 | 0.223 | 0.224 | 27.3 | 7 cm (11 cm) |
| S1 entre sí | 0.209 | 0.211 | 0.203 | 40.1 | |
| S3 entre sí | 0.319 | 0.296 | 0.273 | 32.4 | |
| P2 entre sí | 0.407 | 0.414 | 0.401 | 22.8 | |

Lectura: la métrica ordena como el ojo (S1 peor que S3; P2 contradice la fase 1). La fase 1 ya entra incoherente (0.44). Recortar el final del clip no mejora la consistencia. Hoja de las máscaras de R5 y R6: `data/worldstereo_test/photo21/depth_ma/sheet_consensus_S3.jpg`.

## Leyenda

- **Fase 1**: caminos = las 3 caminatas por las direcciones transitables (2 tramos de 4 m cada una) más la vuelta (`wonder_0`), de `ws_test.py prep`.
- **360 al final**: giro completo al final de cada caminata larga, en 4 tramos de 90° (21 frames, 4.5°/frame). Guía como la fase 1 (nube del panorama). El primer cuarto empieza en el último frame generado de la caminata y cada uno en el último del anterior (`chain_from.txt`).
- **Fase 2 (NBV)**: `ws_nbv.py plan`, tramos que empiezan en un frame de la fase 1 y se mueven o rodean lo que falta.
- **Guía fase 2**: qué se marca como hueco para que WorldStereo lo invente.
  - agresiva: vacío más todo lo que ninguna vista vio desde menos de 20°.
  - conservadora: solo lo vacío. El resto del render del splat se da como guía.
  - nube de puntos: render de una nube fusionada de los frames, hueco solo donde no cae ningún punto (lo que vio WorldStereo al entrenar).
- **Giro puro** (filas S, en el veredicto): para cada vídeo de giro, frames f y f+2 con puntos SIFT y una homografía. En una rotación pura todos los puntos encajan con ella; con traslación, los cercanos no. Control: recortes del panorama (rotación exacta) 99 % a menos de 2 px; caminata recta 37 a 47 %. S1: 24 a 60 %, giro medido de 3 a 11.6° por cada 9° pedidos, o sea que ningún giro obedeció a la cámara. S3: 57 a 78 %, 8.1 a 9.7°.
- **Tope hueco/frame**: máximo de píxeles por inventar en un frame del tramo.
- **Huecos alumno**: `ws_nbv.py audit`. Cámaras en una rejilla de 1 m dentro de 5 m de radio, 8 orientaciones, solo en sitios libres y alcanzables, sin las que miran a un seto a menos de 2 m. Un píxel es hueco si ninguna vista generada vio ese punto, sin nada delante, desde menos de 20° de la dirección del alumno (el suelo no cuenta). Mide cobertura, no calidad: lo que inventa mal también cuenta como visto.
- **LPIPS, SSIM, PSNR**: `ws_splat.py score`, sobre los 21 frames reservados de la fase 1, los mismos en todas las filas, sin el cielo, renderizando el `splat.ply` guardado (las cifras del log de `train` salen algo mejores porque no recortan el color a [0,1]; no mezclarlas). PSNR "visto" = píxeles que el panorama cubría; "inventado" = los que WorldStereo puso.
- **Filas R** (por qué el detalle de los vídeos no llega al 3D): `ws_diag.py` renderiza el splat desde las cámaras de entrenamiento de `spg_0_2`, `reconstruct_0` y `spg_1_1` y mide el desplazamiento (flujo óptico) y la nitidez (varianza del laplaciano) frente al frame. En S3 los giros pierden del 75 al 85 % de la nitidez con 2.5 a 4 px de desplazamiento mediano. En R2 se renderiza desde la cámara corregida.
- **Nube**: coste de generar en Nebius (RTX PRO 6000, eu-south1, unos 1.80 $/h). La reconstrucción es local en la 3090.

## Cómo reproducir

```
# reconstrucción de un brazo y hoja comparativa
<gsplat env>/bin/python experiments/worldstereo/ws_splat.py train --case photo21 --arm G2
<gsplat env>/bin/python experiments/worldstereo/ws_splat.py arms --case photo21
# huecos del alumno y plan de la fase 2
<gsplat env>/bin/python experiments/worldstereo/ws_nbv.py audit --case photo21 --arm G2
<gsplat env>/bin/python experiments/worldstereo/ws_nbv.py plan --case photo21 --arm G2
<gsplat env>/bin/python experiments/worldstereo/ws_nbv.py stage --case photo21
# 360 al final de las caminatas
<eyeforge-sam3d env>/bin/python experiments/worldstereo/ws_test.py spin --case photo21
# generar solo los tramos nuevos, con los ya hechos en memoria
python experiments/worldstereo/ws_vm.py run photo21 resume        # VM (por defecto desde 2026-10-22)
python experiments/worldstereo/ws_test.py launch --case photo21 --gpu rtx6000 --resume   # serverless, para una segunda escena en paralelo
# consistencia entre generaciones (MapAnything) y máscaras de fuente por zona
<mapanything env>/bin/python experiments/worldstereo/ws_depth.py depth --case photo21
<mapanything env>/bin/python experiments/worldstereo/ws_depth.py anchor --case photo21
<mapanything env>/bin/python experiments/worldstereo/ws_depth.py consistency --case photo21 [--last 0.75]
<mapanything env>/bin/python experiments/worldstereo/ws_depth.py consensus --case photo21 --views views_s3.npz [--rule closest]
# una fuente por objeto (R7): SAM 3 en los frames generados, luego el reparto por vóxel
<sam3 env>/bin/python experiments/worldstereo/ws_objects.py segment --case photo21
<mapanything env>/bin/python experiments/worldstereo/ws_objects.py select --case photo21
# ajustes de WorldStereo (filas K): las tres variantes en una sesión de VM, luego como tramos de photo21
<edge-sdg env>/bin/python experiments/worldstereo/ws_knobs.py stage
<edge-sdg env>/bin/python experiments/worldstereo/ws_knobs.py run
<edge-sdg env>/bin/python experiments/worldstereo/ws_knobs.py link
# diagnóstico: el splat desde las cámaras de entrenamiento
<gsplat env>/bin/python experiments/worldstereo/ws_diag.py S3 R1 R2
# frames de un conjunto de tramos (p. ej. sin los de otra prueba)
<eyeforge-sam3d env>/bin/python experiments/worldstereo/ws_splat.py views --case photo21 --views views_s1.npz --skip nbv_
```

Hojas: `data/worldstereo_test/photo21/splat/arms/sheet.jpg` (brazos), `.../sheet_spin.jpg` (S1), `.../sheet_spg.jpg` (S3 frente a S1), `.../splat/arms/sheet_G2_S1_S3.jpg`, `.../splat/arms/sheet_G2_P2.jpg`, `.../nbv/sheet_audit.jpg`, `.../nbv/sheet_plan.jpg`, `.../splat/arms/sheet_diag_S3_R1_R2.jpg` y `.../splat/arms/sheet_diag_S3_R1_R5_R6.jpg` (filas R), `.../nbv_P2/sheet_audit.jpg`, `.../sheet_result.jpg` (guía frente a WorldStereo por tramo). USDZ: `.../splat/arms/<ID>/splat.usdz`.

Caso `photo21_np` (N2, NS, NSF): `data/worldstereo_test/photo21_np/sheet_N2_NS_NSF.jpg` (final de cada camino, 4 direcciones) y `.../sheet_walk_map.jpg` (planta de caminos y 360, y hacia dónde mira cada frame).

Aviso 2026-10-23: hasta hoy `ws_splat.py train` puntuaba también los frames `spg_` reservados (solo excluía `nbv_` y `spin_`), así que las métricas de S3 y de las filas R no se comparan con G2. Corregido; para comparar, `ws_splat.py score` (solo fase 1).

## Notas

- 2026-10-22: las vetas del seto a 2.5 m del camino salen en todos los brazos G. Es una zona que ningún frame vio, así que la reconstrucción sola no lo arregla. Por eso las pruebas S, N y W tocan la generación.
- 2026-10-22, S1: los 360 en el sitio a 8 m del origen tienen un 40 a 90 % de guía vacía y salen escenas inventadas que contradicen la fase 1; hoja `data/worldstereo_test/photo21/sheet_spin.jpg`.
- 2026-10-22, P2: los tramos con un 40 o 60 % de guía en negro inventaron cosas que contradicen la fase 1. La máscara agresiva borraba también lo que el splat sí tenía (el lateral del SUV).
- Presupuesto en la nube aprobado el 2026-10-22: 3 $ (P2 y S1). Gastado: P2 ~1.0 $ (34 min), S1 ~1.1 $ (36 min), S3 ~0.6 $ (VM, 19 min). Total ~2.7 $.
- 2026-10-22: P2 y S1 se generaron con el trabajo serverless (`ws_test.py launch`). Desde entonces se genera en la VM (`ws_vm.py run`, `docs/nebius_vm_worldstereo_2026-10-22.md`) y el JOB ya no carga el transformador base de Wan (el checkpoint DMD lo trae entero). La primera fila generada así debe comprobar que la generación no cambia; si cambia, no es comparable con las anteriores.
- 2026-10-22, entorno de MapAnything: venv en `/media/lenena-iker/Crucial/tools/envs/mapanything` con el código en `/media/lenena-iker/Crucial/tools/map-anything` (Apache 2.0). Reutiliza el torch del entorno de gsplat con un `.pth`, así que no hay una segunda copia de torch. Pesos en `tools/hf_cache`.
- 2026-10-22, R1 y R2: los pesos no son la causa principal del borrón y el ajuste de pose tampoco. Lo que coincide entre vistas sí llega al 3D (el inicio de `reconstruct_0`, que cuadra con el panorama); el detalle que cada frame inventa a su manera se promedia. Por eso R5 y R6 eligen qué frame enseña cada zona.
- Pendiente como fila aparte: usar la profundidad de MapAnything como objetivo de profundidad y para inicializar el splat. Aún no está hecho: en las caminatas su escala se separa del panorama entre un 0 y un 40 % según el tramo, así que primero hay que decidir cuál de las dos manda.
- **Filas K** (ajustes de WorldStereo): `max_reference` (8) no se prueba porque ningún tramo ha recuperado nunca más de 8 frames de memoria, así que no limita. Los pies de foto ya se comparten por caminata (los giros usan el de su caminata), pero describen la vista hacia delante. Cada tramo de las K recibe la misma semilla (1024) y K0 mide cuánto cambia la consistencia solo con otra semilla frente a S3.

## Otros generadores probados (2026-10-23)

- **LingBot-World v1** (Ant, Wan2.2 A14B, Apache 2.0, poses de cámara; `experiments/lingbot/lb_probe.py`): un solo vídeo de 121 frames, ida 8 m por el camino 0, media vuelta, vuelta al inicio y giro para mirar la vista de partida. RTX PRO 6000 Serverless: 22 min de descarga (160 GB) y 18 min de generación (20 pasos), unos 1,6 $. Resultado: casi no avanza en la ida, a la vuelta inventa otra calle (muro con verja que no existe) y el último frame no se parece al primero (PSNR 9,9). Además pinta las teclas WASD encima del vídeo. Sin memoria útil: descartado. Hoja `data/worldstereo_test/photo21/lingbot/sheet_frames.jpg`.
