# Prueba: un solo splat ajustado a la photosphere y a los clips de GEN3C (2026-10-20)

Sustituye la fusión TSDF (`tsdf_gen3c.py`) por un ajuste de 3D Gaussian Splatting a todas las vistas a la vez, con la photosphere como verdad. Prueba local en la 3090, sin Nebius, con los clips que ya tenemos. Cabe antes de la congelación (2026-10-25 23:59) y, si no gana, se queda como prueba hecha.

## Por qué puede mejorar a la TSDF

| Problema de la TSDF (medido hoy) | Qué cambia con el splat |
|---|---|
| Cada fotograma pone su profundidad (MoGe, escala 1,02-1,24) y el promedio da superficies dobles | La geometría se ajusta para que todas las vistas cuadren a la vez (error de foto entre vistas). La profundidad de MoGe pasa a ser una pista débil, no la verdad |
| Color por vértice en celdas de 4 cm, borroso al lado de la textura | El color es de cada gaussiana, con detalle por debajo del centímetro donde hay vistas cerca |
| Solo 15 de los 121 fotogramas por clip | Se usan los 121 (las poses de los 121 están en el `.npz`, `w2cs_all`) |
| "La panorámica manda" se impone a posteriori, recortando caras | Se impone en la propia pérdida: la panorámica pesa más y GEN3C solo enseña de verdad donde la panorámica no vio |

Es la idea del usuario (pegar lo generado solo en los huecos, con la photosphere como verdad), pero hecha en 3D: lo que se pega es la escena que mejor explica todas las imágenes a la vez.

Lo que no arregla: si GEN3C se inventa cosas distintas en dos clips, el splat promedia (borroso o con "flotantes"). Y donde no pasó ningún clip, sigue la capa de LaMa.

## Estado del arte en una línea por trabajo

- **Lyra (NVIDIA, ICLR 2026)**: destila GEN3C en un decodificador que saca gaussianas directamente del latente. Parte de una sola imagen, no de nuestra panorámica y su geometría, y no cabe en la 3090. Aparcado.
- **DreamScene360 / HoloDreamer / LayerPano3D**: panorámica + profundidad → nube de puntos inicial → 3DGS en dos fases, con regularización de geometría en vistas sintéticas. LayerPano3D separa en capas por profundidad, como nuestro B_layers. Es el patrón que seguimos para la inicialización.
- **GaussVideoDreamer, G4Splat**: 3DGS desde vídeo de difusión, con máscaras de incoherencia y pérdida L1 enmascarada. De aquí sale la pérdida enmascarada por máscara de hueco.
- **Difix3D+ / Fixer (NVIDIA)**: limpia artefactos de render de un splat con difusión de un paso. Mejora posible si el splat sale bien pero sucio. Fuera de esta prueba.

## Herramientas

- **Entrenamiento: gsplat 1.5.3** (Apache 2.0). Entorno nuevo `uv` en `/media/lenena-iker/Crucial/tools/envs/gsplat`: Python 3.10, torch 2.4.1 cu124 y la rueda precompilada `gsplat-1.5.3+pt24cu124` (no hace falta nvcc). ~6 GB; en Crucial quedan 27 GB. Se elige gsplat y no 3DGRUT para tener el bucle propio: pesos por imagen, máscaras, pérdida de profundidad y corrección de color por imagen. 3DGRUT no trae nada de eso sin tocar su código.
- **Exportar a Isaac: 3DGRUT** (ya instalado en `/media/lenena-iker/Crucial/tools/3dgrut`): `threedgrut/export/scripts/ply_to_usd.py` pasa el `.ply` de gsplat a USDZ NuRec, que Isaac Sim 5.1 carga. Se referencia desde un `.usda` con la cúpula de cielo (no se abre el USDZ como raíz).
- **MoGe-2** en el entorno de escena de siempre, solo para precalcular la profundidad de los fotogramas (la misma escala a nuestro proxy por mediana que ya usa `tsdf_gen3c.py`).

## Datos (los que ya hay en local)

| Sitio | Clips con vídeo | Fotogramas |
|---|---|---|
| `PXL_20261018_134318253` | 3 (r3 ccw, r3 cw, r5 ccw) | 3 × 121 |
| `PXL_20261018_132424956` | 1 (r3 ccw) | 121 |

Más los 36 cortes de la panorámica en el punto de captura (12 rumbos × 3 inclinaciones, como en la TSDF), con nuestra profundidad exacta (`proxy_distance.npz`).

## Cómo se entrena (brazo principal, GS1)

1. **Inicialización**: puntos de nuestras capas (delantera de B_layers con el color de la panorámica, trasera con LaMa) más los píxeles de hueco de GEN3C proyectados con su profundidad de MoGe escalada. Sin cielo: el cielo lo sigue poniendo la cúpula de Isaac y sus píxeles no cuentan en la pérdida (máscara de MoGe y del proxy).
2. **Pérdida por imagen**: L1 + SSIM (0,8 / 0,2, lo estándar de 3DGS) con pesos por píxel:
   - cortes de la panorámica: peso 4 (como REAL_WEIGHT hoy);
   - fotogramas de GEN3C, píxel que la panorámica vio (máscara de la caché): peso 0,1. Así no compiten con la foto;
   - fotogramas de GEN3C, píxel de hueco: peso 1. Es lo que GEN3C aporta.
3. **Profundidad**: en lo visto, pérdida L1 contra nuestra profundidad (proxy, exacta). En los huecos, contra MoGe escalada, con peso bajo (0,05), solo para que no salgan flotantes.
4. **Color**: corrección afín por fotograma de GEN3C (matriz 3×4 aprendida, regularizada a la identidad). La panorámica queda fija. Absorbe los cambios de exposición de GEN3C sin teñir la escena.
5. **Densificación**: la estrategia por defecto de gsplat, 15 000 pasos (~15-25 min en la 3090). MCMC solo si salen flotantes.

**Brazo de control GS0**: el mismo splat, todas las imágenes con peso 1, sin máscaras, sin profundidad ni corrección de color. Mide si lo que gana es el splat en sí o las reglas de la photosphere.

## Cómo se mide (lo mismo que hoy, más una medida nueva)

1. **Puerta en el centro**: `psnr_centre` contra los cortes reales. Debe estar como mucho 0,5 dB por debajo de BL_scene (la misma `gate()` de `nbv.py`).
2. **Fidelidad a GEN3C en poses no vistas** (nueva): se reserva uno de cada 10 fotogramas de cada clip sin entrenar con él. Se renderizan GS1, GS0 y la malla TSDF (N01_scene) en esas poses y se compara con el fotograma de GEN3C, solo en los píxeles de hueco (PSNR y LPIPS). Es la comparación justa entre las dos formas de fusionar: cuál reproduce mejor lo que GEN3C mostró.
3. **Huecos y coherencia** en la parrilla de vistas (`views.json`, 0-5 m), con `metrics.py`.
4. **A ojo en Isaac**, como pidió el usuario: hoja con BL_scene, N01_scene (TSDF) y GS1 en el centro, a 3 m y a 5 m, los mismos rumbos, renderizada en Isaac por NuRec. Nunca con el render de fuera de Isaac como prueba final.

## Pasos y tiempos

| Paso | Qué | Tiempo | Corte si falla |
|---|---|---|---|
| 0 | **NuRec en Isaac 5.1**: un splat cualquiera (el de GS0 de una prueba rápida) a USDZ, referenciado y renderizado headless por `render_walk.py` | 1-2 h | Si Isaac no lo pinta, se para y se avisa al usuario antes de seguir (sin Isaac no se puede juzgar como hoy) |
| 1 | Entorno gsplat y `gs_data.py`: fotogramas, poses (convención GL de `w2cs_all` a OpenCV), máscaras de la caché, profundidades del proxy y de MoGe a un `.npz` por sitio | 2-3 h | |
| 2 | `gs_train.py`: el bucle de arriba, brazos GS0 y GS1, salida `.ply` + `scene.usda` en `walk/GS_scene/` | 3-4 h | |
| 3 | Entrenar los dos sitios, medidas 1-3 | ~1 h de GPU | Si GS1 no supera a la TSDF en la medida 2, se para |
| 4 | Render en Isaac y hoja para el usuario | 1 h | El usuario decide |

Total: un día y medio de trabajo, 0 $. Si el usuario lo aprueba tras la hoja, el paso siguiente (antes del 25) es meterlo en `nbv.py fuse` como alternativa a la TSDF, todavía experimental.

## Riesgos

- **GEN3C se contradice entre clips** (sobre todo al girar): el splat saldrá borroso ahí. La medida 2 lo detecta. Es el riesgo principal y no depende del método de fusión.
- **NuRec renderiza con el núcleo de 3DGUT**, no exactamente el rasterizador de gsplat: puede haber pequeñas diferencias. Se ve en la puerta del centro.
- **Componer objetos encima**: el splat no da sombras ni etiquetas. El proxy de malla sigue debajo como "matte" para las sombras (ya lo tenemos). Queda fuera de esta prueba.
- **Cobertura**: con 1-3 clips por sitio, gran parte de lo oculto sigue siendo LaMa. Esta prueba mide la calidad de la fusión, no la cobertura.

## Fuentes

- [Lyra (arXiv 2509.19296)](https://arxiv.org/abs/2509.19296), [código](https://github.com/nv-tlabs/lyra)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [3DGRUT, exportación a USD y NuRec](https://github.com/nv-tlabs/3dgrut)
- [DreamScene360 (ECCV 2024)](https://arxiv.org/abs/2404.06903)
- [LayerPano3D (arXiv 2408.13252)](https://arxiv.org/abs/2408.13252)
- [GaussVideoDreamer (arXiv 2504.10001)](https://arxiv.org/abs/2504.10001)
- [G4Splat (ICLR 2026)](https://iclr.cc/virtual/2026/poster/10007768)
- [Difix3D+ (CVPR 2025)](https://research.nvidia.com/labs/toronto-ai/difix3d)

## Resultado (2026-10-20, noche)

Hecho en local, 0 $. Scripts en `experiments/sphere_walk/`: `gs_moge.py`, `gs_train.py`, `gs_pseudo.py`, `gs_usd.py`, `gs_eval.py`, `gs_sheet.py`, `run_gs.sh`; `render_walk.py` gana capas `usdz`, `--views` y `--set`.

Brazos: GS0 (splat sin reglas), GS1 (reglas de la photosphere), GS2 (GS1 + 240 vistas de pseudoverdad renderizadas desde las capas en posiciones al azar a menos de 5,5 m, solo en los píxeles que la panorámica vio y lejos de sus bordes de profundidad).

| Sitio 134318253 (3 clips) / 132424956 (1 clip) | Capas | TSDF | GS0 | GS1 | GS2 |
|---|---|---|---|---|---|
| Centro, PSNR contra los cortes reales (Isaac) | 31,7 / 33,1 | 31,7 / 33,1 | 28,9 / 28,6 | 29,7 / 28,9 | 30,1 / 29,8 |
| Fotogramas GEN3C reservados, huecos: PSNR | 20,5 / 18,4 | 20,6 / 19,1 | 26,4 / 25,6 | 25,8 / 25,1 | 25,1 / 24,9 |
| Fotogramas GEN3C reservados, huecos: LPIPS | 0,23 / 0,25 | 0,23 / 0,24 | 0,10 / 0,09 | 0,11 / 0,10 | 0,12 / 0,11 |
| Coherencia a 3 m (diferencia media 0-255, menos es mejor) | 2,2 / 2,6 | 2,3 / 2,5 | 9,5 / 7,6 | 8,6 / 7,6 | 6,9 / 6,7 |

Lectura:
- **El splat reproduce lo que GEN3C mostró mucho mejor que la TSDF** (LPIPS a la mitad, +5 dB): la fusión TSDF perdía casi todo lo que GEN3C aportaba.
- **Fuera de las trayectorias de los clips se degrada**: el suelo cercano a 3-5 m sale emborronado y con píxeles transparentes, y el centro queda 2-3 dB por debajo de las capas (no pasa la puerta de 0,5 dB). Las vistas de pseudoverdad (GS2) solo lo mejoran un poco.
- Las reglas de la photosphere (GS1 frente a GS0) apenas cambian nada en estas medidas.
- Causa probable: los clips de GEN3C son círculos de 3 y 5 m mirando hacia dentro; la prueba de paseo mira hacia fuera a lo largo de cada rumbo, por donde ningún clip pasó. Un splat solo es bueno cerca de sus vistas.

Hallazgos técnicos:
- **NuRec en Isaac 5.1 guarda el color con la curva de tono y la deshace en HdrColor (Reinhard)**: el `.ply` va en sRGB y `render_walk.py` aplica `y/(1+y)` a las escenas con splat. Ningún ajuste de carb (`invertToneMap`, `skipTonemapping`, AA) lo cambia.
- `ply_to_usd.py` de 3DGRUT falla sin cámaras de entrenamiento: `gs_usd.py` llama al exportador con `export_cameras=False`.
- Con el peso 4 de la panorámica, el umbral de densificación por defecto de gsplat lleva a 7-20 M de gaussianas; con 0,0006 quedan 1,8-5,6 M.
- `plyfile` lee con mmap: reescribir un `.ply` sobre sí mismo lo destruye.

Hojas en Isaac: `data/sphere_test/<sitio>/walk/gs_sheet.jpg` (capas, TSDF, GS1, GS2) y `hold_sheet.jpg` (fotogramas reservados).

## Validación de la cobertura (2026-10-20, noche)

Pregunta: ¿cuánto empeora el splat al alejarse de sus cámaras de entrenamiento? Es la base del término de cobertura que se quiere añadir a `nbv.py` (Mind the Gap, 2026: el error sigue la distancia angular a la vista de entrenamiento más cercana).

Método:
- `gs_train.py --hold I:A-B --cover` deja fuera del entrenamiento tramos de clips o clips enteros (`--hold I`).
- Mide en gsplat cada fotograma reservado contra el de GEN3C, después de ajustar ganancia y desplazamiento por canal donde vio la panorámica.
- `cover_report.py` cruza esa calidad con la distancia a las cámaras de entrenamiento.
- Cinco entrenamientos, 664 fotogramas, 0 $:
  - tramos de 20, 40 y 80 fotogramas en los dos sitios;
  - dejando fuera cada uno de los 3 clips de 134318253.

Fotogramas reservados dentro de los tramos, por distancia al fotograma entrenado más cercano que mira hacia el mismo lado (los dos sitios juntos):

| Distancia | Giro medio | PSNR todo | PSNR huecos | LPIPS |
|---|---|---|---|---|
| < 5 cm | 0,3° | 28,4 | 29,3 | 0,16 |
| 5-15 cm | 1,7° | 26,7 | 26,0 | 0,18 |
| 15-30 cm | 3,9° | 26,1 | 24,8 | 0,18 |
| 30-50 cm | 6,6° | 24,9 | 22,9 | 0,20 |
| 50-70 cm | 9,3° | 24,1 | 22,0 | 0,22 |
| 70-150 cm | 11-12° | 23,4-23,5 | 21,4-21,9 | 0,24-0,31 |
| clip entero fuera (2,3-5 m, 10-18°) | | 20,8-21,0 | 18,5-18,7 | 0,37-0,38 |

Lectura:
- **La caída es rápida y temprana.** Hacia los 30 cm o 4° ya se ha perdido la mitad, y desde ~70 cm o ~10° el splat está casi en su suelo (~21 dB, lo que da sin ningún clip cerca).
  - Spearman de la distancia con el LPIPS: +0,91; con el PSNR: -0,82.
  - En los arcos, distancia y giro van juntos (correlación 0,89), así que esta prueba no los separa.
- **También empeora lo que la panorámica vio** (de 28 a 24,4 dB entre 0 y 1 m), no solo lo inventado. La geometría de la panorámica no aguanta moverse.
- **Depende del sitio:**
  - en 132424956 los huecos caen a 23,3 dB ya a 5-30 cm;
  - en 134318253 están en 26,1 dB a esa distancia.
  - La escala del núcleo hay que medirla en cada sitio, no fijarla.
- La nitidez relativa del render (0,8-0,9 frente a la de GEN3C) apenas cambia con la distancia. Lo que crece son los errores de forma y de color (LPIPS), no el desenfoque en sí.

Consecuencias para el planificador:
1. **Una cuadrícula del alumno de 1 m × 45° no se puede cubrir con 4-8 clips:** cada fotograma solo cubre un tubo de unos 30 cm y unos 5° alrededor de su pose.
   - Dos opciones:
     - sacar las poses del alumno de dentro de lo cubierto, y que el planificador maximice la variedad de lo cubierto (posiciones × rumbos);
     - o aceptar el suelo de ~21 dB fuera de los clips.
2. **La escala del núcleo se ajusta por sitio en ejecución:**
   - tras cada ronda, un entrenamiento con tramos reservados (unos 5 min en local) mide la caída;
   - la primera ronda usa la última medida guardada.
3. Las rectas con rumbo fijo cubren un tubo largo para un rumbo. Ir girando el rumbo despacio cubre más rumbos, pero cada uno una sola vez.

## Modo `cover` de `nbv.py` (2026-10-20, noche)

Petición del usuario: un círculo de 10 m de diámetro, salvando paredes. GEN3C anda hasta 5 m; las alucinaciones dan igual, lo que importa es que el conjunto sea coherente.

`nbv.py plan --mode cover [--max-hole 0.45] [--pooled] [--budget N]`:
- **Candidatas:** rectas desde el punto de captura cada 15° hasta 5 m.
  - Se cortan a 1 m de lo que esté de pie, o antes del primer fotograma con más hueco que `--max-hole`.
  - Cinco formas de mirar: a un lado, al otro, al frente, atrás, o barriendo de un lado al otro.
  - Usan el mismo pitch que el alumno.
- **Puntuación:** la calidad prevista en las vistas del alumno (rejilla de 1 m, 8 rumbos).
  - Cada vista toma `exp(-delta / delta0)` del fotograma más cercano, contando también los cortes de la panorámica y las rondas anteriores.
  - El núcleo se ajusta por sitio con `cover_report.py` (`walk/cover_fit.json`); `--pooled` usa la mediana de todos los sitios.
- **Selección:** voraz, hasta que un clip suba la media menos de 0,002, o hasta `--budget`.
- `plan.json` guarda los dB previstos tras cada clip.

En 132424956 (ronda 2, núcleo común de 10,6°), la PSNR media prevista del alumno queda así:

| Clips | PSNR prevista | Coste (2 trabajos) |
|---|---|---|
| 0 | 21,8 | |
| 4 | 23,2 | ~3 $ |
| 8 | 23,8 | ~5,3 $ |
| 12 | 24,3 | ~7,5 $ |
| 16 | 24,6 | ~9,7 $ |
| 38 | 25,5 | |

Por el oeste, el área alcanzable se acaba a 2-3 m por las paredes. Los clips que más suben la media son los que barren el rumbo.

Riesgo de coherencia:
- Clips generados en paralelo inventan cada uno lo suyo donde sus vistas se cruzan, y el splat los promedia.
- Generarlos en rondas sucesivas, con la caché sacada del splat fusionado, lo evita, pero cuesta más tiempo de reloj.
