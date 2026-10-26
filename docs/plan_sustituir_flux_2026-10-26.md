# Sustituir FLUX.1-Fill-dev en la panorámica de la ruta F3c (2026-10-26)

Decisión del usuario (2026-10-26): FLUX.1-Fill-dev tiene licencia no comercial y **se sustituye sí o sí**, aunque sea después de la congelación. Este trabajo va aislado: primero se mide la diferencia con un par de pruebas y después se integra.

## Dónde está FLUX y por qué importa

- Está en un solo paso de la ruta F3c: `pano`, en `experiments/worldstereo/wsroute/pano.py`.
  - Funciones `fill_model()` y `pano(case, seed)`.
  - Usa el LoRA `img2scene` de WorldGen (Apache) encima de FLUX.1-Fill-dev, como transformer int4 de nunchaku (`mit-han-lab/svdq-*-flux.1-fill-dev`).
  - Corre en `SIM2CHIP_PANOGEN_PYTHON`.
- Qué hace el paso:
  1. Coloca la foto borrada en un lienzo equirectangular de 1600x800 (`place()`): el campo de visión sale del EXIF (`data/panogen_test/in/<case>.json`), la foto va en el horizonte y el centro de la panorámica es +x.
  2. FLUX rellena el resto con el prompt `data/panogen_test/in/<case>.txt` (`gen_pano_fill_image`: 50 pasos, guía 30, mezcla de la costura).
  3. `paste()` vuelve a pegar la foto a 4096 px.
- Salida: `data/panogen_test/out/<case>/pano_s<seed>.jpg` y `pano_s<seed>_paste.jpg`. La ruta lee el `_paste`.
- Todo lo que viene después depende de esa panorámica, así que no se puede borrar sin sustituirla:
  - **WorldStereo** arranca con 27 recortes suyos como memoria (`walks.py`, `pano_bank`).
  - **Los paseos** salen de la profundidad que mide MoGe-2 sobre ella.
  - **Las medidas** (`splat`, `depth`, `conflict`, `student`) la usan como prueba y para la escala.
  - **El paquete y el render de entrenamiento** la usan como cielo (`sky.jpg`, `sim2chip/walkscene/background.py`).

## Alternativas comerciales (buscadas el 2026-10-26)

| Opción | Base | Licencia | Entrada | Veredicto |
|---|---|---|---|---|
| [mickmumpitz/QWEN-EDIT_360](https://huggingface.co/mickmumpitz/QWEN-EDIT_360) | Qwen-Image-Edit-2509 (Apache 2.0) | MIT; datos CC0 (Poly Haven) | una foto → 360° equirectangular. Prompt: "Create a 360 panoramic image of [scene]. Keep the style the same." Relleno ajustable alrededor de la foto | **Candidata principal.** Qwen-Image-Edit ya está en la ruta (`SIM2CHIP_ERASE_PYTHON`, `erase_qwen.py`, 2511 en nf4) |
| [Gogodr/qwen-image-2.1-edit-pano360-lora](https://huggingface.co/Gogodr/qwen-image-2.1-edit-pano360-lora) | Qwen-Image-2.1 | sin licencia publicada | 1-3 fotos del mismo sitio → 360°, 1536x768 | Descartada mientras no publique licencia |
| [ProGamerGov/qwen-360-diffusion](https://huggingface.co/ProGamerGov/qwen-360-diffusion) | Qwen-Image / 2512 | MIT | solo texto | No sirve: no parte de la foto |
| DiT360 y los LoRA 360 de ComfyUI sobre FLUX.1-dev | FLUX.1-dev | no comercial | | Mismo problema |

Antes de usar la candidata hay que confirmar la licencia en su página y anotarla aquí: los LoRA de la comunidad a veces la cambian.

**Confirmada el 2026-10-26:** `license: mit` en la ficha de Hugging Face (repo sha `f7626f2`, archivo `251018_MICKMUMPITZ_QWEN-EDIT_360_03.safetensors`), base declarada Qwen-Image-Edit-2509 y datos de Poly Haven (CC0). El workflow del autor usa además el LoRA Lightning de lightx2v (Apache 2.0).

## Pruebas

Casos: `f3c_photo21`, `f3c_photo47` y `f3c_photo66`. Ya tienen la panorámica de FLUX en `data/panogen_test/out/<case>/pano_s0_paste.jpg` y la foto borrada en `data/panogen_test/in/<case>.png` (con `.json` y `.txt`). photo21 es la referencia de calidad (`project_scene_route_reference`).

**Prueba 1 (local, 3090, gratis): solo la panorámica.**
- Script aislado en `experiments/panogen/qwen360_test.py`. No toca `wsroute/pano.py` hasta que la prueba convenza.
- Corre en el entorno del borrado (`SIM2CHIP_ERASE_PYTHON`: diffusers con `QwenImageEditPlusPipeline`).
  - Probar primero con la 2511 que ya tenemos y el LoRA.
  - Si el LoRA no casa con la 2511, bajar la 2509 (Apache).
- Brazos, con la misma foto colocada y la misma semilla:
  - **A:** la foto colocada en el lienzo 2:1, como hoy (`place()`), y el LoRA rellena el resto.
  - **B:** la foto tal cual con margen, como espera el LoRA, y después se vuelve a colocar la foto con `paste()`.
  - En los dos casos se pega la foto a 4096 px con el mismo `paste()`.
- Hoja de comparación por caso: FLUX frente a A frente a B. Para cada uno:
  - la panorámica entera;
  - cuatro vistas de 90° (frente, izquierda, espalda, derecha), recortadas con `wsroute.camera.crop`;
  - lo que se mide: horizonte recto, costura izquierda-derecha cerrada, la foto en su sitio sin desfase de escala, suelo continuo, sin objetos grandes inventados cerca y el tiempo por panorámica.
- Guardar en `data/panogen_test/out/<case>/qwen360/` y la hoja en JPG para el móvil.

**Resultado de la Prueba 1 (2026-10-26)**, con la receta del workflow del autor (LoRA 0,9, Lightning 4 pasos de la 2511, 2048x1024 y pasada de costura) sobre la 2511 nf4:
- Gana A. B inventa objetos grandes cerca (farola, edificio de cristal, coche).
- A cierra la costura (0,56-0,74 frente a 0,62-1,34 de FLUX). Horizonte recto y suelo continuo.
- Tiempo: unos 60 s por panorámica (31 s la panorámica y 29 s la costura). FLUX tarda 42-50 s.
- Repetir FLUX con la misma semilla da casi lo mismo (diferencia media de 3-5 sobre 255).
- **Qwen redibuja también la foto, y a otro campo de visión.** En A la dibuja con un fx ×1,39-1,49 (más pequeña que el EXIF), así que `paste()` con el EXIF la pega grande y con fantasmas.
  - El script ajusta fx, yaw, pitch, roll y una distorsión radial k1 sobre la panorámica generada y pega la foto donde Qwen la dibujó. La correlación con la foto sube de 0,47-0,61 a 0,94-0,96.
  - En A sale k1 de 0,09-0,37 (más en la foto más ancha, photo21). En B, que recibe la foto plana, k1 ≈ 0: el LoRA proyecta como una cámara estenopeica sobre un equirectangular, no como un ojo de pez. La k1 de A viene de proyectar dos veces (la entrada ya es equirectangular).

**Prueba 2 (si la 1 convence; VM de Nebius, ~1 $, con aprobación del presupuesto): la escena entera.**
- Caso nuevo `q360_photo21` con la mejor panorámica de Qwen. Ruta F3c completa: `ws_route.py init/run --until bundle`.
- Comparar con `f3c_photo21`:
  - `conflict/F3c/grid.json` (hole, nodata, conflict);
  - las hojas de la rejilla;
  - el `card.jpg` del paquete.
- La ejecución no se toca a mano (`feedback_e2e_runs_hands_off`).

**Resultado de la Prueba 2 (2026-10-26)**, caso `q360_photo21` (brazo A con la pegada ajustada, sin repetir el borrado), VM ~0,91 $:

| F3c | huecos | sin datos | conflicto | conflicto 0-2 m | suelo libre | vistas |
|---|---|---|---|---|---|---|
| FLUX (`f3c_photo21`) | 0,65 % | 1,36 % | 23,1 % | 12,4 % | 171 m² | 1048 |
| Qwen 360 (`q360_photo21`) | 0,41 % | 1,08 % | 32,0 % | 23,1 % | 248 m² | 1496 |

- Con Qwen hay menos huecos, más suelo libre y un `card.jpg` más nítido y fiel a la foto.
- El conflicto entre los tramos de WorldStereo sube unos 9 puntos en todos los radios. Queda al nivel de photo47 y photo66 con FLUX (~33 %).
- Hoja de comparación: `data/worldstereo_test/q360_photo21/compare_f3c.jpg`. Falta verla en Isaac.
- Arreglo hecho durante la prueba: `wsroute/splat.py` importaba `ssim_map` de `sphere_walk/gs_train.py`, que necesita el paquete `photo2scene` retirado en `c161e6b`. Ahora la función vive en `splat.py`.

## Integración (plan, 2026-10-26)

Decisión del usuario: Qwen 360 se queda (brazo A, receta de la Prueba 1). Lo que sigue lleva al pipeline exactamente la receta medida, sin mejoras sin medir.

### Lo que hay que saber (revisado en la web el 2026-10-26)

1. **Licencias de todo lo que corre:**
   - el LoRA 360 es MIT;
   - Qwen-Image-Edit-2511 y Qwen2.5-VL son Apache 2.0;
   - el Lightning de la 2511 (lightx2v) es Apache 2.0 en su ficha.

   Las tres están confirmadas. No queda nada no comercial en el paso.
2. **El LoRA está entrenado sobre la 2509 y lo usamos en la 2511.** Ni el autor ni Qwen dicen nada de esa compatibilidad. Las Pruebas 1 y 2 dicen que funciona. Si algún día cambia la base, hay que volver a medir.
3. **Cómo redimensiona la entrada la pipeline (diffusers 0.41).** La salida es la que se pide (2048x1024). La imagen de entrada se reescala a ~1 MP para el VAE y a 384² para el modelo de visión, así que el detalle de la foto se pierde por fuerza. Por eso hace falta la pegada.
4. **Scheduler de Lightning.**
   - Los autores de Lightning destilaron con shift 3. Recomiendan `base_shift = max_shift = log(3)` y `shift_terminal = None`.
   - Las pruebas usaron el scheduler de la 2511 (0,5/0,9 y `shift_terminal` 0,02). A 2048x1024 sale un shift efectivo de ~2,5.
   - Se mantiene el que se midió. Cambiarlo sería otra variante que habría que medir, y no hace falta.
5. **Costura.** La literatura usa padding circular en el VAE o en el latente (360Anything, PAR). Pero ya cerramos la costura con la pasada del autor (rodar e inpaint), y medida es mejor que la de FLUX. No se toca.
6. **bitsandbytes nf4 en Blackwell (VM RTX PRO 6000, sm_120).** No he encontrado confirmación oficial. Torch 2.7.1+cu128 sí trae sm_120.
   - El riesgo ya existe con el borrador: es el mismo entorno y el mismo nf4. Se comprueba una vez en 2d y sirve para los dos.
   - Con 96 GB cabe en bf16 sin nf4, pero los píxeles cambian respecto a lo medido. En la VM se usa lo mismo que use el borrador.
7. **Polos.** No están medidos aparte. Es donde fallan todos los modelos de 360. La Prueba 2 (cielo del paquete, recortes del banco) salió bien, así que no hay acción.

### Diseño

El paso `pano` se parte en dos, porque el entorno del borrado no tiene OpenCV ni SciPy y el ajuste de la pegada los necesita (igual que `erase`/`publish`):

| Paso | Entorno | Hace | Hecho cuando existe |
|---|---|---|---|
| `pano` | `erase` | coloca la foto por EXIF en el lienzo 2048x1024, gris fuera (brazo A); Qwen + LoRA 360 (0,9) + Lightning, 4 pasos, cfg 1; pasada de costura; escribe `pano.json` (prompt, semilla, modelo, tiempos) | `pano_s<seed>.jpg` |
| `pano_paste` | `scene` | ajusta fx, yaw, pitch, roll y k1 por NCC y pega la foto a 4096 px donde Qwen la dibujó | `pano_s<seed>_paste.jpg` |

- **`wsroute/pano.py`:**
  - De `erase_qwen.load` se separa la carga de la base (nf4 guardado o cuantizado, bf16, en GPU o con descarga a CPU). El borrador sigue igual y `pano` añade encima sus adaptadores, `pano360` y `lightning`.
  - Pasan del script de prueba `edit`, `seam_mask`, `fix_seam`, `place_fit`, `ncc`, `fit_photo` y `paste_fit`.
  - Salen `stub_pytorch3d`, `worldgen` y `fill_model`, y también `paste` con EXIF.
  - `place()` pasa a muestrear con numpy en vez de `cv2.remap`, para poder correr en el entorno del borrado. Un test comprueba que da lo mismo que la versión con cv2.
- **Si el ajuste falla** (NCC < 0,8; el peor medido fue 0,86): no se pega y `_paste` es la panorámica de Qwen escalada a 4096. Al usuario le parecía bien también sin pegada. Queda apuntado en `pano.json`.
- **Prompt:** el del autor con la primera frase de `<case>.txt` (sale del paso `caption`).
- **Ruta:**
  - `steps.py`: el paso `pano` corre en el entorno `erase` y se añade `pano_paste` en el `scene`.
  - `ws_route.py`: subcomando `pano_paste`.
  - `paths.py`: fuera `"pano"` de `ENVS` y la línea de `SIM2CHIP_PANOGEN_PYTHON`.
  - `config.py`: el comentario de `pano_seed`.
  - `__init__.py` y `walks.py`: docstrings.
- **Casos viejos:** `f3c_photo21/47/66` tienen sus pasos hechos, así que la ruta no regenera nada. Los casos nuevos salen con Qwen.
- **Producto:**
  - `sim2chip/scene_job.py`: fuera `SIM2CHIP_PANOGEN_PYTHON` de `ROUTE_VARS`.
  - El paquete (`bundle`) apunta el modelo de la panorámica en `scene.json`, para saber de dónde viene cada escena.
- **`experiments/panogen/qwen360_test.py`** se queda como registro de las pruebas. `panogen_test.py` y los scripts de `sphere_walk` que usan WorldGen son experimentos cerrados: no se tocan, aunque dejen de correr sin el entorno.

### Validación

1. **Gratis, local (3090):** caso nuevo `q360i_photo21` desde `pano`, comparado con la Prueba 1.
   - La panorámica debe salir igual que `A_s0.jpg` (misma receta y semilla).
   - La pegada, igual que `A_s0_paste.jpg`.
   - Hoja de comparación.
2. **Tests:** `tests/test_scene_route.py` comprueba que la receta tiene `pano` (`erase`) y `pano_paste` (`scene`) y que el `place` en numpy coincide con el de cv2.
3. **La ruta entera desde una foto (pagando la VM)** no hace falta repetirla ahora: `q360_photo21` ya la pasó con esta panorámica. Se volverá a probar en la VM con 2d.

### Pendiente de decisión del usuario

- **Las 3 escenas públicas** (`f3c_photo21/47/66`, en `runs/web_local`) llevan panorámica de FLUX. Hay tres opciones:
  - cambiar photo21 por `q360_photo21`, que ya está hecha;
  - rehacer photo47 y photo66 con Qwen (~1 $ de VM cada una);
  - dejarlas como están.
- **Borrar del disco**, tras confirmar carpeta por carpeta:

  | Carpeta | Tamaño |
  |---|---|
  | `Crucial/tools/hf_cache/models--black-forest-labs--FLUX.1-Fill-dev` | 9,5 GB |
  | `Crucial/tools/hf_cache/models--mit-han-lab--svdq-int4-flux.1-fill-dev` | 6,4 GB |
  | `Crucial/tools/hf_cache/models--LeoXie--WorldGen` | 43 MB |
  | `Crucial/tools/hf_cache/models--black-forest-labs--FLUX.1-schnell` | vacío |
  | `Crucial/tools/envs/panogen` | 2,1 GB |
  | `Crucial/tools/WorldGen` | 2,5 MB |

  En total, unos 18 GB. El disco de la VM no tiene FLUX: allí solo corre WorldStereo.
- **2d (todo en la VM):** un entorno menos que instalar (WorldGen). En el disco de pesos solo se añade el LoRA 360 (282 MB), porque el nf4 y Lightning ya los necesita el borrador.

### Hecho (2026-10-27)

- **Código según el diseño.** Las 3 escenas públicas se quedan como están (usuario).
- **Validación local** (caso `q360i_photo21`, paso `pano` sin `.env` cargado, como lo lanza la ruta):
  - panorámica y pegada idénticas a la Prueba 1 (diferencia media 0,0), con el mismo ajuste (fx ×1,41, NCC 0,948);
  - tiempos: carga 18 s, panorámica 33 s y costura 30 s.
- **Gotcha:** Qwen con Lightning es determinista, pero un cambio de 1-2 niveles en algunos píxeles de la entrada da otra panorámica (diferencia media 13 sobre 255).
  - Por eso `place()` en numpy reproduce bit a bit el `cv2.remap` (coordenadas a 1/32 de píxel y pesos enteros).
  - Un test exige que coincidan exactamente.
- **Tests:** 304 pasan.
- **Borrado del disco, 18 GB** (aprobado por el usuario):
  - las carpetas de la tabla de arriba;
  - 7 blobs en `hf_cache/blobs/` que solo usaban FLUX y nunchaku;
  - la línea `SIM2CHIP_PANOGEN_PYTHON` de `.env`.

  El entorno `panogen` era un venv apoyado en `eyeforge-qwenedit`, que sigue intacto.

### Orden

1. Código (pano.py, ruta, paths, scene_job, tests), unas 2 h.
2. Validación local 1, unos 10 min de GPU.
3. Documentación: CLAUDE.md no nombra FLUX. Hay que actualizar `plan_integracion_f3c` y `NOTES.md`.
4. Commit, cuando el usuario lo pida.
5. Borrado del disco, carpeta por carpeta.

## Reglas

- **GPU:** antes de lanzar algo, mirar `nvidia-smi` por si otro agente la está usando; avisar a 80 °C y parar a 85 °C.
- **Datos pesados:** van en `data/` (el SSD Crucial).
- **Nebius:** nada sin presupuesto aprobado.
- **Commits:** solo cuando el usuario lo pida, en inglés y sin coautor.
- **Idioma:** respuestas y documentos en español; código en inglés.
