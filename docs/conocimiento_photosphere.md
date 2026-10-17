# Conocimiento previo a la ruta de esfera (2026-10-16)

Lo que hay que saber para implementar `docs/roadmap_photosphere.md` con criterio de experto: ámbitos, estado del arte, herramientas disponibles y qué decisión concreta sale de cada uno. Hecho antes de empezar S.1, a petición del usuario.

## Ámbitos

1. Geometría de panorámicas y metadatos de la captura.
2. Profundidad monocular de 360°.
3. Plano del suelo, vertical y escala.
4. Malla de proxy a partir de profundidad.
5. Segmentación y borrado sobre una esfera.
6. Composición por render diferencial (Debevec) y oclusión.
7. Luz: panorámica LDR, cielo y sol.
8. Retoque de cámara (realismo de sensor).
9. Colocación de objetivos.

## 1. Geometría y metadatos

- Las 4 esferas del 2026-10-16 traen GPano con recorte (cubren de unos 30° arriba a unos 28° abajo; la de 172729 llega a 66° arriba), `PoseHeadingDegrees` (rumbo de brújula del centro de la panorámica), GPS y hora. No traen `PosePitchDegrees` ni `PoseRollDegrees`: el cosido del Pixel nivela el horizonte con el giroscopio y se da por bueno. Se verifica con la normal del suelo (ámbito 3).
- Tres convenciones distintas conviven y hay que pasar entre ellas con una sola función con test:
  - `photosphere.py`: longitud 0 en la columna central, cámara OpenCV (+y abajo).
  - MoGe (`moge/utils/panorama.py`): `theta = (1-u)·2π`, polo en +z.
  - DomeLight de USD (latlong, convención OpenEXR): polo en +Y, longitud 0 hacia +Z. Isaac usa Z arriba, así que el domo va girado. El giro se hace con la transformada del prim y se comprueba con un render de prueba (un punto de color en una dirección conocida).
- Cortes en perspectiva, nunca la equirectangular directa: SAM 3, Qwen y MoGe están entrenados con fotos normales.

## 2. Profundidad de 360°

- **MoGe (MIT)**, ya instalado en `/media/lenena-iker/Crucial/tools/MoGe`, con pesos de MoGe-2 en `tools/moge_ckpt/moge-2-vitl`. Su `infer_panorama.py` usa v1: corta 12 vistas de 90° (vértices de un icosaedro) a 512 px y funde el logaritmo de la distancia resolviendo gradientes y laplaciano por mínimos cuadrados (LSMR), de grueso a fino. Cambiar a v2 es cambiar el import y el checkpoint.
- La fusión por gradientes pierde la escala absoluta. La escala sale después: del suelo y la altura de captura (ámbito 3), con la mediana de las distancias métricas de MoGe-2 por vista como control.
- Lo no capturado (negro del lienzo) debe entrar como máscara inválida en `pred_masks`, o envenena la fusión. Las vistas del icosaedro que caen enteras fuera de la captura se saltan.
- Alternativas revisadas y descartadas por licencia (el producto es comercial y AGPL):
  - Depth Any Panoramas (Insta360, dic. 2025): métrica, pesos publicados, **CC BY-NC 4.0**.
  - UniK3D (CVPR 2025): cualquier cámara, panorámicas incluidas, **CC BY-NC-SA 4.0**.
  - RPG360 (2025): sin entrenar, misma idea que MoGe (cubo + escala por cara + grafo) con normales. Como referencia si la fusión de MoGe falla en costuras.

## 3. Suelo, vertical y escala

- RANSAC sobre los puntos que SAM 3 llama suelo (calzada, acera, aparcamiento, césped). La normal da la vertical; la distancia de la cámara al plano, la altura medida.
- Escala = altura real / altura medida. La real sale del spec (la esfera se hace a la altura de la cámara, P.7), por defecto ~1,5 m.
- Con 28° hacia abajo y 1,5 m de altura, el suelo se ve a partir de ~2,8 m: suficiente para el ajuste. El suelo cercano se cubre prolongando el plano.
- Si la normal se aparta de la vertical más de 2-3°, el cosido no estaba nivelado: se corrige girando la esfera, no la cámara.

## 4. Malla de proxy

- Malla de la equirectangular de profundidad, sin cielo, con el suelo sustituido por el plano ajustado (más limpio para recibir sombras).
- Cortar las caras que cruzan saltos de profundidad (cociente de distancias entre vecinos o arista larga respecto a la distancia). Si no, salen "sábanas" entre un coche y la fachada que tapan objetivos y proyectan sombras falsas. Es el fallo clásico de mallar profundidad.
- El proxy no lleva textura propia: solo tapa, recibe sombras y sostiene (P.4). Su albedo se ajusta como hoy el del soporte en `bg_isaac`.

## 5. Segmentación y borrado

- SAM 3 sobre los cortes, máscaras de vuelta a la panorámica con peso por distancia al centro del corte.
- Borrado: un borrador por corte da rellenos distintos en los solapes y fundirlos deja fantasmas. Mejor elegir para cada objeto el corte donde queda más centrado, borrar ahí y reproyectar solo esa zona. La costura izquierda-derecha de la equirectangular es el sitio típico de fallo (SE360, dic. 2025), y en cortes en perspectiva no existe.
- Borrar también la sombra del objeto (SE360 lo hace en la misma máscara). Si no, el coche nuevo aparece junto a la sombra de uno que ya no está.
- Los fantasmas del cosido del móvil (medio coche en movimiento) los etiqueta el profesor en cada vista, nunca una regla aparte.

## 6. Render diferencial y oclusión

- Debevec (1998) divide en escena lejana (no cambia), escena local (con reflectancia estimada, recibe sombras) y objetos sintéticos. Solo se renderiza el **cambio** en la escena local, así que basta una reflectancia aproximada. `bg_isaac.py` ya lo hace: fondo × (A/B) y objetos con la exposición que lleva B a la foto. Se generaliza tal cual a una cámara que gira.
- La oclusión sale sola si el proxy está en la pasada A: la máscara de objeto se toma de la segmentación por instancia (lo visible del objeto), y donde el proxy queda delante se conserva el fondo real.
- El "Matte Object" del RTX de Omniverse existe, pero sus notas de versión registran fallos (bordes al componer, mates borrosos en tiempo real). Nos quedamos con A/B, que no depende del renderizador y ya está medido.
- DiPIR (NVIDIA, ECCV 2024) ajusta luz y tono con un modelo de difusión como juez y render diferenciable. Es la versión cara de nuestro ajuste de réplica; queda como idea, no para este plazo.

## 7. Luz

- Alhaija (IJCV 2018): la panorámica correcta como luz apenas importa (49,7 frente a 49,2 con otra y 49,1 sin ninguna). No se reconstruye HDR.
- La panorámica es LDR sRGB y el sol satura. Se pasa a lineal, el sol se quita del domo y va como luz direccional aparte (como hoy).
- La parte de arriba no capturada del domo se rellena con el color medio de la fila de cielo más alta.
- **Dirección del sol exacta sin estimarla:** GPS + hora (EXIF) dan la posición del sol, y `PoseHeadingDegrees` la lleva a la panorámica. Sirve de valor inicial o sustituye al ajuste por sombras cuando hay sol directo.
- Las esferas del 2026-10-16 se hicieron entre las 19:20 y las 19:31 (CEST), con el sol en el horizonte o ya puesto: luz difusa, sombras suaves. El ajuste debe dar una fracción de sol cercana a cero. Para probar sombras duras hace falta una esfera a pleno sol.

## 8. Retoque de cámara

- Alhaija: desplazamiento de color (aberración cromática), desenfoque por profundidad y curvas de color y gamma para igualar la estadística de las fotos reales. Sin retoque, 49,7 → 43,8 AP50; es de lo que más pesa.
- Carlson et al. (ECCVW 2018): aberración cromática, desenfoque, exposición, ruido y temperatura de color, aleatorios y con base física, reducen el salto sintético-real en detección.
- Aplicarlo a la **imagen compuesta entera**, no solo al objeto: el fondo viene de un móvil (limpio y cosido) y el destino es otro sensor. El objeto se renderiza a más resolución y se reduce con el mismo filtro que el corte del fondo, para que tengan la misma nitidez.
- Perfil OV2640 (ESP32-CAM, ~66° de FOV): bloques de JPEG en zonas lisas incluso a calidad alta, ruido y desenfoque en poca luz, parpadeo de 50 Hz con luz artificial. El perfil de hardware decide los rangos.

## 9. Colocación

- Alhaija: sobre el plano del suelo con segmentación de calzada y rumbo al azar rinde como a mano; al azar en 3D, peor; más de 5 coches por imagen, peor.
- 3D Copy-Paste (NeurIPS 2023): posición, tamaño, orientación y luz plausibles suman (42,13 → 43,34 mAP en SUN RGB-D); evita choques con la habitación por búsqueda restringida.
- GeoSim (CVPR 2021): coloca según carriles y sin choques, renderiza y compone con oclusión por geometría.
- Para S.6: plano del suelo + tipo de suelo de SAM 3 + rumbo según los objetivos borrados o la calle; descartar posiciones cuya caja choque con el proxy por encima del suelo; de 1 a 5 objetivos; visibles en la vista.

## 10. Antes de S.5 y S.6 (2026-10-17)

Repaso hecho tras S.4, antes de la réplica.

- **Perfil que importa: el móvil.** No hay fotos reales de ESP32-CAM, y la mayoría de usuarios empezarán con el móvil (usuario, 2026-10-17). El fondo ya sale del móvil, así que el retoque no debe ensuciar el fondo: el objeto renderizado se lleva a la nitidez y al ruido medidos en el propio corte de la panorámica, y luego se mide la diferencia entre la panorámica y las fotos sueltas del usuario (que sí verá el alumno en test) para aplicarla a toda la imagen.
  - Ruido: modelo Poisson-Gauss (varianza lineal en la intensidad; Foi et al., IEEE TIP 2008), estimado en bloques lisos de una sola imagen. Para JPEG de móvil basta una varianza por nivel de brillo, medida en bloques de poca textura.
  - Nitidez: anchura de los bordes (perfil de un borde fuerte) en el corte de fondo y en el render, misma medida en los dos.
  - El perfil ESP32 se queda como primera estimación hasta que haya fotos reales.
- **Rumbo de los objetos borrados (réplica).** Una silueta no distingue el frente de la trasera (lo dice `bg_build.fit_pose`).
  - El eje largo sale de la geometría: los puntos del coche en la profundidad original (`depth.npz`, antes del borrado), vistos desde arriba, con la orientación por mínimos rectángulos. Se afina por IoU de silueta como en `fit_pose`.
  - Medido en S.5 (2026-10-17): la caja sola pone en fila los coches aparcados en batería, y el modelo acierta su rumbo; con renders de rumbo conocido su acimut queda a una mediana de 5° (convención: 0 es frente a la cámara, crece hacia la derecha del frente). Se usa para el rumbo y el sentido, y la caja afina ±15°.
  - El sentido (frente o trasera) lo puede dar **Orient Anything V2** (NeurIPS 2025, sobre VGGT): acimut, elevación y giro desde una foto, y simetrías. Código CC-BY-4.0 y pesos en Hugging Face (`Viglong/OriAnyV2_ckpt`, 5 GB). La licencia de los pesos no está escrita aparte: confirmarla antes de usarlo en producto. Para la réplica basta como control; si no se puede usar, el sentido se deja al ajuste fotométrico (frente y trasera tienen colores distintos: faros, lunas).
- **Igualar color.** Alhaija igualó curvas y gamma globales. La armonización de composiciones (Wang et al., CVPR 2023) dice que lo global no basta: hace falta un mapa de sombreado local. En nuestro caso ese mapa lo da el render diferencial (A/B). Queda ajustar por réplica: exposición y dominante globales (mundo gris de `flow_photometry`), una ganancia de brillo por objeto (como `bg_isaac.fit_colours`, sin tocar el tono) y la fuerza del cielo frente al sol.
- **Sombras con cielo cubierto o sol puesto.** Con la luz de cielo, la sombra de contacto es oclusión ambiental: el path tracing con domo ya la calcula si el proxy está bien. Lo que la estropea es un proxy malo (las sombras salen más cortas de lo que son; No Shadow Left, CVPR 2021). DiPIR (ECCV 2024) usa un simple plano de suelo como receptor de sombras, que es lo que da nuestro suelo plano hasta 25 m. Control en la réplica: oscuridad bajo el coche real (panorámica original / borrada) frente a A/B.
- **Luz de las esferas del 16.** El sol está bajo el horizonte en las cuatro, así que el ajuste se reduce a la intensidad del domo y la exposición. El barrido de dirección del sol de `bg_isaac` solo hará falta con esferas a pleno sol.
- **S.6.** Etiquetas desde Replicator (instancias y cajas 2D ajustadas a lo visible), revisión del VLM (`bg_review` con Cosmos Reason 2) y filtro de parecido (`likeness.py`, SigLIP 2) contra las fotos del usuario. En terreno inclinado (park) el objeto se inclina con la normal local del suelo.

## Fuentes

- [Alhaija et al., IJCV 2018](https://ar5iv.labs.arxiv.org/html/1708.01566)
- [Debevec, SIGGRAPH 1998](https://www.pauldebevec.com/Research/IBL/debevec-siggraph98.pdf)
- [Carlson et al., Modeling Camera Effects, ECCVW 2018](https://arxiv.org/abs/1803.07721)
- [3D Copy-Paste, NeurIPS 2023](https://github.com/gyhandy/3D-Copy-Paste)
- [GeoSim, CVPR 2021](https://arxiv.org/abs/2101.06543)
- [DiPIR, ECCV 2024](https://research.nvidia.com/labs/toronto-ai/DiPIR/)
- [Depth Any Panoramas](https://github.com/Insta360-Research-Team/DAP), [UniK3D](https://arxiv.org/abs/2503.16591), [RPG360](https://arxiv.org/abs/2509.23991)
- [SE360, edición semántica en 360°](https://arxiv.org/pdf/2512.19943)
- [Notas de versión del renderizador RTX](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_release_notes.html), [DomeLight de USD](https://openusd.org/dev/user_guides/schemas/usdLux/DomeLight_1.html)
- [Calidad de imagen de la OV2640 (foro de Espressif)](https://esp32.com/viewtopic.php?p=65848)
- [Orient Anything V2, NeurIPS 2025](https://arxiv.org/abs/2601.05573), [código](https://github.com/SpatialVision/Orient-Anything-V2)
- [Wang et al., Semi-Supervised Parametric Real-World Image Harmonization, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Semi-Supervised_Parametric_Real-World_Image_Harmonization_CVPR_2023_paper.pdf)
- Foi et al., Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data, IEEE TIP 2008
- [No Shadow Left Behind, CVPR 2021 (suplemento)](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Zhang_No_Shadow_Left_CVPR_2021_supplemental.pdf)
