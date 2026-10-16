# Cómo generar escenas para datos sintéticos en Isaac (investigación, 2026-10-16)

Pregunta del usuario: ¿cuál es la mejor manera de generar una escena para datos sintéticos en Isaac? Y, cuestionando el sistema, ¿conviene separar la entrada del sintético (varias fotos por escena, tantas escenas como quiera el usuario) de las fotos de entrenamiento?

Investigación previa en `docs/investigacion_photo2scene_2026-10-13.md` (estado del arte de una foto a escena) y `docs/escena_agente_literatura.md`. Aquí se compara con lo que hace la industria y con nuestras propias medidas.

## Respuesta corta

Para nuestro caso (cámara pequeña y fija, pocas fotos reales, contar o detectar objetos), lo que mejor funciona según la literatura y la industria es:

1. **Fondo real del sitio** donde estará la cámara, no un mundo inventado: la foto del usuario borrada, con geometría aproximada para oclusiones y sombras. Es nuestra imitación (fase 2) y la ruta de soporte.
2. **El esfuerzo en el objetivo, no en el fondo**: muchas instancias distintas de lo que se detecta, en muchas poses, posiciones plausibles y oclusiones, con la luz de la escena. La variedad pesa más que el fotorrealismo.
3. **Reconstrucción 3D real (varias fotos o vídeo) solo cuando la cámara tiene que moverse** (robots, drones). La vía oficial de NVIDIA es 3DGUT + NuRec en Isaac Sim 5.x.
4. **Montar la escena entera desde una sola foto** (lo que hacemos en la calle con SAM 3D, plantillas y casas) es frontera de investigación, no práctica de producción.

## Lo que dicen nuestras propias medidas

| Medida | Resultado | Fuente |
|---|---|---|
| Botellas, 24 fotos, 3 ejecuciones (A.4) | solo fotos 0,779 ± 0,011; con escena 0,764 ± 0,031 (ruido) | `roadmap_photo2scene.md` |
| Botellas, 8 fotos, 3 ejecuciones (A.4) | solo fotos 0,697 ± 0,013; con escena 0,691 ± 0,023 (ruido) | idem |
| Botellas pocas fotos, dosis (D.1) | 0,661 solo fotos; 10 vistas 0,703; 20 vistas 0,695 | memoria D.1 |
| HOPE, sintético revisado | 0,830 frente a 0,819 solo real; sin revisar 0,814 (daña) | `experiments/NOTES.md` |
| Calle, contar coches (V.6) | 15 vistas de calle 0,637 frente a ~0,65 solo fotos | noche 2026-10-14 |

Conclusión: ninguna escena nuestra ha movido el F1 del alumno más allá del ruido (±0,03), salvo pistas pequeñas con muy pocas fotos. Ya el 2026-10-14 se anotó: "lo que falta son coches, no calle".

Dato clave del código: `flow_populate.py` coloca solo las cosas borradas **de la misma foto**, como mallas SAM 3D hechas desde una vista. Para contar coches, el alumno ve una y otra vez los mismos 1 o 2 coches de esa foto. Según la literatura de abajo, esto es lo que más limita, más que la calidad de las casas. **Es una hipótesis**: se mide antes de decidir (plan, punto 1).

## Lo que hacen los profesionales

| Enfoque | Cómo funciona | Quién | Evidencia |
|---|---|---|---|
| **Randomización estructurada (SDR)** | Assets de catálogo colocados al azar con sentido (coches en la calzada) sobre fondos variados | NVIDIA Replicator, Edge Impulse (detectores para microcontroladores con Omniverse) | En KITTI supera al sintético fotorrealista y a datos reales de otro dominio |
| **Cortar y pegar** | Recortes de objetos sobre fondos reales | Investigación e industria (barato) | Competitivo con datos etiquetados a mano; falla la luz entre objeto y fondo |
| **Objetos 3D en escena real 2,5D o 3D** | Fondo real (foto con profundidad o splat) y objetos 3D insertados con luz estimada | Conducción (nuScenes con 3DGS), drones (UAVTwin) | UAVTwin: +2,5 a +13,7 % mAP; en nuScenes, la diversidad geométrica de la colocación pesa más que la de apariencia |
| **Gemelo digital multivista** (3DGUT, NuRec) | Vídeo o fotos, luego COLMAP, splat en Isaac, objetos encima | NVIDIA (tutorial con smartphone), robótica | Autodesk: los splats dan algo menos de mAP que las mallas de fotogrametría, pero necesitan menos fotos |
| **Mundo generado desde una imagen** | Difusión de vídeo a splat (Lyra 2.0, Marble) | NVIDIA Research, World Labs | Sin medidas de detección; Marble es de pago y Lyra no tiene pesos comerciales |
| **Capa generativa encima** (Cosmos Transfer 2.5) | Profundidad, segmentación y bordes de Isaac convertidos en vídeo fotorrealista | NVIDIA | Mejora políticas de robots; sin cifras de detección. 264-445 s por clip de 93 fotogramas en H100 |

### Qué importa para entrenar un detector (literatura)

- **Variedad antes que realismo.** Un survey de octubre de 2026 lo resume así: aumentar la variación "suele ser más eficaz que aumentar el realismo visual".
- **Puntos de vista y colocación.** En detección con pocos datos reales (YOLO11 con el 2,5 % de las imágenes reales), colocar objetos 3D al azar sobre HDRIs da 62,8 mAP@50. Colocarlos "físicamente" sobre planos de fondos RGBD reales, con luz estimada (lo más parecido a nuestra imitación), da 58,2, y cortar y pegar 55,6. Los autores lo atribuyen a que las restricciones físicas reducen la diversidad de puntos de vista.
- **Distractores y contexto.** Sin objetos de fondo en el entrenamiento aumentan los falsos positivos.
- **Un poco de real ayuda mucho**, y el sintético sin control de calidad daña (también lo vimos en HOPE).
- **Ojo:** en nuestras medidas la plantilla genérica (azar sobre HDRI) dañó (botellas, 8 fotos: 0,570 frente a 0,672). La diferencia está en los assets: el artículo usa capturas multivista de los objetos reales, y la plantilla usaba modelos de internet. Otra vez apunta a la calidad y variedad del objetivo.

## NuRec y 3DGUT en Isaac: qué hay y qué falla

- Isaac Sim 5.0 y posteriores cargan escenas NuRec (USDZ); nosotros tenemos 5.1. En 6.x pasan a `ParticleField` y el USDZ propio queda obsoleto.
- 3DGRUT: Apache 2.0, CUDA 11.8 a 13.0 (12.8 por defecto). Unos 11 min por escena de MipNeRF360 en una RTX 5090 (3DGUT); en L40 con MCMC y 1M primitivas, unos 76 min. Entrada estilo COLMAP; también soporta rolling shutter.
- Captura: unas 100 fotos con un 60 % de solape, obturador de 1/120 a 1/250 s en exterior, balance de blancos fijo.
- **Límites que nos afectan:**
  - El splat no tiene física ni sombras propias: hace falta un proxy de malla (suelo) marcado como "matte" para recibir las sombras de los objetos insertados, y debe coincidir con la geometría del splat o salen artefactos blancos.
  - NuRec no aparece en las pasadas de datos sintéticos (profundidad, segmentación): las da el proxy. Nosotros etiquetamos con nuestras mallas o con el profesor, así que no bloquea, pero las oclusiones de un objeto tras un árbol del splat no salen en las etiquetas sin proxy.
  - La luz está horneada: no se puede pasar de día a noche ni mover el sol, solo exposición y tinte. Para variar la luz haría falta Cosmos Transfer.
  - El splat solo se ve bien cerca del recorrido grabado: tampoco permite extrapolar lejos.
  - Si se abre un `.usdz` como escena raíz, no se le pueden añadir assets: hay que referenciarlo desde un `.usda`.
  - Con el proxy visible, el RGB se estropea; sin él, fallan la profundidad y la segmentación (experiencia de Ekumen).

## La propuesta del usuario: entrada separada para el sintético

Hoy una sola foto tiene que dar fondo, puntos de vista y objetos. La propuesta separa:
- **Fotos de entrenamiento**: igual que ahora (las de la cámara real, aunque sean malas).
- **Escenas para sintético** (opcional): cada escena son varias fotos del mismo sitio, y el usuario añade tantas como quiera.

Valoración:
- **A favor:**
  - Los puntos de vista nuevos salen de fotos reales y no hay que inventarlos (adiós a plantillas, casas, solares, faldas, enderezado).
  - Más fondos distintos, que es lo que la literatura dice que ayuda.
  - El fondo no tiene que salir de una foto de la cámara de 2 MP.
  - Abre la puerta a 3DGUT/NuRec, que encaja con la historia NVIDIA.
  - Botellas ya funciona así casi sin cambios: cada foto de la escena pasa por la ruta de soporte.
- **En contra:**
  - Más esfuerzo para el usuario.
  - Reconstrucción 3D frágil (COLMAP falla con poca textura o poco solape).
  - Luz horneada.
  - Integrarlo entero (3DGUT en el producto) no cabe con garantías en 14 días.
- **Versión mínima que sí cabe:** una escena son 3 a 20 fotos del sitio y cada foto se imita por separado (la fase 2 que ya funciona). No hace falta fusionarlas en 3D para ganar variedad de fondo y de punto de vista. La fusión con 3DGUT queda como paso siguiente.

## Opciones para los 14 días

| Opción | Qué es | Coste | Riesgo | Evidencia de que ayude |
|---|---|---|---|---|
| A. Objetivo variado | Poblar con las instancias de todas las fotos (y, si hace falta, un catálogo filtrado de modelos o capturas multivista del objeto), con colocación estructurada y muchas poses | 2-3 días | Bajo | Literatura fuerte; nuestra nota de V.6 |
| B. Escenas como entrada aparte, mínima | Varias fotos por escena, cada una imitada; la fase 3 apagada por defecto | 2-3 días (web, biblioteca, ruta) | Bajo-medio | Variedad de fondos; botellas ya va |
| C. 3DGUT/NuRec | Vídeo de una calle, splat en Isaac, coches insertados, proxy de suelo | 4-6 días para una escena demo | Medio-alto | UAVTwin, nuScenes 3DGS; NVIDIA |
| D. Casas cousin | Fase 3 de una foto con casas paramétricas | 2 días más | Medio | Mejor que fix5 a ojo; sin medir en el alumno |
| E. Cosmos Transfer | Variar luz y tiempo sobre los renders | 1-2 días de prueba | Medio (coste) | Sin cifras de detección |
| F. Seguir parcheando la extrapolación | | Abierto | Alto | Nada medido a favor |

## Recomendación

1. **Primero, una medida que decide (1 día).** Botellas con 8 fotos y calle, 3 ejecuciones por brazo, comparando el poblado actual (las cosas de la misma foto) con el poblado con instancias de todas las fotos de entrenamiento. Si sube, el cuello de botella era el objetivo y no el fondo.
2. **A + B** como producto para la entrega. El esfuerzo pasa del fondo al objetivo. La fase 3 por plantilla se apaga por defecto, y la variedad de puntos de vista sale de las escenas que aporte el usuario. Así se borra código heurístico en lugar de añadirlo, y eso responde a la pregunta de calidad del código.
3. **C solo si 1 y 2 están estables hacia el 22-23 de octubre**, como escena demo para el vídeo (historia NVIDIA: Isaac, NuRec, 3DGUT) y medida con el alumno. No en el producto antes de la entrega.
4. **D y E** quedan aparcadas. El cousin queda como prueba hecha (`p2s_test/compare/cousin_2026-10-16/`).

Decisiones que necesita el usuario:
- Si se cambia el reparto 90/10 decidido el 2026-10-11, porque la fase 3 pasa a salir de escenas reales y no de plantillas.
- La entrada separada de escenas para sintético: sí o no, y en qué versión.
- Si se hace la prueba de NuRec, y cuándo.

## Resultado de la medida de variedad del objetivo (2026-10-16)

bottles_few: 8 fotos de entrenamiento y 20 de test, 10 vistas por trabajo, F1 int8 con 5 semillas por ejecución. En los brazos con escena, la composición A es la semilla 0 (entrenada dos veces) y la composición B es la semilla 1. Script en `p2s_test/scripts/arms_pool.sh` y `arms_pool4.sh`; resultados en `arms.log`, en las secciones "#### POOL"; hoja en `p2s_test/compare/pool_2026-10-16.jpg`.

| Brazo | Composición A | Composición B | Media |
|---|---|---|---|
| Solo fotos | 0,675 / 0,694 / 0,701 | | 0,690 |
| 1 escena, sus objetos (producto) | 0,737 / 0,737 | 0,735 | 0,736 |
| 1 escena + objetos de las 8 fotos | 0,688 / 0,670 | rechazada | 0,679 |
| Hasta 8 escenas, sus objetos | 0,653 / 0,612 | 0,754 | 0,673 |
| Hasta 8 escenas + objetos de las 8 fotos | 0,613 / 0,648 | 0,627 | 0,629 |

- La escena del producto mejora por primera vez por encima del ruido: +0,046, estable con dos composiciones distintas. Es la foto 41, la escena mejor construida.
- Mezclar objetos de otras fotos no ayuda: queda igual o peor. En la hoja, las mallas mezcladas salen oscuras y rotas, y la revisión del VLM rechaza escenas enteras por "formas rotas o huecas" (una composición de "1 escena + mezcla" se quedó sin vistas).
- Varias escenas: mucho ruido según la composición (de 0,61 a 0,75), porque varias fotos dan escenas peores que la 41.
- Conclusión: con las mallas de SAM 3D de una sola vista, el cuello de botella es la calidad de cada objeto, no la variedad. La hipótesis de la sección anterior no se confirma en esta forma. `SIM2CHIP_SCENE_POOL` se queda apagado.
- Bug encontrado: el entrenamiento se quedaba sin memoria de GPU tras componer porque el VLM local de la revisión seguía cargado. `train.run` lo descarga ahora al empezar.

## Fuentes

- [Structured Domain Randomization (NVIDIA)](https://arxiv.org/pdf/1810.10093)
- [Cut, Paste and Learn](https://arxiv.org/html/1708.01642)
- [An Annotation Saved is an Annotation Earned](https://arxiv.org/pdf/1902.09967)
- [Object-Centric Data Synthesis for Category-level Object Detection](https://arxiv.org/html/2511.23450v1)
- [Domain generalization and synthetic data in object detection (survey, 2026)](https://arxiv.org/html/2610.00030)
- [Ablaciones de domain randomization en detección](https://link.springer.com/article/10.1007/s44196-025-00817-4)
- [Gaussian splatting como generador de datos para detección 3D (nuScenes)](https://arxiv.org/abs/2504.16740)
- [UAVTwin](https://arxiv.org/abs/2504.02158)
- [Autodesk: gemelos digitales para entrenar detectores](https://www.research.autodesk.com/blog/how-digital-twins-accelerate-ai-model-deployment/)
- [NVIDIA: reconstruir una escena en Isaac Sim con un smartphone](https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/)
- [Isaac Sim 5.1: Neural Volume Rendering](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_nurec.html)
- [Isaac Sim 6.2: Neural Volume Rendering](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_nurec.html)
- [Omniverse NuRec rendering (límites del proxy)](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/neural-rendering.html)
- [Ekumen: NuRec en Isaac Sim](https://ekumenlabs.com/blog/posts/nurec-bridging-visual-gap-in-sim/)
- [3DGRUT](https://github.com/nv-tlabs/3dgrut)
- [Isaac Sim y World Labs Marble](https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble/)
- [Lyra (NVIDIA)](https://github.com/nv-tlabs/lyra)
- [Cosmos Transfer 2.5, tiempos de inferencia](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md)
- [Edge Impulse con Omniverse Replicator](https://docs.edgeimpulse.com/projects/expert-network/nvidia-omniverse-replicator)
- [AnySplat (sin COLMAP)](https://arxiv.org/html/2505.23716v1)
- [ComGS: luz horneada al insertar objetos en splats](https://mlanthology.org/iclr/2026/gao2026iclr-comgs/)
