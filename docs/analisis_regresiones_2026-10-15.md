# Regresiones de photo2scene del 2026-10-15 y qué cambiar

Contexto: reconstrucción de las fases 2 y 3 sobre la extracción congelada de la línea base (`p2s_test/now_2026-10-15` frente a `p2s_test/fix3_2026-10-15`, hojas en `p2s_test/compare/antes_despues_2026-10-15/`). Veredicto del usuario y mío: mejoran home_office, corner_timber y en parte terrace; empeoran mucho basque_street, photo68, church_corner y straight_road. Nada está en commit.

## Lo que hace Mira-Scene (código en `Crucial/tools/mira_src/infer_scripts`)

1. **Un solo marco de gravedad: la normal del suelo.** `4_estimate_floor.py` ajusta el plano del suelo (RANSAC más dos SVD) y la gravedad es su normal; todo se expresa en ese marco, con el suelo en y = 0.
2. **Objetos verticales con la rotación restringida, no corregida después.** `solve_similarity_transforms_gravity`: el eje vertical del objeto va exacto a la gravedad y solo se ajustan el giro sobre la vertical (yaw), la escala y la traslación, en forma cerrada contra los puntos de profundidad de su máscara. El objeto queda vertical y sobre sus píxeles a la vez.
3. **Apoyo rígido.** `GeometryPlacementBackend`: un objeto sobre el suelo solo se traslada en vertical hasta que su punto más bajo toca y = 0; sobre otro objeto, busca la cara de apoyo debajo y se mueve con un tope de desplazamiento horizontal. Nunca deforma mallas. Las colisiones solo se diagnostican.

## Causas de las regresiones (medidas)

### 1. Cosmos en la elección de especies (paso 2.1c, `flow_section.py`)

La sección se recalcula en la fase 2 y ahora la elige Cosmos (hereda `MODEL` de `flow_vegetation`). Resultado por franja (antes, Qwen, frente a después):

| Escena | Antes | Después |
|---|---|---|
| photo68 | árboles Sugar_Maple y Chinese_Juniper, seto Holly | sin elección en las 3 franjas: 0 plantas (antes 21) |
| straight_road | Hawthorn, Holly, Yew, Thevetia | sin elección en las 4 franjas |
| terrace | American_Beech, Red_Maple | sin elección |
| corner_timber | Scarlet_Oak | árbol sin elección |
| basque_street | Golden_Chain, Holly, Thevetia | Pampas_Grass, Hibiscus, Oleander (flores rojas que no están en la foto) |

Es la causa de photo68 (muros de piedra sin verde), de las flores de basque y de parte de straight_road y terrace. Probable mecanismo del vacío: `map_within(..., seconds=600)` con 3 votos por franja en un servidor local que tarda de 20 a 60 s por respuesta, o respuestas que no pasan el formato; en ambos casos falla en silencio.

### 2. Girar `up` con las verticales de la foto (`lean.photo_vertical` en `piece_build.py`)

Ángulo entre `up` y la normal del suelo, antes frente a después: church_corner 7,1 a 8,6°, terrace 2,5 a 8,3°, straight_road 0,4 a 3,6°, straight_whitewash 1,6 a 4,0°, basque 4,0 a 5,6°. photo68 y corner_timber no cambian (no se giró) y son justo las que no empeoran por geometría. Una calle con 8° de pendiente es rara; lo más probable es que el giro esté mal estimado. Con `up` separado del suelo:

- `standing.straighten` gira cada edificio sobre su centro, lo que le quita sus píxeles y le inclina la base (church_corner: base de 1,45 m de alto en un extremo);
- `crossroad.lifts` y `standing.flatten` nivelan punto a punto contra el plano inclinado, lo que cizalla;
- el parche de hoy, `standing.seat`, estira paredes, pero no arregló las copias.

Ya había una parte de esto antes de hoy (`up` sale de las paredes y no del suelo, commit 1029658), pero el giro lo agrava.

### 3. Menores

- Comprobación de vistas con Cosmos: más estricta que Qwen, así que los recuentos no se comparan entre columnas.
- `HARD_SUN_STOPS`, los umbrales de luz del día y `MIN_COVER` se fijaron mirando estas mismas fotos.

## Fuentes externas útiles

- Gravedad desde una foto: GeoCalib (ECCV 2024, código Apache-2.0, pesos CC-BY-4.0) estima intrínsecos y gravedad con incertidumbre; [repo](https://github.com/cvg/geocalib), [artículo](https://arxiv.org/pdf/2409.06704). El punto de fuga vertical equivale a la gravedad, pero depende de detectar líneas y se confunde con líneas del suelo; [Pautrat et al. ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Pautrat_Vanishing_Point_Estimation_in_Uncalibrated_Images_with_Prior_Gravity_Direction_ICCV_2023_paper.pdf) usan una gravedad previa para guiar los puntos de fuga, no al revés.
- Mundo Manhattan y Atlanta (una vertical más horizontales no ortogonales, el caso de una calle): [Coughlan y Yuille 2000](https://proceedings.nips.cc/paper_files/paper/2000/file/90e1357833654983612fb05e3ec9148c-Paper.pdf), [Joo et al. CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Joo_Globally_Optimal_Inlier_CVPR_2018_paper.pdf).
- Pose vertical con solo yaw: [GOPI](https://arxiv.org/pdf/2608.06836) (traslación más yaw periódico con gravedad conocida), [Total3DUnderstanding](https://arxiv.org/pdf/2002.12212) (marco alineado con la gravedad).
- Colocación por restricciones: [Infinigen Indoors](https://arxiv.org/pdf/2406.11824) (recocido simulado sobre un grafo de relaciones; un mueble apoyado en la pared solo se desliza por la línea pared-suelo; por etapas, de grande a pequeño).
- Calles procedurales: [Parish y Müller 2001](https://history.siggraph.org/?p=116923) (red de calles, parcelas, edificio por parcela) y gramáticas de forma CGA para fachadas.
- Física como control: SimuScene (ya en `docs/investigacion_photo2scene_2026-10-13.md`), soltar y medir el desplazamiento.

## Cambios propuestos, de más a menos seguro

1. **Volver a Qwen en la elección de especies** (`flow_section`, `flow_vegetation`) hasta medir otra cosa. Y que no falle en silencio: si una franja se queda sin especie, usar una por defecto de su tipo (seto: Privet; árbol: la más común del catálogo) y avisar en el log.
2. **Quitar el giro de `up` por verticales y `standing.seat`.** Quedarse con `lean` solo como medida informativa.
3. **Adoptar el marco de Mira para colocar:** gravedad igual a la normal del suelo cuando el suelo se ve bien (la calle es casi plana en todas las fotos de prueba). Si el suelo no se ve, las paredes. GeoCalib, solo como comprobación que avise cuando discrepen más de 2 o 3°, y medido antes de usarlo para decidir.
4. **Enderezar como Mira:** en vez de girar sobre el centro y luego subir o bajar, ajustar con la vertical fija solo yaw y traslación contra los puntos de profundidad de la máscara del edificio, y apoyar con una traslación rígida. Sin deformar y sin cizallar en las plantillas (`crossroad.lifts` y `street_template` con una traslación por objeto, como ya se hace con los assets de NVIDIA).
5. **Extrapolación con cámaras cerca de la de la foto** (decidido con el usuario): pocos metros a los lados, otra altura y otro giro, para que casi todo lo visto sea la escena reconstruida.
6. **Detectores de regresión baratos en cada ejecución**, escritos en un JSON y comparados con la línea base: ángulo entre `up` y el suelo, hueco de la base por edificio, franjas sin especie, plantas colocadas frente a la cobertura de plantas de la foto, ΔE desde la cámara de la foto. Habrían avisado de photo68 y de church_corner sin mirar hojas.
7. **Comprobación física por gravedad** (SimuScene) en lugar del "floating" del VLM: determinista y gratis.

Validación: desde cero sobre las 15 fotos y con dos repeticiones, antes de dar nada por bueno. Si no mejora, se descartan los cambios (el usuario lo pidió así).
