# Alternativas para agujeros y "sin datos" (2026-10-25)

Ruta WorldGen + WorldStereo + gsplat (G2). Objetivo: que las cámaras de Replicator, colocadas en cualquier punto del área, no vean agujeros ni zonas sin datos. El conflicto entre tramos es el tercer fallo.

## Qué medimos

Herramienta: `conflict_map.py` (scratchpad de la sesión, todavía no está en el repo). Cada píxel de una cámara alumno se lleva a su punto 3D del splat y se proyecta en los fotogramas generados por WorldStereo, usando la profundidad de MapAnything calibrada por fotograma (mediana d/z contra el splat; sin calibrar, MapAnything se queda entre un 10 y un 30 % corto y los "sin datos" se inflan del 16 al 55 %).

- **Agujero**: no hay gaussiana. Hasta un 3,5 % de los píxeles bajo el horizonte.
- **Sin datos**: hay gaussiana, pero ningún fotograma la confirma ni la contradice. Un 16 % de media en las vistas probadas (photo21_np, posiciones (9,-4), (8,-4), (0,0) y (-2,0), 8 direcciones cada una). Origen:
  - 64 % oculto: todos los fotogramas que miran ahí ven algo más cerca (reverso o interior de setos, suelo detrás de objetos). Las gaussianas que lo pintan son más grandes y alargadas (agujas y borrones).
  - 26 % fuera de vista: direcciones que nadie recorrió, suelo a los pies.
  - 7 % sin profundidad de MapAnything.
- **Conflicto**: los fotogramas que ven el punto se contradicen en profundidad.

Por distancia al paseo más cercano, lo no visto pasa de 0,07 (menos de 0,5 m) a 0,77 (6 a 8 m). Codo hacia los 2 m.

Hoja de referencia: `data/worldstereo_test/photo21_np/sheet_conflict_G2_spots.jpg`.

## Propuestas

| # | Propuesta | Qué ataca | Antecedentes |
|---|---|---|---|
| 1 | Fixer localizado y progresivo (F3): cámaras alumno en la franja de 2 a 4,5 m, máscara de agujeros y sin datos, varias rondas con poses que se acercan poco a poco, LaMa para los agujeros, reconstrucción posterior | borrones y agujas cerca de lo visto | F0, F1, FX y GenFusion GF1 no ganaron a G2. F1 no seguía la receta de Difix3D+ (una ronda, salto completo, reentreno desde cero, agujeros descartados) |
| 2 | Más recorridos de WorldStereo | fuera de vista | U1 (encadenados) sin ganancia; U2 (todos desde el origen) por medir; S1, S3 y NS (360 en el sitio) peor o igual |
| 3 | La photosphere como fuente de datos en la medida y en el entrenamiento | fuera de vista cerca del centro | G2 ya supervisa con los puntos del panorama; falta contarla en la medida |
| 4 | Mapa de conflicto en toda la rejilla y en photo27 | medida base para comparar todo lo demás | nuevo |
| 5 | Recorridos de WorldStereo dirigidos por el mapa (next-best-view puntuado con agujeros y sin datos) | oculto y fuera de vista | el NBV de fase 2 puntuaba huecos de la guía, no sin datos |
| 6 | Inpainting de vídeo con máscara (Wan VACE, Apache 2.0) sobre recorridos alumno renderizados | oculto, agujeros | nuevo |
| 7 | Segunda photosphere desde otro centro (por ejemplo a 6 m), con WorldGen partiendo de un render del splat | zona lejana | nuevo |
| 8 | Podar las gaussianas que ningún fotograma confirma, o limitar escala y alargamiento | agujas y borrones | G5 (tope de anisotropía 10:1 sobre G0) sin efecto; L1 (poda) planeada el 2026-10-24, no hecha. La poda por evidencia del mapa es nueva |
| 9 | Prior de suelo (plano o profundidad de MoGe) | rayas del suelo lejano | L0b ortofoto del suelo: LaMa gana en la hoja, el usuario rechazó el resultado en Isaac |
| 10 | Supervisión con la profundidad de MapAnything calibrada | conflicto | pendiente desde el 2026-10-22 por la deriva de escala; la calibración por fotograma la resuelve |
| 11 | Filtrar o pesar tramos por consistencia, pérdida robusta, apariencia por tramo | conflicto | A1 (inventar una vez) y R7 (una fuente por objeto) sin cambio visible; best-of-N semillas abierto |
| 12 | Repartir las cámaras de Replicator según la calidad del mapa; no colocar objetivos sobre zonas sin datos | esquiva el problema en el dataset | nuevo |

## Valoración (2026-10-25)

Criterio: la mayor parte de los sin datos (64 %) son sitios que ningún fotograma vio porque algo los tapa. Ahí no falta nitidez, falta contenido. Solo lo arregla algo que genere ese contenido desde un punto de vista nuevo: WorldStereo (2, 5), VACE (6) o una segunda photosphere (7). Fixer (1) mejora lo que ya existe, pero no inventa zonas grandes. Lo fuera de vista (26 %) se cubre con recorridos nuevos (2, 5) y, cerca del centro, con la photosphere (3).

| # | Impacto esperado en sin datos | Coste | Riesgo | Nota |
|---|---|---|---|---|
| 4 | ninguno (medida) | local, ~1 h | bajo | requisito para comparar el resto; incluye contar la photosphere (3) |
| 8 | convierte borrones en agujeros honestos | local, minutos | medio: más agujeros visibles | diagnóstico barato; G5 fue un tope ciego, no una poda por evidencia |
| 5 | alto: ataca el 64 % y el 26 % | VM ~1 $ | medio: los tramos largos se contradicen (C1, C2) | tramos cortos que salen de lo ya caminado y miran hacia las zonas sin datos; guía = render del splat (como S3, que los giros obedecen) |
| 1 | medio, en la franja de 2 a 4,5 m | local, horas | medio: los intentos previos ablandaron | pulido después de 5, no sustituto |
| 6 | alto si funciona | local, montaje nuevo | alto: modelo sin probar, coherencia 3D sin garantía | alternativa si 5 falla |
| 2 | medio | VM ~1 $ | conocido | U2 responde ya a "más recorridos desde el origen" |
| 10 | bajo en sin datos, algo en conflicto | local | bajo | barato, pero no ataca el objetivo |
| 11 | bajo | local | conocido | A1 y R7 sin cambio |
| 9 | bajo | local | conocido | ortofoto rechazada |
| 7 | alto en lejos | VM, montaje nuevo | alto | caro e incierto |
| 12 | no arregla el 3D | local, poco | bajo | red de seguridad del producto, se hace en cualquier caso |

Orden propuesto: 4 (con U2 y photo27) → 8 → 5 → 1 sobre lo que quede. 12 en paralelo. 6 si 5 no basta.

## Resultados

Herramienta en el repo: `experiments/worldstereo/ws_conflict.py` (grid, sheet, gaussians, prune, compare). La photosphere cuenta como un tramo más. Hojas en `data/worldstereo_test/<caso>/conflict/G2/`.

### 4. Medida base (rejilla de 1 m, 8 direcciones, radio 10 m)

| Caso | Vistas | Agujeros | Sin datos | Conflicto (píxeles) | Consistencia entre tramos |
|---|---|---|---|---|---|
| photo21_np G2 | 928 | 1,3 % | 3,0 % | 13,6 % | |
| e2e_photo27 G2 | 2480 | 2,8 % | 12,3 % | 14,7 % | 0,267 |
| u2_photo27 G2 | 2520 | 2,4 % | 7,1 % | 16,1 % | 0,319 |
| u1_photo27 G2 | 2520 | 2,4 % | 7,1 % | 18,4 % | 0,238 |

- El 16 % de sin datos de la hoja de posiciones era de las peores vistas, no de toda el área.
- En photo21 los objetivos se concentran en el extremo lejano (x 7 a 10, lado derecho), donde no llega ningún paseo.
- En photo27 (espacio abierto) los sin datos casi no dependen de la distancia al paseo (6 a 9 % ya a menos de 0,5 m en U2): son sobre todo zonas ocultas (84 %) y máscara de MapAnything (15 %).
- U1 y U2 bajan los sin datos un 43 % frente a e2e: lo que lo consigue son las órbitas view* que comparten. U2 (tramos desde el origen, sin encadenar) tiene menos conflicto que U1 (16,1 % frente a 18,4 %), pero algo más que e2e. Nitidez de U2 entre e2e y U1 (`data/worldstereo_test/u2/measure_e2e_u1_u2.json`).

### 8. Poda por evidencia: descartada

`gaussians` + `prune` en photo21_np: 77 % de las gaussianas respaldadas, 11 % contradichas, 12 % sin respaldo. Quitar las sin respaldo (12 %) o también las contradichas (21,5 %) empeora en todas las vistas: se pierde textura fina del asfalto, aparecen zonas oscuras y, con las contradichas, manchas negras. Las agujas del primer plano siguen. El centro de una gaussiana grande no está en la superficie, así que la prueba por centro no sirve para podar. Hoja: `data/worldstereo_test/photo21_np/conflict/sheet_compare_G2_G2_unbacked_G2_unbacked_contra.jpg`.

### 5. Recorridos dirigidos

`ws_nbv.py audit|plan --targets conflict --radius 10 --max-unknown 0.3`: los objetivos son los píxeles agujero y sin datos de `ws_conflict.py` (los agujeros toman la profundidad de la superficie más cercana para tener sitio en el espacio); la guía solo deja hueco ahí; los tramos salen de fotogramas de tramos con los que se entrenó el splat; tope del 30 % de hueco por fotograma (P2 usaba 60 % y una guía agresiva, e inventaba cosas que contradecían la fase 1).

Prueba U2N (2026-10-25, aprobada por el usuario): 8 tramos de 3,5 m planificados así sobre U2 (`data/worldstereo_test/u2_photo27/nbv_cf/`), guía = render del splat con hueco solo en los objetivos y cielo de la photosphere (hueco por fotograma 5 a 10 %). VM 915 s, ~0,46 $. Los tramos generados son nítidos y siguen la guía, sin inventar objetos (`nbv_cf/sheet_legs_generated.jpg`). El splat de U2 previo se guardó como arma `G2_base`.

| Rejilla (2520 vistas) | Agujeros | Sin datos | Conflicto |
|---|---|---|---|
| U2 (G2_base) | 2,4 % | 7,1 % | 16,1 % |
| U2 + 8 tramos dirigidos (G2) | 2,0 % | 6,3 % | 21,7 % |
| solo radio 8 a 10 m | 3,4 % → 2,4 % | 8,0 % → 6,8 % | 19,8 % → 27,8 % |

Lectura: bajan agujeros y sin datos (sobre todo en el borde), pero el conflicto sube: los tramos nuevos contradicen a los anteriores en el suelo lejano (más fotogramas que opinan, y no coinciden). A la vista la ganancia es pequeña: menos agujas en algunas vistas, algo más de borrón en otras; el borde a 8 m sigue mal en ambos (`conflict/sheet_compare_G2_base_G2.jpg`). Planificador: 8 tramos cubren solo el 0,7 % de los píxeles objetivo (8,9 %); en photo27 los objetivos están repartidos por todo el área.

### 1. F3, Fixer localizado y progresivo (en curso)

`ws_fixer.py pseudo_cf` + `views --base` + armas F3a, F3b, F3c en `ws_splat.py`; cadena en `data/worldstereo_test/u2/f3_chain.sh`. 40 cámaras alumno a 2 a 4,5 m de lo caminado con más objetivos; ronda a 1/3, 2/3 y 3/3 del camino desde el fotograma de entrenamiento más cercano, cada una renderizada del splat de la anterior; Fixer t200 (el τ de Difix3D+); solo enseñan los píxeles objetivo (sin datos o conflicto, dilatados 4 px), con peso 0,3. Hoja de la ronda a: `splat/arms/F3a/sheet_fix.jpg`.

Resultado F3 (tres rondas, ~50 min locales, 0 $):

| Arma | Agujeros | Sin datos | Conflicto | LPIPS held-out | Nitidez novel |
|---|---|---|---|---|---|
| G2_base (U2) | 2,4 % | 7,1 % | 16,1 % | 0,230 | 217 |
| G2 (U2 + 8 tramos) | 2,0 % | 6,3 % | 21,7 % | 0,243 | 217 |
| F3c (G2 + Fixer 3 rondas) | 1,6 % | 6,4 % | 24,0 % | 0,249 | 196 |

(LPIPS de G2_base no es comparable: su held-out no incluye los tramos nbv.) A la vista F3c es el más limpio: desaparecen manchas blancas y agujas del asfalto, el primer plano y la mediana se ven menos emborronados; es algo más suave. Los números de conflicto suben porque las vistas falsas mueven la geometría respecto a la profundidad de MapAnything de los fotogramas (que ya se contradicen entre sí): con vistas falsas en el entrenamiento, el conflicto deja de medir solo calidad. Hoja: `data/worldstereo_test/u2_photo27/conflict/sheet_compare_G2_base_G2_F3c.jpg`.

### Reparación de todo el área en photo21_np (2026-10-25)

Caso bueno de medida (photo27 se descarta como ejemplo). Objetivo: todas las cámaras alumno del área; las 16 vistas de (9,-4) y (8,-4) solo miden. Cadena en `data/worldstereo_test/photo21_np/post_area.sh` y `post_area_c.sh`.

- 4 tramos a la zona de (9,-4)/(8,-4) (`nbv_00..03`) y 5 tramos del planificador de todo el área (`nba_00..04`, solo 5 superaron la ganancia mínima; cubren el 0,73 % de un 1,66 % de píxeles objetivo). Dos VM, ~0,3 $ y ~0,37 $.
- G2 = N2 base + 9 tramos. F3 sobre todo el área: 80 cámaras alumno por ronda (tramo 1 a 99 m de lo caminado), tres rondas, ~50 min locales.
- Las tres armas se miden con la misma evidencia (todos los tramos), así que la rejilla de G2_base se recalcula y cambia respecto a la medida base (más tramos opinan, más conflicto).

| Arma | Agujeros | Sin datos | Conflicto | LPIPS held-out |
|---|---|---|---|---|
| G2_base | 1,35 % | 2,17 % | 16,8 % | (no comparable) |
| G2 (+ 9 tramos) | 0,69 % | 1,17 % | 13,9 % | 0,247 |
| F3c (G2 + Fixer 3 rondas) | 0,23 % | 0,83 % | 16,8 % | 0,290 |

A la vista (`conflict/sheet_compare_G2_base_G2_F3c.jpg`): G2 quita casi todas las agujas y astillas del asfalto y el muro que tenía G2_base; F3c además limpia las manchas blancas de las hojas y deja el suelo liso, a cambio de verse más suave. Mejor arma visual: F3c; mejor equilibrio por números: G2 (conflicto baja y LPIPS no sube). Hojas de 5 columnas: `conflict/<arma>/sheet_spots.jpg`; rejillas: `conflict/<arma>/sheet_grid.jpg`.
