# Match: acercar el sintético a las fotos del usuario

Idea del usuario (2026-09-29): que un modelo con razonamiento mire las fotos del usuario y los renders y ajuste la escena en unas pocas iteraciones hasta que se parezcan. Ningún Nemotron de Token Factory ve imágenes, así que se reparte el trabajo:

1. **Look de cámara** (`sim2chip/look.py`, sin modelo): color en Lab (media y dispersión), nitidez de bordes y ruido de las fotos a la resolución del modelo. Se guarda en `spec.look` y `points`/`crops` lo aplican a cada render (transferencia de color Reinhard con estadísticas de conjunto, blur y ruido si hacen falta, intensidad aleatoria 0,6-1 por imagen para no perder variedad).
2. **Escena** (`sim2chip/match.py`), en rondas:
   - render de prueba de 48 frames en `<job>/match/it<k>/` (copia las etapas de assets del padre; solo se rehace lo que el cambio alcanza, así que no vuelve a pasar SAM 3 ni TRELLIS);
   - Kimi-K3 (visión) ve dos mosaicos, fotos arriba y renders abajo, y lista diferencias concretas;
   - Nemotron 3 Ultra convierte las diferencias en cambios del spec, solo a través de una lista blanca de palancas con rangos (`LEVERS`: reparto de lugares, distancia y altura de cámara, tamaño del objetivo, tiempo, tags de cielo y suelo validados contra Poly Haven, props, frames vacíos con mínimo del 10%);
   - se renderiza de nuevo y el cambio se queda solo si el KID (features ImageNet de MobileNetV2) baja al menos 0,02 (el ruido con 48 renders es de unos 0,02);
   - lo que ninguna palanca puede arreglar sale en `cannot_fix`: dice qué le falta al generador.

Solo se usan las fotos del lado de entreno. Salida: `<job>/match/report.json`, `<job>/match/spec.json` (spec propuesto) y mosaicos por ronda.

```
sim2chip match data/jobs/maceta_counter_11/spec.json --job maceta_counter_11
```

## Primera prueba: macetas (`maceta_counter_11`, 2026-09-29)

KID frente a las 18 fotos de entreno (suelo: una mitad de las fotos contra la otra, 0,018):

| Paso | KID |
|---|---|
| Renders tal cual | 0,383 |
| Con el look de cámara | 0,341 |
| Ronda 1 (fachada 0,69 a 0,2, balcón 0,3, interior 0,3, nuevo lugar suelo 0,2): aceptada | 0,288 |
| Ronda 2 (hasta 20 macetas, encuadre 0,05-0,9, suelo gastado) | 0,353, deshecha |
| Ronda 3 (armarios, coches, toldos, más tiempos) | 0,277, deshecha (mejora < 0,02) |

Duración: unos 25 min en la 3090. El render de prueba tarda 20 s; lo lento es Kimi-K3 (3-4 min por mosaico, los dos en paralelo). Ultra, unos 17 s.

F1 en test real (18 fotos, 92 macetas, 3 semillas, mismo código):

| | F1 | Error de conteo |
|---|---|---|
| C: control | 0,202 ± 0,029 | 3,43 |
| A: + look de cámara | 0,228 ± 0,023 | 2,89 |
| B: + look + escena (ronda 1) | 0,212 ± 0,007 | 3,45 |

Lectura: el look ayuda un poco y es gratis; el cambio de escena acerca los renders según KID, pero no mueve el F1. Con 18 fotos de test las diferencias caben en el ruido. Kimi y Ultra coinciden en que el hueco principal no tiene palanca: las macetas son cilindros negros vacíos (sin tierra, plantas ni variedad), las escenas están vacías, el suelo está demasiado limpio y falta controlar la inclinación de la cámara.

## Pendiente

- [ ] Palancas que pide `cannot_fix`: variedad del objetivo (tierra, plantas, materiales), mezcla de texturas de suelo (calvas, manchas), inclinación de cámara, ventanas con marco.
- [ ] Probar con más fotos de test o con un segundo caso antes de decidir si `match` va por defecto.
- [ ] Web: mosaico antes y después y cambios en lenguaje llano para que el usuario acepte.
- [ ] Nebius: el bucle llama a `LocalBackend`; en la nube, mantener Isaac vivo entre rondas.
- [ ] Coste por ejecución (tokens de Kimi y Ultra) en el panel de uso.

## Segunda prueba: césped del parque (`specs/grass_park.json`, 69 fotos cenitales, 2026-09-29)

Objetivo provisional: hojas caídas (conteo), porque la tarea `grid` aún no existe; lo que se mide es el suelo. Cambios para esta prueba: suelo mezclado (`ground_mix.py`: base con parches de otras texturas de Poly Haven a su tamaño real y franja con borde recto), `camera.pitch_deg`, Kimi-K2.6 en vez de K3 (30 s frente a 4-24 min por comparación), 2 cambios por ronda y palancas bloqueadas (`--lock camera.height_m`: hechos del usuario que el bucle no puede contradecir).

KID (entre fotos reales 0,034): 0,783 tal cual, 0,689 con look, 0,653 cámara cenital + más hojas, **0,530** inclinación 30-60° + cielos con sol. Los parches de suelo (tierra, grava, musgo) empeoraron (1,03) y la franja de acera también (0,645).

Lecciones:
- Sin bloquear la altura, Kimi creyó que la cámara estaba "a centímetros" (las hojas 3D salen enormes) y Ultra bajó la cámara a 0,1-0,3 m: suelo pixelado. Los hechos del usuario deben ir bloqueados.
- Con 4 cambios por ronda uno malo tapaba uno bueno.
- La QA dejó pasar "hojas" que son ramas enteras o una roca con musgo; con el tag `grass` Poly Haven devuelve también tejados de paja y adoquines.
- Pendiente de entender por qué los parches empeoran (¿contraste entre texturas, cobertura 30-70% demasiado alta?): es la siguiente prueba, idealmente con comparación foto a foto.
