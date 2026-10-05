# Cuándo pedir, mantener y apagar la GPU: qué dice la literatura

Investigación del 2026-10-05 para ajustar la GPU caliente (`sim2chip/warm.py`). Solo propone; los cambios de valores necesitan la aprobación del usuario y, mejor, la telemetría pendiente (`TODO.md`: tiempo de subida de fotos a entrenar y abandonos).

## Nuestros números

| Concepto | Valor | Fuente |
|---|---|---|
| Arranque de la GPU caliente hasta lista | 4,4 min (2,6 arranque, 1,8 pesos y kits) | `TODO.md`, medido en Nebius 2026-10-04 |
| Precio L40S | 1,55 $/h, unos 0,026 $/min | `sim2chip/backends/nebius.py` (`USD_PER_HOUR`) |
| Coste de un arranque | unos 0,11 $ de GPU, más 4,4 min de espera del usuario si no se esconde | derivado |
| Keep-alive por subida o ronda (`KEEP_S`) | 10 min, unos 0,26 $ | `warm.py` |
| Apagado sin tareas (`IDLE_S`) | 5 min, unos 0,13 $ | `warm.py` |
| Tope de vida (`HOURS`) | 2 h | `warm.py` |
| Entrenamiento PicoDet, 3 semillas | antes 11,6 min; estimado ahora 2-2,5 min (sin medir en Nebius) | `experiments/NOTES.md`, 2026-10-05 |

Con el entrenamiento ya corto, el arranque (4,4 min) pasa a ser lo más largo de la vida de la GPU en un trabajo típico: decidir bien cuándo pedirla y cuándo soltarla pesa más que antes.

## 1. El problema del alquiler de esquís (base teórica)

Mantener la GPU encendida sin trabajo es "alquilar" (se paga por minuto) y apagarla y volver a pedirla es "comprar" (se paga de golpe el arranque). Sin saber cuándo llega la siguiente tarea:

- **Determinista**: esperar ociosa hasta haber gastado en espera lo mismo que cuesta un arranque, y entonces apagar. Nunca cuesta más del doble que el óptimo que conociera el futuro (2-competitivo), y ningún algoritmo determinista lo mejora. Karlin, Manasse, Rudolph y Sleator, "Competitive snoopy caching", *Algorithmica* 3, 1988.
- **Aleatorizado**: elegir el momento de apagar al azar con una distribución concreta baja la cota a e/(e-1), unos 1,58. Karlin, Manasse, McGeoch y Owicki, "Competitive randomized algorithms for nonuniform problems", *Algorithmica* 11, 1994, pp. 542-571. https://link.springer.com/article/10.1007/BF01189993
- Resumen accesible: https://en.wikipedia.org/wiki/Ski_rental_problem

**Para nosotros**: si solo contamos dinero, el punto de equilibrio es esperar tanto como dura un arranque, unos 4,4 min. Nuestro `IDLE_S` de 5 min ya está ahí. Si además contamos la espera del usuario (un arranque visible son 4,4 min parado y un riesgo de abandono), el coste de "comprar" sube y la espera óptima también.

## 2. DelayedOff y AutoScale (servidores con arranque largo)

- Gandhi, Gupta, Harchol-Balter y Kozuch, "Optimality analysis of energy-performance trade-off for server farm management", *Performance Evaluation* 67(11), 2010, pp. 1155-1171. https://www3.cs.stonybrook.edu/~anshul/perf_power_2010.pdf
  - Política **DelayedOff**: un servidor se apaga solo si pasa `t_wait` sin recibir trabajo; si llega uno, el contador vuelve a cero. `t_wait` es constante, independiente de la carga.
  - Demuestran que es asintóticamente óptima con llegadas de Poisson de tasa desconocida.
  - **Regla práctica n.º 4**: `t_wait ≈ T_OFF · P_ON / P_IDLE` (tiempo de arranque por la razón entre lo que cuesta arrancar y lo que cuesta estar ocioso). Es el argumento del alquiler de esquís. Con su ejemplo (arranque 200 s, 240 W encendido, 150 W ocioso) sale 320 s.
- Gandhi, Harchol-Balter, Raghunathan y Kozuch, "AutoScale: Dynamic, Robust Capacity Management for Multi-Tier Data Centers", *ACM TOCS* 30(4), 2012. https://www.cs.cmu.edu/~harchol/Papers/tocs12.pdf
  - Implementado en 38 servidores reales con arranque de 260 s. Usan `t_wait` = 120 s y el resultado es **poco sensible a `t_wait` entre 60 y 260 s**.

**Para nosotros**: en la nube se paga igual arrancando que ociosa (`P_ON / P_IDLE` = 1), así que la regla da `t_wait ≈ T_OFF` ≈ 4,4 min, lo mismo que el alquiler de esquís. AutoScale sugiere que afinar más allá de ese orden de magnitud apenas cambia el resultado.

## 3. Serverless in the Wild (keep-alive según el historial)

Shahrad et al., "Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider", USENIX ATC 2020. https://www.usenix.org/system/files/atc20-shahrad.pdf

- Toda la carga de Azure Functions: el 45 % de las aplicaciones se invoca como mucho una vez por hora y el 81 % como mucho una vez por minuto. La frecuencia varía en 8 órdenes de magnitud.
- Entonces AWS y Azure usaban un keep-alive fijo de 10 y 20 min tras cada ejecución.
- Un keep-alive fijo más largo reduce los arranques en frío, pero gasta mucho más: pasar de 10 min a 1 h baja los arranques en frío de la aplicación en el percentil 75 del 50,3 % al 25 %, y un keep-alive fijo de 2 h gasta casi un 30 % más de memoria ociosa que el de 10 min.
- **Política híbrida**: por aplicación, un histograma de los tiempos entre invocaciones (hasta 4 h). Con el percentil 5 fija una **ventana de pre-calentado** (descarga tras ejecutar y vuelve a cargar justo antes de cuando suele llegar la siguiente) y con el percentil 99 el **keep-alive**, con un margen del 10 %. Si el histograma no es representativo (coeficiente de variación de sus barras bajo), vuelve a un keep-alive fijo; si los tiempos se salen del rango, predice con ARIMA. Consigue los mismos arranques en frío con mucho menos gasto, o muchos menos con el mismo gasto.

**Para nosotros**: es justo la telemetría que tenemos pendiente. Con el histograma de "subida de fotos → pulsa entrenar" (y "ronda de calibración → siguiente ronda"), `KEEP_S` dejaría de ser 10 min fijos: percentil 99 del hueco más un margen. Si el percentil 5 del hueco fuera mayor que el arranque (usuarios que tardan mucho en la entrevista), no haría falta pedir la GPU al subir fotos, sino más tarde.

## 4. Otras ideas académicas aplicables

- **FaasCache** (Fuerst y Sharma, ASPLOS 2021, https://cgi.luddy.indiana.edu/~prateeks/papers/faascache-asplos21.pdf): mantener caliente equivale a una caché de objetos de distinto tamaño; con Greedy-Dual-Size-Frequency (frecuencia, tamaño en memoria y coste de arranque) baja los arranques en frío unas 3 veces frente al TTL fijo. Para nosotros: si la GPU compartida tiene que decidir qué modelos deja cargados (SAM 3, SigLIP 2, Paddle) cuando hay varios usuarios, este es el criterio.
- **IceBreaker** (Roy, Patel y Tiwari, ASPLOS 2022, http://www1.ece.neu.edu/~ningfang/SimPaper/icebreaker-ASPLOS22.pdf): mantener caliente en máquinas baratas lo que tiene poca probabilidad de llegar y en las caras lo probable. Para nosotros ya pasa en parte: con todo en caché, la ronda de calibración corre en el servidor sin GPU.
- **ServerlessLLM** (Fu et al., OSDI 2024, https://www.usenix.org/system/files/osdi24-fu.pdf): en lugar de mantener más tiempo, acortar el arranque (carga de pesos por niveles de almacenamiento cercanos a la GPU); 10-200 veces menos latencia. Para nosotros: cada minuto que se le quita al arranque (kits, pesos) baja también la espera óptima. Es la palanca que ya usamos con los kits del bucket.

## 5. Valores por defecto de la industria

| Plataforma | Espera antes de apagar | Fuente |
|---|---|---|
| AWS Lambda (2020) | 10 min | Shahrad et al. 2020 |
| Azure Functions (2020) | 20 min | Shahrad et al. 2020 |
| Knative | `scale-to-zero-grace-period` 30 s, `scale-to-zero-pod-retention-period` 0 s | https://knative.dev/docs/serving/autoscaling/scale-to-zero/ |
| Kubernetes cluster-autoscaler | `--scale-down-unneeded-time` 10 min, `--scale-down-delay-after-add` 10 min | https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/FAQ.md |
| Modal | `scaledown_window` 60 s (entre 2 s y 20 min); `min_containers` y `buffer_containers` para tener siempre algo caliente | https://modal.com/docs/guide/cold-start |
| RunPod Serverless | idle timeout 5 s, se cobra la espera; FlashBoot conserva el estado para revivir más rápido | https://docs.runpod.io/serverless/endpoints/endpoint-configurations |

Lectura: las plataformas con arranques de segundos (Modal, RunPod con FlashBoot, Knative) esperan segundos o un minuto; las que arrancan nodos enteros (cluster-autoscaler) esperan 10 min. Es la misma regla de antes: la espera sigue al coste del arranque. Nuestro arranque es de minutos, así que nuestra espera debe ser de minutos.

## 6. Recomendaciones

Decisión del usuario (2026-10-05): aplicar con valores más baratos, **2 min sin tareas y 5 min de keep-alive** (antes 5 y 10). Queda por debajo del punto de equilibrio teórico (4,4 min): quien vuelva entre unos 2 y 6 min después paga un arranque visible. La telemetría (`logs/warm_events.jsonl`, `warm.gaps()`) dirá cuántas veces pasa. Recomendaciones originales:

1. **`IDLE_S` = 5 min: mantener.** Coincide con el punto de equilibrio del alquiler de esquís y DelayedOff (arranque 4,4 min, mismo precio ociosa que arrancando). Si el arranque baja, bajarlo en proporción.
2. **`KEEP_S` = 10 min: decidir con datos.** Registrar en la telemetría prevista el hueco entre subida y entrenar, y entre rondas. Con unas decenas de sesiones, aplicar la idea de Serverless in the Wild: keep-alive = percentil 99 del hueco + 10 %, con 10 min como valor por defecto mientras no haya datos suficientes.
3. **Pedir la GPU al subir fotos: mantener, pero medir abandonos.** Cuesta como mucho un arranque más un keep-alive por abandono (unos 0,11 + 0,26 $). Si los abandonos tras subir fotos son frecuentes, pedirla en una señal más tardía (por ejemplo, al confirmar las clases) solo si el hueco que queda sigue siendo mayor que 4,4 min.
4. **No aleatorizar el apagado.** El paso de 2 a 1,58 de cota es teórico y difícil de explicar al jurado; con un solo trabajo a la vez no compensa.
5. **Tope de 2 h: mantener** como red de seguridad ante una GPU olvidada.
6. Lo que más rinde sigue siendo **acortar el arranque** (ServerlessLLM): cada minuto quitado baja a la vez la espera del usuario y la espera óptima.
