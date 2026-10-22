# WorldStereo en una VM de Nebius apagada (2026-10-22)

Script: `experiments/worldstereo/ws_vm.py` (`disk`, `prep`, `gpu`, `run <caso>`, `stop`, `status`). Ejecuta el mismo
`job.sh` y la misma imagen que `ws_test.py launch`, pero en una VM RTX PRO 6000 de eu-south1 que se enciende, corre y
se apaga por API. Las salidas van al prefijo `worldstereo_vm/` del bucket para no mezclarse con las de `ws_test.py`.

## Montaje

- Disco `eyeforge-ws-weights`: Network SSD Non-replicated de 279 GiB (3 bloques de 93 GiB, ~330 MiB/s de lectura según
  la documentación). Guarda la caché de Hugging Face (118 GB) y el data-root de Docker (19 GB). Le quedan 125 GB libres.
- El disco se llenó con una VM sin GPU (cpu-d3 8vcpu) que después se borró. La descarga de HF tardó 517 s frente a los
  ~1050 s dentro del trabajo, y la imagen 141 s.
- VM `eyeforge-ws-gpu`: arranque de 50 GiB desde `ubuntu24.04-cuda13.0`, que ya trae Docker con runtime NVIDIA. El
  disco de pesos va como secundario.
- Error encontrado: el Docker de Ubuntu 24.04 usa el almacén de containerd y extrae las capas en
  `/var/lib/containerd`, en el disco de arranque, aunque se cambie `data-root`. Se desactiva con
  `"features": {"containerd-snapshotter": false}`.

## Medida (photo21 con 15 tramos nbv/reconstruct)

| Paso | Trabajo Serverless (ejecución 3b) | VM apagada (medido) |
|---|---|---|
| Arranque y descarga de la imagen | ~6 min | 3.3 min (imagen ya en disco: 1 s) |
| Pesos | 17.5 min de descarga de HF | 1 s (ya en caché) |
| Carga de pesos a memoria | ~0 (recién bajados, en RAM) | ~5-6 min de lectura en frío del NRD (solo los 14 trozos del transformador base: 3:08) |
| 15 tramos | ~16 min | ~16 min |
| **Total GPU** | **~40 min estimados, ~1.20 $** | **25.9 min, 0.78 $** |

Ahorro: ~14 min y ~0.42 $ por ejecución. El coste fijo por arranque baja de ~24 min a ~9 min.

## Coste mientras existe

- Disco NRD de 279 GiB: ~14.8 $/mes. Disco de arranque de 50 GiB: ~3.6 $/mes. Total ~18 $/mes (~0.6 $/día), con la
  VM apagada.
- Se amortiza con unas 45 ejecuciones al mes solo por dinero. Lo que se gana sobre todo es tiempo de iteración.
- Para borrarlo: `nebius compute instance delete` (borra también su disco de arranque) y `nebius compute disk delete`
  del disco de pesos.

## Transformador base fuera (hecho 2026-10-22, decisión del usuario)

El checkpoint DMD trae el transformador entero (sin claves que falten), así que los 65 GB del transformador base de
Wan se leían y luego se sobrescribían. `ws_test.py` (JOB) ya no los descarga y `run_ws.py` hace que diffusers lea
el config de Wan con los pesos del DMD. Probado en la imagen con un modelo pequeño de la misma clase, en fp32 y bf16:
mismos pesos, dtypes y buffers que la carga de siempre, con los pesos base borrados. Ahorro esperado (sin medir en
GPU): ~3 min de carga en la VM y ~8 min de descarga en el trabajo Serverless.

## Mejoras posibles (sin probar)

- Disco de 465 GiB (~25 $/mes): reduciría a la mitad lo que quede de carga.
- Varias escenas por arranque: tras la primera, los pesos salen de la RAM (218 GB) y solo se pagan los tramos.

## Cuidados

- Nunca `shutdown` dentro de la VM: Nebius lo trata como fallo y sigue cobrando. `ws_vm.py run` siempre la para por
  API al final, aunque falle.
- Al encender puede no haber RTX PRO 6000 libre. El trabajo Serverless sigue funcionando como alternativa.
- `ws_vm.py run` sobrescribe `data/worldstereo_test/job.sh`, igual que `ws_test.py launch`. No lanzar los dos a la vez.
