# Despliegue de la demo pública

Decisión del usuario (2026-10-03): EyeForge comparte el servidor Hetzner de a-letheia (<server-ip>, 2 vCPU, 4 GB). a-letheia puede caerse unas horas sin problema. El trabajo de GPU va a Nebius; el servidor solo sirve la web, la entrevista y las subidas.

## Cómo encaja

- **Nginx de a-letheia** (rama `eyeforge-proxy` de su repo, commit 1a93d6d):
  - tiene los puertos 80 y 443;
  - sirve `eyeforge.app` con el certificado de origen de Cloudflare (`/etc/ssl/cloudflare/eyeforge.app.pem` y `.key`);
  - pasa las peticiones al contenedor `eyeforge:8000` por la red Docker externa `eyeforge`.
- **Contenedor `eyeforge`** (`deploy/docker-compose.yml` en `/opt/eyeforge`):
  - sin puertos publicados;
  - topes de 1,5 GB de memoria y 1,5 CPU;
  - datos en `/opt/eyeforge/data` (trabajos, subidas, `accounts.db`);
  - perfil de la CLI de Nebius en `/opt/eyeforge/nebius`;
  - secretos en `/opt/eyeforge/.env` (plantilla `deploy/env.example`, chmod 600).
- **Cloudflare** (gratis):
  - proxy con SSL "Full (strict)";
  - corta las peticiones de más de 100 MB, así que la web sube las fotos en tandas de unos 40 MB.

## Primera vez

1. **Usuario: Cloudflare.**
   - Añadir `eyeforge.app` y poner sus nameservers en Porkbun.
   - Crear el registro A `@` (y `www`) a <server-ip>, con proxy.
   - Recrear los registros MX del reenvío de correo de Porkbun.
   - Crear el certificado de origen y copiarlo al servidor en `/etc/ssl/cloudflare/`.
2. **Usuario: OAuth.**
   - Cliente de Google con la redirección `https://eyeforge.app/auth/google/callback`.
   - OAuth App de GitHub con el callback `https://eyeforge.app/auth/github/callback`.
3. **Usuario: secretos.** Crear `/opt/eyeforge/.env` en el servidor a partir de `deploy/env.example`. Nunca por chat.
4. **Servidor** (con el visto bueno del usuario):
   - swap de 2 a 4 GB, porque no tiene ninguno y quedan 1,9 GB libres;
   - `docker network create eyeforge`.
5. **Credenciales de Nebius** (tarea "Restaurar las credenciales de Nebius" de `TODO.md`):
   - copiar el fichero de clave de la cuenta de servicio a `/opt/eyeforge/nebius/`;
   - crear el perfil dentro del contenedor, para que las rutas sean las suyas:
     `docker exec -it eyeforge nebius profile create --service-account-file /root/.nebius/<fichero>.json ...`
6. **Desplegar EyeForge** desde la workstation: `deploy/deploy.sh`.
   - Construye la imagen aquí: el servidor no tiene memoria para compilar la web y las dependencias.
   - La carga en el servidor y arranca el contenedor.
7. **Desplegar a-letheia** con la rama `eyeforge-proxy`. Basta con recrear su contenedor de nginx. Su `deploy.sh` reconstruye todo con `--no-cache`.
8. **Comprobar:**
   - entrar con una cuenta normal (sin créditos);
   - entrar con una de jurado (`docker exec eyeforge sim2chip accounts judge <correo>`);
   - lanzar una ejecución y verla desde el móvil con la misma cuenta.

## Mantenimiento

Por SSH, dentro del contenedor:
- `docker exec eyeforge sim2chip accounts list`;
- `accounts judge|unjudge|ban|unban <correo>`;
- `credits grant <correo> <n>`;
- logs con `docker logs -f eyeforge`.

Copia diaria (2026-10-09): `deploy/backup.sh`, a las 04:15 hora del servidor, por `/etc/cron.d/eyeforge-backup`. Ambos los instala `deploy.sh`.
- Copia en caliente de `accounts.db` (saldos y pagos) y `library.db`, comprobadas y comprimidas.
- Se guardan 7 días en `/opt/eyeforge/data/backups`, con el registro en `backup.log`.
- Restaurar: parar el contenedor, `gunzip` de la copia sobre `data/accounts.db` y arrancarlo.
- Las copias están en el mismo disco: no protegen si se pierde el servidor.

Actualizar la versión: `deploy/deploy.sh` otra vez. Los datos están en el volumen y no se pierden.
