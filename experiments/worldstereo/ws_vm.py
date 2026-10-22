"""WorldStereo on a stopped Nebius VM whose disk already holds the weights and the image (eu-south1).

The Serverless job spends ~17.5 of its 31 min downloading ~126 GB of weights from Hugging Face (run 3b,
2026-10-22). Here a non-replicated SSD keeps the Hugging Face cache and Docker's data root between runs: the GPU VM
is started, runs ws_test.py's own job script in the same image, and is stopped again.

    python experiments/worldstereo/ws_vm.py disk            # create the weights disk (once)
    python experiments/worldstereo/ws_vm.py prep            # CPU VM: format, pull image, download weights, delete VM
    python experiments/worldstereo/ws_vm.py gpu             # create the GPU VM on that disk, left stopped
    python experiments/worldstereo/ws_vm.py run photo21_vm  # start, run the case, fetch, stop (add `resume` to keep done legs)
    python experiments/worldstereo/ws_vm.py fetch photo21_vm  # outputs again, e.g. after a failed run
    python experiments/worldstereo/ws_vm.py stop            # stop the GPU VM by hand
    python experiments/worldstereo/ws_vm.py status

Read speed grows with the disk size: NRD gives 110 MiB/s per 93 GiB, so 279 GiB reads ~330 MiB/s (docs,
2026-10-22). Never `shutdown` inside the VM: Nebius counts it as a failure and keeps billing; stop goes through
the API. Outputs go to their own S3 prefix so ws_test.py's fetch never sees them.
`set -a; . ./.env; set +a` first, like ws_test.py launch."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test  # noqa: E402

PROJECT = ws_test.GPUS["rtx6000"][2]  # eu-south1
SUBNET = "vpcsubnet-e07kkk34en8a16h34a"
DISK, DISK_GIB = "eyeforge-ws-weights", 279  # 3 NRD units of 93 GiB
CPU_VM, GPU_VM = "eyeforge-ws-prep", "eyeforge-ws-gpu"
USER = "eyeforge"
MNT = "/mnt/w"
REGISTRY = os.environ.get("SIM2CHIP_NEBIUS_REGISTRY", "")
CLI = os.environ.get("SIM2CHIP_NEBIUS_CLI") or str(Path.home() / ".nebius/bin/nebius")
LOG = ws_test.DATA / "vm_times.jsonl"
ws_test.PREFIX = "worldstereo_vm"

# Same downloads as ws_test.JOB, so its snapshot_download finds everything in the cache.
DOWNLOAD = """from huggingface_hub import snapshot_download
snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
snapshot_download("hanshanxue/WorldStereo", allow_patterns=["__MODEL__/config.json", "__MODEL__/model.safetensors"])
snapshot_download("facebook/dinov2-base")""".replace("__MODEL__", ws_test.MODEL)

# Mount the weights disk (formatted on first use) and move Docker's data root onto it. Without the classic
# storage driver, Ubuntu 24.04's Docker extracts layers under /var/lib/containerd on the small boot disk.
MOUNT = f"""set -e
dev=/dev/disk/by-id/virtio-weights
sudo blkid $dev || sudo mkfs.ext4 -q -L weights $dev
sudo mkdir -p {MNT} && (mountpoint -q {MNT} || sudo mount -o noatime $dev {MNT})
grep -q {MNT} /etc/fstab || echo "LABEL=weights {MNT} ext4 noatime,nofail 0 2" | sudo tee -a /etc/fstab
command -v docker || nvidia-smi -L || (sudo apt-get update -qq && sudo apt-get install -y -qq docker.io)
sudo mkdir -p {MNT}/docker {MNT}/hf
f=/etc/docker/daemon.json; [ -f $f ] || echo '{{}}' | sudo tee $f
if ! grep -q {MNT}/docker $f; then
  sudo python3 -c "import json; d=json.load(open('$f')); d['data-root']='{MNT}/docker'; d['features']={{'containerd-snapshotter': False}}; json.dump(d, open('$f','w'))"
  sudo systemctl restart docker
fi
sudo docker info --format '{{{{.DockerRootDir}}}} {{{{.Driver}}}}'
"""


def neb(*args, check=True) -> str:
    r = subprocess.run([CLI, *args], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"nebius {' '.join(args[:3])} failed: {r.stderr[-800:]}")
    return r.stdout


def get(kind: str, name: str) -> dict | None:
    out = neb("compute", kind, "get-by-name", "--parent-id", PROJECT, "--name", name, "--format", "json", check=False)
    return json.loads(out) if out.strip().startswith("{") else None


def note(**kw):
    kw["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a") as f:
        f.write(json.dumps(kw) + "\n")
    print(kw, flush=True)


def ip(vm: dict) -> str:
    return vm["status"]["network_interfaces"][0]["public_ip_address"]["address"].split("/")[0]


def ssh(host: str, cmd: str, check=True, **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10", f"{USER}@{host}", cmd], text=True, **kw)
    if check and r.returncode:
        sys.exit(f"ssh failed ({r.returncode}): {cmd[:200]}")
    return r


def wait_ssh(name: str) -> str:
    """Wait for RUNNING and an answering sshd; returns the public address."""
    t0 = time.time()
    while time.time() - t0 < 900:
        vm = get("instance", name)
        if vm and vm["status"].get("state") == "RUNNING" and vm["status"].get("network_interfaces"):
            host = ip(vm)
            if ssh(host, "true", check=False, capture_output=True).returncode == 0:
                return host
        time.sleep(10)
    sys.exit(f"{name} not reachable after 15 min")


def create_vm(name: str, platform: str, preset: str, image_family: str, boot_gib: int, stopped=False):
    disk = get("disk", DISK) or sys.exit(f"no disk {DISK}: run `disk` first")
    key = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
    user_data = f"#cloud-config\nusers:\n  - name: {USER}\n    sudo: ALL=(ALL) NOPASSWD:ALL\n" \
                f"    shell: /bin/bash\n    ssh_authorized_keys:\n      - {key}\n"
    neb("compute", "instance", "create", "--parent-id", PROJECT, "--name", name,
        "--resources-platform", platform, "--resources-preset", preset,
        "--boot-disk-attach-mode", "read_write",
        "--boot-disk-managed-disk-name", f"{name}-boot", "--boot-disk-managed-disk-type", "network_ssd",
        "--boot-disk-managed-disk-size-gibibytes", str(boot_gib),
        "--boot-disk-managed-disk-source-image-family-image-family", image_family,
        "--secondary-disks", json.dumps([{"attach_mode": "read_write", "device_id": "weights",
                                          "existing_disk": {"id": disk["metadata"]["id"]}}]),
        "--network-interfaces", json.dumps([{"name": "eth0", "subnet_id": SUBNET, "ip_address": {},
                                             "public_ip_address": {}}]),
        "--cloud-init-user-data", user_data, *(["--stopped"] if stopped else []))


def docker_login(host: str):
    token = neb("iam", "get-access-token").strip()
    ssh(host, f"sudo docker login {REGISTRY.split('/')[0]} -u iam --password-stdin", input=token,
        capture_output=True)


def image() -> str:
    return f"{REGISTRY}/eyeforge-worldstereo:{ws_test.image_tag()}"


def cmd_disk():
    if get("disk", DISK):
        return print(f"{DISK} exists")
    neb("compute", "disk", "create", "--parent-id", PROJECT, "--name", DISK,
        "--type", "network_ssd_non_replicated", "--size-gibibytes", str(DISK_GIB))
    print(f"created {DISK} ({DISK_GIB} GiB NRD)")


def cmd_prep():
    """A CPU VM fills the disk without paying GPU time, then is deleted (the disk is not its own, it stays)."""
    if not get("instance", CPU_VM):
        create_vm(CPU_VM, "cpu-d3", "8vcpu-32gb", "ubuntu24.04-driverless", 20)
    t0 = time.time()
    host = wait_ssh(CPU_VM)
    note(step="prep_boot", s=round(time.time() - t0))
    try:
        ssh(host, MOUNT)
        docker_login(host)
        t0 = time.time()
        ssh(host, f"sudo docker pull -q {image()}")
        note(step="prep_pull", s=round(time.time() - t0))
        t0 = time.time()
        ssh(host, f"sudo docker run --rm -e HF_XET_HIGH_PERFORMANCE=1 -v {MNT}/hf:/root/.cache/huggingface "
                  f"--entrypoint python {image()} -c '{DOWNLOAD}'")
        note(step="prep_weights", s=round(time.time() - t0))
        # Cold read of the biggest file: what a stopped VM pays on every start.
        r = ssh(host, f"sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; "
                      f"f=$(sudo find {MNT}/hf -name '*.safetensors' -size +5G | head -1); "
                      f"sudo dd if=$f of=/dev/null bs=4M 2>&1 | tail -1; sudo du -sh {MNT}/hf {MNT}/docker; df -h {MNT}",
                capture_output=True)
        print(r.stdout)
        note(step="prep_read", out=r.stdout.strip().splitlines()[0] if r.stdout else "")
    finally:
        neb("compute", "instance", "delete", "--id", get("instance", CPU_VM)["metadata"]["id"])
        print(f"deleted {CPU_VM}")


def cmd_gpu():
    if get("instance", GPU_VM):
        return print(f"{GPU_VM} exists")
    create_vm(GPU_VM, *ws_test.GPUS["rtx6000"][:2], "ubuntu24.04-cuda13.0", 50, stopped=True)
    print(f"created {GPU_VM} (stopped)")


def cmd_stop():
    vm = get("instance", GPU_VM)
    if vm and vm["status"].get("state") not in ("STOPPED", "STOPPING"):
        neb("compute", "instance", "stop", "--id", vm["metadata"]["id"])
    print(f"{GPU_VM} stopped")


def cmd_run(case: str, resume: str = ""):
    """Start the GPU VM, run ws_test.JOB in the image with the cached weights, fetch, always stop."""
    vm = get("instance", GPU_VM) or sys.exit(f"no {GPU_VM}: run `gpu` first")
    max_hours = 1.0
    ws_test.launch(case, "rtx6000", max_hours, dry=True, resume=resume == "resume")  # scene up, job.sh, no job
    from sim2chip.cloud_sync import S3Store
    log_url = S3Store().s3.generate_presigned_url(
        "put_object", Params={"Bucket": ws_test.BUCKET, "Key": f"{ws_test.PREFIX}/out/job.log"},
        ExpiresIn=int((max_hours + 2) * 3600))
    t0 = time.time()
    if vm["status"].get("state") != "RUNNING":
        neb("compute", "instance", "start", "--id", vm["metadata"]["id"])
    try:
        host = wait_ssh(GPU_VM)
        note(step="gpu_boot", case=case, s=round(time.time() - t0))
        ssh(host, MOUNT, capture_output=True)
        ssh(host, "sudo docker info | grep -qi nvidia")  # the CUDA image's own Docker; apt's has no GPU runtime
        subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "LogLevel=ERROR", str(ws_test.DATA / "job.sh"), f"{USER}@{host}:/tmp/ws_job.sh"],
                       check=True)
        docker_login(host)
        t1 = time.time()
        ssh(host, f"sudo docker pull -q {image()}")  # a no-op when prep already pulled it
        note(step="gpu_pull", case=case, s=round(time.time() - t1))
        t1 = time.time()
        ssh(host, f"timeout {int(max_hours * 3600)} sudo docker run --rm --gpus all --shm-size 32g "
                  f"-e LOG_URL='{log_url}' -v {MNT}/hf:/root/.cache/huggingface -v /tmp/ws_job.sh:/opt/ws_job.sh:ro "
                  f"{image()} bash /opt/ws_job.sh", check=False)
        note(step="gpu_job", case=case, s=round(time.time() - t1))
    finally:
        cmd_stop()
        note(step="gpu_total", case=case, s=round(time.time() - t0))
    ws_test.fetch(case)


def cmd_status():
    for kind, name in (("disk", DISK), ("instance", CPU_VM), ("instance", GPU_VM)):
        r = get(kind, name)
        print(name, r["status"].get("state") if r else "-")


if __name__ == "__main__":
    {"disk": cmd_disk, "prep": cmd_prep, "gpu": cmd_gpu, "run": cmd_run, "fetch": ws_test.fetch, "stop": cmd_stop,
     "status": cmd_status}[sys.argv[1]](*sys.argv[2:])
