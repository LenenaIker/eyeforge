"""What the web tells the user while a run waits for the cloud or works on a stage."""
from sim2chip.server import phase


def test_phase_follows_the_cloud_then_the_stages():
    log = ["job j  backend nebius", "  nebius job aijob-1 (gpu-l40s-a)", "  state PROVISIONING"]
    assert "assign an L40S" in phase(log, True)
    log += ["  state STARTING", "  state IMAGE_PULLING"]
    assert "image" in phase(log, True)
    log += ["  state RUNNING"]
    assert "installing the environment" in phase(log, True)
    log += ["> label (vision model on Nebius Token Factory)"]
    assert phase(log, True) == "Running label on Nebius"
    assert "failed" in phase(log + ["FAILED: stage label did not finish"], True)
    assert phase(log, False) is None


def test_phase_of_a_local_run():
    assert phase(["job j  backend local", "> render (local GPU)"], True) == "Running render"
