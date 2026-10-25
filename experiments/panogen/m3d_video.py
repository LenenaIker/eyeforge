"""Runs Matrix-3D's code/panoramic_image_to_video.py (under torchrun, cwd = the Matrix-3D repo) with one change:
DiffSynth's ModelConfig calls ModelScope (~4 MB/s here) even for files checkpoints/ already holds; files in place
skip the download. DiffSynth also redirects Wan's T5 and tokenizer to Wan-AI/Wan2.1-T2V-1.3B, so that folder holds
symlinks to the Wan2.2-TI2V-5B copies."""
import glob
import os
import runpy
import sys

sys.path.insert(0, os.path.abspath("code"))
sys.path.insert(0, os.path.abspath("code/DiffSynth-Studio"))
from diffsynth.utils import ModelConfig  # noqa: E402

download = ModelConfig.download_if_necessary


def download_if_missing(self, *a, **k):
    if self.path is None and self.model_id and self.origin_file_pattern and \
            glob.glob(os.path.join(self.local_model_path or "./models", self.model_id, self.origin_file_pattern)):
        self.skip_download = True
    return download(self, *a, **k)


ModelConfig.download_if_necessary = download_if_missing
sys.argv = ["code/panoramic_image_to_video.py", *sys.argv[1:]]
runpy.run_path("code/panoramic_image_to_video.py", run_name="__main__")
