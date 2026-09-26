"""Stage registry, in execution order. Stage modules import heavy libraries only inside run()."""
from __future__ import annotations

from .base import Context, Stage
from .convert_assets import ConvertAssets
from .crops import Crops
from .evaluate import Evaluate
from .export import Export
from .fetch_assets import FetchAssets
from .generate_assets import GenerateAssets
from .ingest import Ingest
from .label import Label
from .points import Points
from .preview_assets import PreviewAssets
from .qa_assets import QAAssets
from .render import Render
from .train import Train

STAGES: list[Stage] = [Label(), GenerateAssets(), Ingest(), FetchAssets(), ConvertAssets(), PreviewAssets(),
                       QAAssets(), Render(), Crops(), Points(), Train(), Evaluate(), Export()]
BY_NAME = {s.name: s for s in STAGES}

__all__ = ["Context", "Stage", "STAGES", "BY_NAME"]
