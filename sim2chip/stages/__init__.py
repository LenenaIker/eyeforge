"""Stage registry, in execution order. Stage modules import heavy libraries only inside run()."""
from __future__ import annotations

from .base import Context, Stage
from .convert_assets import ConvertAssets
from .crops import Crops
from .evaluate import Evaluate
from .export import Export
from .fetch_assets import FetchAssets
from .ingest import Ingest
from .qa_assets import QAAssets
from .render import Render
from .train import Train

STAGES: list[Stage] = [Ingest(), FetchAssets(), ConvertAssets(), QAAssets(), Render(), Crops(), Train(), Evaluate(),
                       Export()]
BY_NAME = {s.name: s for s in STAGES}

__all__ = ["Context", "Stage", "STAGES", "BY_NAME"]
