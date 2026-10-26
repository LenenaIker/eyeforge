"""Every relative import names something its module has, read from the source without importing it. Pyflakes
does not look inside other modules, and most of ours are imported inside functions (Isaac's Python, the GPU
workers), so a renamed function only breaks when that code path runs: scene_pick kept importing has_focal after it
became no_focal."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [ROOT / "sim2chip", ROOT / "experiments" / "worldstereo" / "wsroute"]


def defined(tree: ast.Module) -> set[str]:
    """Names a module defines at its top level, including inside if/try blocks there."""
    names: set[str] = set()

    def visit(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    names.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, (ast.If, ast.Try, ast.With)):
                visit(n.body)
                for h in getattr(n, "handlers", []):
                    visit(h.body)
                visit(getattr(n, "orelse", []))
                visit(getattr(n, "finalbody", []))
    visit(tree.body)
    return names


def target(src: Path, level: int, module: str | None) -> Path:
    base = src.parent
    for _ in range(level - 1):
        base = base.parent
    return base.joinpath(*module.split(".")) if module else base


def test_relative_imports_name_what_their_module_has():
    missing = []
    for pkg in PACKAGES:
        for src in pkg.rglob("*.py"):
            for n in ast.walk(ast.parse(src.read_text())):
                if not isinstance(n, ast.ImportFrom) or not n.level:
                    continue
                t = target(src, n.level, n.module)
                if t.is_dir():
                    if (t / "__init__.py").exists():
                        has = defined(ast.parse((t / "__init__.py").read_text()))
                    else:
                        has = set()
                    has |= {p.stem for p in t.glob("*.py")} | {p.name for p in t.iterdir() if p.is_dir()}
                elif t.with_suffix(".py").exists():
                    has = defined(ast.parse(t.with_suffix(".py").read_text()))
                else:
                    missing.append(f"{src.relative_to(ROOT)}:{n.lineno} no module {'.' * n.level}{n.module or ''}")
                    continue
                missing += [f"{src.relative_to(ROOT)}:{n.lineno} {a.name} not in {'.' * n.level}{n.module or ''}"
                            for a in n.names if a.name != "*" and a.name not in has]
    assert not missing, "\n".join(missing)
