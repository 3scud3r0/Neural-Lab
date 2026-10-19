"""Empacota fontes e evidências; exclui ambientes, caches e saídas locais."""
import argparse
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root.parent / "ouroboros-engine.zip")
    args = parser.parse_args()
    excluded = {".venv", "venv", "__pycache__", ".git", "outputs", "outputs-v2", "build", "dist", ".pytest_cache", ".cache"}
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in excluded or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.is_file() and not path.is_symlink() and path.resolve() != destination and path.suffix != ".pyc":
                archive.write(path, Path(root.name) / relative)
    with zipfile.ZipFile(destination) as archive:
        corrupted = archive.testzip()
        if corrupted:
            raise RuntimeError(f"Arquivo corrompido: {corrupted}")
        print(f"{destination}: {len(archive.namelist())} arquivos, {destination.stat().st_size} bytes")


if __name__ == "__main__":
    main()
