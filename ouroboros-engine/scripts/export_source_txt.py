"""Concatena fontes, configurações e documentação em um único TXT legível."""
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[1]
INCLUDED_DIRS = ("configs", "data", "ouroboros", "scripts", "tests", "docs", "third_party")
INCLUDED_ROOT_FILES = (".gitignore", ".gitattributes", "README.md", "pyproject.toml", "requirements.txt", "THIRD_PARTY_NOTICES.md")
INCLUDED_EXTENSIONS = {".py", ".html", ".js", ".ts", ".css", ".c", ".h", ".s",
                       ".toml", ".yaml", ".yml", ".json", ".md", ".sh", ".ini", ".ovm", ".txt"}


def source_files():
    paths = {ROOT / name for name in INCLUDED_ROOT_FILES}
    for directory in INCLUDED_DIRS:
        paths.update(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and not path.is_symlink()
                     and (path.suffix.lower() in INCLUDED_EXTENSIONS or path.name in {"LICENSE", "NOTICE", "COPYING"})
                     and not any(part in {"build", "dist", "__pycache__"} or part.endswith(".egg-info") for part in path.parts))
    # Exemplos de código que a programadora produziu na execução validada.
    paths.update(path for path in (ROOT / "validation/demo/programs").glob("*.py") if path.is_file())
    paths.update(path for path in (ROOT / "validation-v2/programming/generated").glob("*.ovm") if path.is_file())
    if (ROOT / "validation-v2/synthesis/discovered.ovm").is_file():
        paths.add(ROOT / "validation-v2/synthesis/discovered.ovm")
    return sorted((path for path in paths if path.is_file()), key=lambda p: p.relative_to(ROOT).as_posix())


def export(destination):
    destination = Path(destination)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = source_files()
    if destination.resolve() in {path.resolve() for path in files}:
        raise ValueError("O TXT de saída não pode sobrescrever um arquivo-fonte")
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        output.write("OUROBOROS ENGINE 2.0.0 — FONTES CONSOLIDADAS\n")
        output.write(f"Arquivos de texto incluídos: {len(files)}\n")
        output.write("Formato: fontes, configurações, dados e documentação em UTF-8.\n")
        output.write("Os pesos .pt e as capturas PNG são artefatos binários separados.\n")
        output.write("Esta exportação reúne fontes; não é um executável compilado.\n")
        output.write("Inclui bibliotecas upstream com suas licenças, créditos e hashes.\n")
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            output.write("\n" + "=" * 88 + "\n")
            output.write(f"ARQUIVO: {relative}\n")
            output.write("=" * 88 + "\n")
            contents = path.read_text(encoding="utf-8")
            output.write(contents)
            if not contents.endswith("\n"):
                output.write("\n")
    print(f"{destination}: {len(files)} fontes concatenadas, {destination.stat().st_size} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="CODIGO_COMPLETO.txt")
    args = parser.parse_args()
    export(args.output)


if __name__ == "__main__":
    main()
