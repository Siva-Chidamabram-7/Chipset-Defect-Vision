from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "Chipset-Defect-Vision_full_workspace_export.txt"
SKIP_DIRS = {".git"}
CODE_DIR_HINTS = {
    "app",
    "frontend",
    "scripts",
    "training",
}
CODE_FILE_NAMES = {
    "dockerfile",
    "makefile",
    "readme.md",
    "requirements.txt",
    ".gitignore",
    ".dockerignore",
    "data.yaml",
    "classes.txt",
}
CODE_EXTENSIONS = {
    ".bat",
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".env",
    ".go",
    ".graphql",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".properties",
    ".ps1",
    ".py",
    ".rb",
    ".scss",
    ".sh",
    ".sql",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
IMAGE_EXTENSIONS = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}
NON_CODE_EXTENSIONS = {
    ".cache",
    ".doc",
    ".docx",
    ".pdf",
    ".pickle",
    ".pth",
    ".pt",
    ".pyc",
    ".zip",
}
NON_CODE_PATH_PARTS = {
    "__pycache__",
    "data/labels",
    "data/images",
    "incoming_data",
    "raw_data",
    "runs",
    "training/data",
    "weights",
}
MAX_TEXT_FILE_SIZE = 512 * 1024


def relative_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def os_walk_sorted(root: Path) -> Iterable[tuple[str, list[str], list[str]]]:
    import os

    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(name for name in dir_names if name not in SKIP_DIRS)
        file_names.sort()
        yield current_root, dir_names, file_names


def iter_dirs_and_files() -> tuple[list[Path], list[Path]]:
    dirs: list[Path] = []
    files: list[Path] = []
    for current_root, dir_names, file_names in os_walk_sorted(ROOT):
        current_path = Path(current_root)
        dirs.append(current_path)
        for file_name in file_names:
            files.append(current_path / file_name)
    return dirs, files


def is_text_file(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_TEXT_FILE_SIZE:
            return False
        raw = path.read_bytes()
    except OSError:
        return False
    if b"\x00" in raw:
        return False
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            raw.decode(encoding)
            return True
        except UnicodeDecodeError:
            continue
    return False


def looks_like_code_file(path: Path) -> bool:
    rel = relative_posix(path)
    lower_rel = rel.lower()
    if path.name.lower() == OUTPUT_PATH.name.lower():
        return False
    if path.suffix.lower() in IMAGE_EXTENSIONS | NON_CODE_EXTENSIONS:
        return False
    if any(part in lower_rel for part in NON_CODE_PATH_PARTS):
        return False
    if path.name.lower() in CODE_FILE_NAMES:
        return is_text_file(path)
    if path.suffix.lower() in CODE_EXTENSIONS:
        return is_text_file(path)
    if any(part in CODE_DIR_HINTS for part in path.parts):
        return is_text_file(path)
    return False


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return "[Unable to decode file content]"


def build_export() -> tuple[Path, int, int, int, int]:
    dirs, files = iter_dirs_and_files()
    code_files = [path for path in files if looks_like_code_file(path)]
    image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    other_files = [path for path in files if path not in code_files and path not in image_files]
    extension_counts = Counter(path.suffix.lower() or "[no extension]" for path in files)

    lines: list[str] = []
    lines.append("Chipset Defect Vision Workspace Export")
    lines.append("=" * 40)
    lines.append(f"Root folder: {ROOT}")
    lines.append(f"Total folders (excluding .git): {len(dirs)}")
    lines.append(f"Total files (excluding .git): {len(files)}")
    lines.append(f"Embedded code/text files: {len(code_files)}")
    lines.append(f"Image files listed by name: {len(image_files)}")
    lines.append(f"Other non-code files listed by name: {len(other_files)}")
    lines.append("")
    lines.append("Extension Summary")
    lines.append("-" * 40)
    for extension, count in sorted(extension_counts.items()):
        lines.append(f"{extension}: {count}")
    lines.append("")
    lines.append("Folder Inventory")
    lines.append("-" * 40)
    for directory in dirs:
        rel = "." if directory == ROOT else relative_posix(directory)
        lines.append(rel)
    lines.append("")
    lines.append("Image File Names")
    lines.append("-" * 40)
    for file_path in image_files:
        lines.append(relative_posix(file_path))
    lines.append("")
    lines.append("Other Non-Code File Names")
    lines.append("-" * 40)
    for file_path in other_files:
        lines.append(relative_posix(file_path))
    lines.append("")
    lines.append("Embedded Code And Text Files")
    lines.append("-" * 40)
    for index, file_path in enumerate(code_files, start=1):
        lines.append("")
        lines.append(f"[{index}] {relative_posix(file_path)}")
        lines.append(f"Size: {file_path.stat().st_size} bytes")
        lines.append("-" * 40)
        lines.append(read_text_file(file_path))
        lines.append("")
        lines.append("=" * 80)

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return OUTPUT_PATH, len(dirs), len(files), len(code_files), len(image_files)


if __name__ == "__main__":
    output_path, folder_count, file_count, code_count, image_count = build_export()
    print(f"Created: {output_path}")
    print(f"Folders: {folder_count}")
    print(f"Files: {file_count}")
    print(f"Embedded code/text files: {code_count}")
    print(f"Image files listed by name: {image_count}")
