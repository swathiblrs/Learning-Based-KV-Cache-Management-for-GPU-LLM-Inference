"""Build the small source bundle used by the GPU notebook, without datasets/weights."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
out = root/'results'/'gpu-serving-source.zip'
out.parent.mkdir(exist_ok=True)
paths = [root/'pyproject.toml',root/'README.md']
for directory,pattern in [('src','*.py'),('tests','*.py'),('docs','*.md'),('deploy','*')]:
    paths.extend(p for p in (root/directory).rglob(pattern)
                 if p.is_file() and "secrets" not in p.relative_to(root).parts
                 and not p.name.startswith(".env")
                 and (directory != "deploy" or p.suffix in (".json", ".yaml", ".yml", ".txt")))
checksums = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path,path.relative_to(root))
    archive.writestr('source-manifest.json',json.dumps(checksums,indent=2))
print(out)
