"""Create a self-contained Colab notebook: code and public sample, no uploads."""
import base64
import hashlib
import json
from pathlib import Path
import textwrap

root=Path(__file__).resolve().parents[1]
archive=(root/'results/gpu-serving-source.zip').read_bytes()
digest=hashlib.sha256(archive).hexdigest()
encoded='\n'.join(textwrap.wrap(base64.b64encode(archive).decode(),120))
notebook=json.loads((root/'notebooks/gpu_public_pressure_upload.ipynb').read_text())
notebook['cells'][0]['source']=['# Public-trace GPU benchmark — no uploads required\n',
    '\n','1. Select **Runtime → Change runtime type → T4 GPU → Save**.\n',
    '2. Click **Runtime → Run all**.\n',
    '3. Keep the tab connected and download the results ZIP when it finishes.\n',
    '\n','All project code and the verified public metadata sample are included in this notebook. There is no file chooser, source ZIP upload, Drive mount or manual code edit. Dependencies and SmolLM2 weights download automatically.\n',
    '\n','The fixed pilot uses 24 ordered, uniformly sampled requests, six retained prefixes and a 96 MiB cache. Four policies run twice plus one correctness reference: 216 request executions. This is a capped, surrogate-token replay of public production metadata, not live-production validation.\n']
setup='''#@title Automatic setup — no file upload needed
import os, sys, subprocess, hashlib, json, base64, io, zipfile
from pathlib import Path
from google.colab import files
# Frozen source and verified CC-BY-4.0 public metadata, embedded below.
EMBEDDED_SOURCE_SHA256 = "DIGEST"
EMBEDDED_SOURCE = b"""ENCODED"""
source_bytes = base64.b64decode(EMBEDDED_SOURCE)
assert hashlib.sha256(source_bytes).hexdigest() == EMBEDDED_SOURCE_SHA256, "Embedded source checksum failed"
repo = Path("/content") / ("kv-public-auto-" + EMBEDDED_SOURCE_SHA256[:12])
repo.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source_bytes)) as z:
    assert "source-manifest.json" in z.namelist()
    for name in z.namelist():
        assert (repo / name).resolve().is_relative_to(repo.resolve()), "Invalid archive path"
    z.extractall(repo)
for name, expected in json.loads((repo / "source-manifest.json").read_text()).items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, "Source file checksum failed"
os.chdir(repo)
os.environ["USE_TF"] = "0"
print("Project and verified public sample loaded automatically.", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "virtualenv>=20.26,<21"], check=True)
subprocess.run([sys.executable, "-m", "virtualenv", "--system-site-packages", "--no-download", ".colab-venv"], check=True)
python = str(repo / ".colab-venv/bin/python")
subprocess.run([python, "-m", "pip", "install", "-e", ".[serving,traces]"], check=True)
print("Setup complete. The next cells verify the GPU and run the benchmark.", flush=True)
'''.replace('DIGEST',digest).replace('ENCODED',encoded)
notebook['cells'][1]['source']=setup.splitlines(True)
notebook['cells'][1]['metadata']={'cellView':'form'}
notebook['cells'][3]['source']=['## Verify the included sample\n','The sample is extracted automatically from the notebook. No dataset download or manual file selection is needed.\n']
notebook['cells'][4]['source']=[line.replace('Use the updated source ZIP that includes the sample','Embedded sample is missing; reopen the latest self-contained notebook') for line in notebook['cells'][4]['source']]
notebook['metadata']['colab']={'name':'gpu_public_pressure.ipynb','provenance':[]}
notebook['metadata']['accelerator']='GPU'
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        cell['execution_count']=None;cell['outputs']=[]
for name in ('gpu_public_pressure.ipynb','gpu_public_pressure_upload.ipynb'):
    (root/'notebooks'/name).write_text(json.dumps(notebook,indent=2)+'\n')
print('Built self-contained notebook; source SHA256:',digest)
