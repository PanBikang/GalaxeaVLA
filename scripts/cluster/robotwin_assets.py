"""Reuse shared read-only scene assets and isolate writable robot configurations."""
import json
import subprocess
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
patch = root / 'scripts/cluster/robotwin_device.patch'
already_applied = subprocess.run(['git','apply','--reverse','--check',str(patch)],
                                cwd=root / 'third_party/RoboTwin',capture_output=True).returncode == 0
if not already_applied:
    subprocess.run(['git','apply',str(patch)],cwd=root / 'third_party/RoboTwin',check=True)
source = Path('/public/node03/users/panbk/data/RoboTwin/assets')
target = root / 'third_party/RoboTwin/assets'
target.mkdir(parents=True, exist_ok=True)
for name in ['background_texture', 'objects']:
    src, dest = source / name, target / name
    assert src.is_dir(), src
    if not dest.exists() and not dest.is_symlink():
        dest.symlink_to(src, target_is_directory=True)
    assert dest.resolve() == src.resolve(), dest
if not (target / 'embodiments').exists():
    with zipfile.ZipFile(source / 'embodiments.zip') as archive:
        members = [n for n in archive.namelist() if n.startswith('embodiments/')]
        assert members and all('..' not in Path(n).parts for n in members)
        archive.extractall(target, members=members)
subprocess.run([str(root / '.venv-robotwin/bin/python'), 'script/update_embodiment_config_path.py'],
               cwd=root / 'third_party/RoboTwin', check=True)
out = root / 'runs/robotwin/provenance'
out.mkdir(parents=True, exist_ok=True)
record = {'asset_source': str(source), 'robotwin_commit': subprocess.check_output(
    ['git','rev-parse','HEAD'],cwd=root / 'third_party/RoboTwin',text=True).strip(),
    'curobo_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root / 'third_party/curobo',text=True).strip()}
(out / 'sources.json').write_text(json.dumps(record,indent=2))
print(record,flush=True)
