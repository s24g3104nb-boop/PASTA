"""Build a working-tree Windows distribution from explicit code/asset allowlists."""
import argparse
from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile
from datetime import datetime,timedelta,timezone

APP='pasta_weight_app_v2'
DENIED_PARTS={'data','user_data','__pycache__','.venv','.git','.pytest_cache','models','backups'}

def inputs(root):
    files=[]
    for path in (root/APP).rglob('*.py'):
        rel=path.relative_to(root)
        if (len(rel.parts)==2 or rel.parts[1] in {'pasta_ml','experiments','tests'}) and not set(rel.parts)&DENIED_PARTS and not path.is_symlink():files.append(rel)
    files += [Path(APP)/n for n in ('requirements.txt','requirements-dev.txt','requirements-research.txt','ui/design.css','sample_images/demo_spread_12.png','sample_images/demo_bundle_16.png')]
    files += [Path('.streamlit/config.toml')]
    files += [Path('tools/check_safari_layout.py')]
    files += [Path('docs/assets/capture_sheet')/n for n in ('Pasta_Lens_A4_reference_v1.pdf','Pasta_Lens_A4_reference_v1.svg','geometry.json','README_JA.md')]
    files += [Path('docs')/n for n in ('32_CAPTURE_COLLECTION_PLAN.md','35_UI_POLISH_REPORT.md','36_MANUAL_WEIGHT_RECOVERY.md','37_INTEGRATION_RELEASE.md','38_DENSE_WEIGHT_DIAGNOSIS.md')]
    return sorted(set(files))


def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Use a new output directory; previous packages are never overwritten')
    output.mkdir(parents=True)
    for rel in inputs(root):
        src=root/rel
        if not src.is_file() or src.is_symlink():raise ValueError('Missing/unsafe input: '+str(rel))
        target=output/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    for src in sorted((root/'packaging/windows').iterdir()):
        if src.suffix not in {'.bat','.py','.md'}:continue
        if src.is_symlink():raise ValueError('Unsafe packaging input: '+str(src))
        target=output/src.name
        if src.suffix=='.bat':target.write_bytes(src.read_text().replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
        else:shutil.copy2(src,target)
    # pip resolves includes relative to the requirements file, not the shell cwd.
    requirements=(root/'packaging/windows/requirements.txt').read_text().replace('-r ../../pasta_weight_app_v2/requirements.txt', '-r pasta_weight_app_v2/requirements.txt')
    (output/'requirements.txt').write_text(requirements)
    shutil.copy2(root/'packaging/windows/.gitignore',output/'.gitignore')
    info=dict(date=datetime.now(timezone(timedelta(hours=9))).date().isoformat(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source=('working_tree_including_uncommitted_changes' if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip() else 'committed_tree'),personal_data_included=False,models_included=False)
    (output/'RELEASE_INFO.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    from importlib.metadata import version
    (output/'TESTED_VERSIONS.txt').write_text('Linux/Python3.12 verification environment; NOT a Windows lock file\n'+'\n'.join(n+'=='+version(n) for n in ('streamlit','opencv-python-headless','numpy','Pillow','pandas'))+'\n')
    paths=sorted(p for p in output.rglob('*') if p.is_file())
    (output/'FILES.txt').write_text('\n'.join(str(p.relative_to(output)).replace('\\','/') for p in paths)+'\nFILES.txt\nMANIFEST_SHA256.txt\n')
    paths=sorted(p for p in output.rglob('*') if p.is_file())
    (output/'MANIFEST_SHA256.txt').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(output).as_posix() for p in paths)+'\n')
    archive=output.with_suffix('.zip')
    if archive.exists():raise ValueError('Archive already exists')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,output.name+'/'+p.relative_to(output).as_posix())
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    print(build(Path(__file__).resolve().parents[1],args.output))
