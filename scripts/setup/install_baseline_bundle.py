"""Install the extra-dependency ZIP into an existing nova3r environment offline."""
import argparse
import json
import platform
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def check_compiler():
    """Exercise the shared-library link used by Triton's CUDA helper builds."""
    compiler = shlex.split(os.environ.get('CC', 'gcc'))
    if not compiler or shutil.which(compiler[0]) is None:
        raise SystemExit('Missing C compiler. Install build-essential and python3-dev, or set CC.')
    with tempfile.TemporaryDirectory(prefix='nova3r-compiler-check-') as temporary:
        folder = Path(temporary)
        source = folder / 'check.c'
        source.write_text('#include <Python.h>\n#include <stdlib.h>\n'
                          'void *baseline_alloc(size_t n) { return malloc(n); }\n')
        result = subprocess.run(
            compiler + [str(source), '-shared', '-fPIC', '-O3',
                        '-I' + sysconfig.get_path('include'),
                        '-o', str(folder / 'check.so')],
            text=True, capture_output=True,
        )
        if result.returncode:
            raise SystemExit(
                'GenReCon compiler/linker prerequisite check failed:\n' + result.stderr +
                '\nOn Debian/Ubuntu, rerun with --install-system-deps (requires apt/network '
                'and root or sudo), or repair build-essential, libc6-dev, and the '
                'libgcc development package matching your compiler. '
                'Python wheels do not supply these system libraries.'
            )
    print('GenReCon compiler/linker prerequisite check passed.', flush=True)


def install_system_dependencies():
    if shutil.which('apt-get') is None:
        raise SystemExit('--install-system-deps currently supports Debian/Ubuntu (apt-get) only.')
    prefix = [] if os.geteuid() == 0 else ['sudo']
    if prefix and shutil.which('sudo') is None:
        raise SystemExit('--install-system-deps requires root or sudo.')
    subprocess.run(prefix + ['apt-get', 'update'], check=True)
    subprocess.run(prefix + ['apt-get', 'install', '--reinstall', '-y',
                             'build-essential', 'libc6-dev', 'python3-dev'], check=True)
    # Reinstall the active GCC's development package too: build-essential alone
    # will not restore missing files in an already-installed libgcc package.
    compiler = shlex.split(os.environ.get('CC', 'gcc'))
    version = subprocess.check_output(compiler + ['-dumpversion'], text=True).strip().split('.')[0]
    if not version.isdigit():
        raise SystemExit('Could not determine GCC major version; repair libgcc development files manually.')
    subprocess.run(prefix + ['apt-get', 'install', '--reinstall', '-y',
                             f'libgcc-{version}-dev'], check=True)


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',nargs='?',type=Path,default=root.parent.parent/'deps.zip')
    parser.add_argument('--dry-run',action='store_true',help='Validate and show pip actions without installing packages')
    parser.add_argument('--install-system-deps', action='store_true',
                        help='Repair Debian/Ubuntu compiler prerequisites using apt (requires network and root/sudo)')
    args=parser.parse_args()
    if args.dry_run and args.install_system_deps:
        parser.error('--dry-run cannot be combined with --install-system-deps')
    if not args.archive.is_file(): parser.error(f'Archive not found: {args.archive}')
    if args.install_system_deps:
        install_system_dependencies()
    check_compiler()
    import torch
    import torchvision
    with tempfile.TemporaryDirectory(prefix='nova3r-baseline-extras-') as temporary:
        folder=Path(temporary)
        with zipfile.ZipFile(args.archive) as archive:
            names=archive.namelist()
            if 'deps/manifest.json' not in names: raise SystemExit('Not an extra-dependency bundle (missing deps/manifest.json).')
            for info in archive.infolist():
                rel=PurePosixPath(info.filename)
                if rel.is_absolute() or '..' in rel.parts or (info.external_attr>>16)&0o170000==0o120000:
                    raise SystemExit(f'Unsafe ZIP entry: {info.filename}')
            manifest=json.loads(archive.read('deps/manifest.json'))
            for name, actual in [('torch',torch.__version__),('torchvision',torchvision.__version__)]:
                if actual != manifest[name]: raise SystemExit(f'{name} mismatch: bundle requires {manifest[name]}, environment has {actual}. Rebuild bundle for this environment.')
            if torch.version.cuda != manifest['cuda']:
                raise SystemExit(f'CUDA mismatch: bundle requires {manifest["cuda"]}, environment has {torch.version.cuda}')
            if platform.machine()!=manifest['platform'] or platform.python_version().split('.')[:2]!=manifest['python'].split('.')[:2]:
                raise SystemExit('Python version or CPU architecture does not match bundle.')
            archive.extractall(folder)
        base=folder/'deps'
        for group in ('common','volfill'):
            wheels=[base/'wheels'/group/item['wheel'] for item in manifest[group].values()]
            if not wheels or not all(p.is_file() for p in wheels): raise SystemExit(f'Missing wheels for {group}')
            command=[sys.executable,'-m','pip','install','--no-index','--no-deps']
            if args.dry_run: command+=['--dry-run']
            if group=='volfill':
                command+=['--target',str(Path(sys.prefix)/'share/nova3r-baselines/volfill'),'--upgrade']
            command += [str(p) for p in wheels]
            subprocess.run(command,check=True)
        if not args.dry_run:
            subprocess.run([sys.executable,'-m','pip','check'],check=True)
        print('Bundle validation complete.' if args.dry_run else 'Extra baseline dependencies installed. Existing PyTorch/CUDA stack preserved.')
if __name__=='__main__': main()
