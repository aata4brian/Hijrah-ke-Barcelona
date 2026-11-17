"""Verify byte-identical inference in a new output folder; never modify a submission."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import site
import subprocess
import sys
from importlib import metadata
from .data import fingerprint
from .evaluate import atomic_json


def environment_evidence():
    """Inspect isolation; do not equate a venv with a fresh installation."""
    prefix = Path(sys.prefix).resolve()
    config = prefix / 'pyvenv.cfg'
    settings = {}
    if config.is_file():
        for line in config.read_text(encoding='utf-8-sig').splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                settings[key.strip().lower()] = value.strip().lower()
    packages, outside, missing = {}, [], []
    for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'pandas',
                 'threadpoolctl', 'numba', 'llvmlite', 'torch', 'opencv-contrib-python'):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        location = Path(distribution.locate_file('')).resolve()
        packages[name] = distribution.version
        if not location.is_relative_to(prefix):
            outside.append(name)
    isolated = (sys.prefix != sys.base_prefix
                and settings.get('include-system-site-packages') == 'false'
                and not site.ENABLE_USER_SITE and not outside and not missing)
    return dict(isolated_environment=isolated, clean_environment=isolated,
                fresh_installation_verified=False, python_version=sys.version.split()[0],
                include_system_site_packages=settings.get('include-system-site-packages'),
                user_site_enabled=bool(site.ENABLE_USER_SITE),
                packages=packages, packages_outside_environment=outside,
                missing_required_packages=missing,
                limitation='Isolation is inspected automatically; a fresh installation needs its separate install log.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error('--timeout must be positive')
    if not args.model.is_file() or not args.expected.is_file() or not args.data_dir.is_dir():
        parser.error('Model, expected CSV and dataset folder must already exist')
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / args.expected.name
    status = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  complete=False, byte_identical=False,
                  model_sha256=fingerprint(args.model),
                  expected_sha256=fingerprint(args.expected),
                  target_ground_truth_read=False,
                  **environment_evidence())
    result_path = args.output / 'reproduction_result.json'
    atomic_json(result_path, status)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
               LOKY_MAX_CPU_COUNT='4', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', '-m', 'riset_v4.predict', '--model', str(args.model),
               '--data-dir', str(args.data_dir), '--output', str(destination)]
    try:
        with (args.output / 'inference.log').open('w', encoding='utf-8') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout, check=False)
        status['exit_code'] = process.returncode
        if process.returncode:
            raise RuntimeError(f'Inference exited with {process.returncode}; see inference.log')
        status['actual_sha256'] = fingerprint(destination)
        status['byte_identical'] = status['actual_sha256'] == status['expected_sha256']
        if not status['byte_identical']:
            raise RuntimeError('Regenerated CSV is not byte-identical to the expected candidate')
        status['complete'] = True
    except Exception as error:
        status['error'] = str(error)
        raise
    finally:
        status['finished_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(result_path, status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
