"""Optional local performance-ledger integration; no changes to model training."""
from pathlib import Path
import subprocess
import sys
import warnings


def sync_registry(project=None, strict=False):
    project = Path(project) if project is not None else Path(__file__).resolve().parents[1]
    workspace = project.parent
    script = workspace / 'perbarui_performa.py'
    if not script.is_file() or not (workspace / 'konfigurasi_performa.json').is_file():
        return None  # Standalone research copies need not contain the local team ledger.
    try:
        result = subprocess.run([sys.executable, str(script), '--root', str(workspace)],
                                capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=90, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or
                               f'Performance registry exited with {result.returncode}')
        print('Pencatatan performa: ' + result.stdout.strip(), flush=True)
        return result.stdout.strip()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        if strict:
            raise
        warnings.warn(f'Hasil riset tersimpan, tetapi CSV gabungan belum diperbarui: {error}. '
                      'Jalankan PERBARUI_PERFORMA.bat setelah menutup CSV di Excel.', RuntimeWarning)
        return None
