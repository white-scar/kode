#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Server-side queue for the current Kode component ablation.

One worker manages one dataset. It waits for an idle GPU, runs a 1-epoch
smoke test for linear and affine drift, then runs the full 10-fold / 1000-epoch
component ablation. At most two workers should be launched (maize + arab).
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/data/fandata/dashuaige')
REPRO = ROOT / 'reproduce'
OUT = ROOT / 'outputs' / 'reproduce' / 'queue_logs'
LOCKDIR = REPRO / '.gpu_queue_locks'
PY = Path('/data/fandata/WWC/anaconda3/envs/dashuaige/bin/python')
HP = REPRO / 'knode_hyperparam_ablation.py'
MODES = 'ode_hyper,iter_hyper'

SCRIPT = {
    'maize': REPRO / 'knode_maize_np_tp1_affine.py',
    'arab': REPRO / 'knode_arab_np_tp1_affine.py',
}
OUTDIR = {
    'maize': ROOT / 'outputs' / 'kode_result' / 'maize',
    'arab': ROOT / 'outputs' / 'kode_result' / 'arab',
}


def log(msg, path):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def gpu_rows():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'],
        text=True,
    )
    rows = []
    for raw in out.strip().splitlines():
        parts = [x.strip() for x in raw.split(',')]
        if len(parts) >= 3:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def free_gpu_indices():
    return [idx for idx, mem, util in gpu_rows() if mem < 1000 and util < 5]


def remove_stale_locks():
    if not LOCKDIR.exists():
        return
    for lock in LOCKDIR.glob('gpu_*.lock'):
        try:
            pid = int(lock.read_text(encoding='utf-8').strip() or '0')
        except Exception:
            pid = 0
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if not alive:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


def acquire_gpu(log_path):
    LOCKDIR.mkdir(parents=True, exist_ok=True)
    while True:
        remove_stale_locks()
        for idx in free_gpu_indices():
            lock = LOCKDIR / f'gpu_{idx}.lock'
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                log(f'acquired GPU {idx}', log_path)
                return idx, lock
            except FileExistsError:
                continue
        log(f'no free GPU; waiting 60s (free={free_gpu_indices()})', log_path)
        time.sleep(60)


def run_cmd(cmd, tag, log_path, gpu):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    log(f'RUN {tag} on GPU {gpu}: ' + ' '.join(str(x) for x in cmd), log_path)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '=' * 80 + '\n')
        subprocess.run([str(x) for x in cmd], cwd=str(REPRO), env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    log(f'DONE {tag}', log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['maize', 'arab'], required=True)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f'component_ablation_{a.dataset}.log'
    log(f'worker start: dataset={a.dataset}', log_path)

    gpu, lock = acquire_gpu(log_path)
    try:
        script = SCRIPT[a.dataset]
        # Smoke: 1 fold, 1 epoch, all four component variants.
        for drift in ['linear', 'affine']:
            smoke_tag = f'_smoke_{drift}'
            run_cmd([PY, script, '--epochs', '1', '--foldsets', '1',
                     '--drift', drift, '--mode', MODES, '--tag', smoke_tag],
                    f'{a.dataset}_smoke_{drift}', log_path, gpu)
            smoke_summary = OUTDIR[a.dataset] / f'knode_tp1_all_summary{smoke_tag}.csv'
            if not smoke_summary.exists():
                raise RuntimeError(f'smoke output missing: {smoke_summary}')

        # Full component ablation: 10 fold-sets, 1000 epochs.
        for drift in ['linear', 'affine']:
            run_cmd([PY, script, '--epochs', '1000', '--foldsets', '10',
                     '--drift', drift, '--mode', MODES, '--tag', f'_{drift}'],
                    f'{a.dataset}_full_{drift}', log_path, gpu)

        # Hyperparameter ablation: run the restored current-protocol runner.
        run_cmd([PY, HP, '--dataset', a.dataset, '--run-all',
                 '--epochs', '1000', '--foldsets', '10',
                 '--drift', 'affine', '--mode', 'ode_hyper,iter_hyper'],
                f'{a.dataset}_hyperparam', log_path, gpu)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        log(f'released GPU {gpu}', log_path)

    log(f'worker complete: dataset={a.dataset}', log_path)


if __name__ == '__main__':
    main()