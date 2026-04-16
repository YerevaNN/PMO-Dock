"""
Shared local DockingOracle implementation (qvina02 subprocess + OpenBabel 3D).

Moved under `benchmark/docking_oracle/` so benchmark assets own docking.
"""

from functools import cache
import os
import time
import shutil
from shutil import rmtree
import subprocess
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import numpy as np
from openbabel import pybel


_ANTITARGET_TARGETS = frozenset({"7uyt", "5ut5", "7uyw"})
_FAIL_AFFINITY = 99.9
_FAIL_THRESHOLD = 99.0


def _aggregate_antitarget_affinities(runs: list[list[float]], agg: str) -> list[float]:
    arr = np.asarray(runs, dtype=np.float64)
    fail = arr >= _FAIL_THRESHOLD
    masked = np.where(fail, np.nan, arr)
    agg = (agg or "max").strip().lower()
    if agg == "mean":
        per = np.nanmean(masked, axis=0)
    else:
        per = np.nanmin(masked, axis=0)
    all_fail = np.all(fail, axis=0)
    per = np.where(np.isnan(per) | all_fail, _FAIL_AFFINITY, per)
    return per.tolist()


def _obabel_executable() -> str:
    for key in ("OBABEL_CMD", "OBABEL"):
        p = os.environ.get(key)
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    w = shutil.which("obabel")
    return w if w else "obabel"


def _get_grids_dir():
    env = os.environ.get("DOCKING_GRIDS_DIR")
    if env:
        return os.path.abspath(env)
    # prefer benchmark-owned path, but fall back to legacy repo location
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docking_grids")
    if os.path.isdir(here):
        return here
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        legacy = os.path.join(project_root, "docking_oracle", "docking_grids")
        if os.path.isdir(legacy):
            return legacy
    return here


TARGET_BOX = {
    "fa7": {"center": (10.131, 41.879, 32.097), "size": (20.673, 20.198, 21.362)},
    "parp1": {"center": (26.413, 11.282, 27.238), "size": (18.521, 17.479, 19.995)},
    "5ht1b": {"center": (-26.602, 5.277, 17.898), "size": (22.5, 22.5, 22.5)},
    "jak2": {"center": (114.758, 65.496, 11.345), "size": (19.033, 17.929, 20.283)},
    "braf": {"center": (84.194, 6.949, -7.081), "size": (22.032, 19.211, 14.106)},
    "5ut5": {"center": (-8.5, 12.0, 22.0), "size": (18.0, 20.0, 18.0)},
    "6nzp": {"center": (13.0, -5.4, 27.3), "size": (20.0, 22.0, 18.0)},
    "7uyt": {"center": (6.1, 8.6, -19.3), "size": (22.0, 22.0, 22.0)},
    "7uyw": {"center": (15.7, 9.5, 5.5), "size": (20.0, 24.0, 20.0)},
}


@cache
def quickvina_predictor(target, exhaustiveness=None):
    return DockingOracle(target, exhaustiveness=exhaustiveness)


class DockingOracle:
    _docking_semaphore = None
    _max_concurrent_docking = None
    _semaphore_lock = threading.Lock()

    def __init__(self, target, exhaustiveness=None):
        if target not in TARGET_BOX:
            raise ValueError(f"Unsupported target '{target}'")
        box = TARGET_BOX[target]
        self.target = target
        self.box_center = box["center"]
        self.box_size = box["size"]
        grids_dir = _get_grids_dir()
        self.vina_program = os.path.join(grids_dir, "qvina02")
        self.receptor_file = os.path.join(grids_dir, f"{target}.pdbqt")
        if exhaustiveness is not None:
            self.exhaustiveness = exhaustiveness
        elif target in ("6nzp", "7uyt", "5ut5", "7uyw"):
            self.exhaustiveness = 8
        else:
            self.exhaustiveness = 1
        self.num_cpu_dock = int(os.environ.get("NUM_CPU_DOCK", 2))
        self._init_semaphore()
        self.num_modes = 10
        self.timeout_gen3d = 30
        self.timeout_dock = 100
        self.temp_dir = self._make_temp_dir()

    def _init_semaphore(self):
        available = multiprocessing.cpu_count()
        max_dock = os.environ.get("MAX_CONCURRENT_DOCKING")
        if max_dock is not None:
            max_dock = int(max_dock)
        else:
            max_dock = max(1, int(available / self.num_cpu_dock * 0.9))
        with DockingOracle._semaphore_lock:
            if DockingOracle._docking_semaphore is None:
                DockingOracle._docking_semaphore = threading.Semaphore(max_dock)
                DockingOracle._max_concurrent_docking = max_dock
                if logging.getLogger().handlers:
                    logging.info(
                        "DockingOracle: max_concurrent_docking=%s (semaphore), %s CPUs per call",
                        max_dock,
                        self.num_cpu_dock,
                    )

    def _make_temp_dir(self):
        base = os.environ.get("DOCKING_TMP_DIR")
        if base:
            os.makedirs(base, exist_ok=True)
            return base
        out = os.environ.get("OUT_DIR", "/tmp")
        t = time.time()
        i = 0
        while True:
            tmp = os.path.join(out, "tmp", f"{t:.4f}", f"tmp{i}")
            if not os.path.exists(tmp):
                os.makedirs(tmp, exist_ok=True)
                return tmp
            i += 1

    def gen_3d(self, smi, ligand_mol_file):
        obabel = _obabel_executable()
        argv = [obabel, f"-:{smi}", "--gen3D", "-O", ligand_mol_file]
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            close_fds=True,
        )
        try:
            out, _ = process.communicate(timeout=self.timeout_gen3d)
            rc = process.returncode
            if rc != 0:
                raise RuntimeError(f"obabel failed (rc={rc}): {out.strip()[:400]}")
            if not os.path.exists(ligand_mol_file) or os.path.getsize(ligand_mol_file) == 0:
                raise RuntimeError(
                    f"obabel produced no mol file: {ligand_mol_file} (out={out.strip()[:400]})"
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdout:
                process.stdout.close()
            if process.stderr and process.stderr != process.stdout:
                process.stderr.close()

    def docking(
        self,
        receptor_file,
        ligand_mol_file,
        ligand_pdbqt_file,
        docking_pdbqt_file,
        seed: int = 0,
        docking_semaphore=None,
    ):
        if docking_semaphore is not None:
            docking_semaphore.acquire()
        try:
            ms = list(pybel.readfile("mol", ligand_mol_file))
            m = ms[0]
            m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
            run_line = (
                f"{self.vina_program} --receptor {receptor_file} --ligand {ligand_pdbqt_file} --out {docking_pdbqt_file}"
                f" --center_x {self.box_center[0]} --center_y {self.box_center[1]} --center_z {self.box_center[2]}"
                f" --size_x {self.box_size[0]} --size_y {self.box_size[1]} --size_z {self.box_size[2]}"
                f" --cpu {self.num_cpu_dock} --num_modes {self.num_modes} --exhaustiveness {self.exhaustiveness}"
                f" --seed {seed}"
            )
            process = subprocess.Popen(
                run_line.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                close_fds=True,
            )
            try:
                result, _ = process.communicate(timeout=self.timeout_dock)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if process.stdout:
                    process.stdout.close()
                if process.stderr and process.stderr != process.stdout:
                    process.stderr.close()
            result_lines = result.split("\n")
            check_result = False
            affinity_list = []
            for line in result_lines:
                if line.startswith("-----+"):
                    check_result = True
                    continue
                if not check_result:
                    continue
                if line.startswith("Writing output") or line.startswith("Refine time"):
                    break
                lis = line.strip().split()
                if not (lis and lis[0].isdigit()):
                    break
                affinity_list.append(float(lis[1]))
            return affinity_list
        finally:
            if docking_semaphore is not None:
                docking_semaphore.release()

    def _process_single_molecule(self, idx_smi_tuple, seed: int = 0):
        idx, smi = idx_smi_tuple
        sem = DockingOracle._docking_semaphore
        tid = threading.current_thread().ident
        ligand_mol = f"{self.temp_dir}/ligand_{tid}_{idx}.mol"
        ligand_pdbqt = f"{self.temp_dir}/ligand_{tid}_{idx}.pdbqt"
        dock_pdbqt = f"{self.temp_dir}/dock_{tid}_{idx}.pdbqt"
        try:
            self.gen_3d(smi, ligand_mol)
        except Exception as e:
            logging.warning("gen_3d error for molecule %s: %s", idx, str(e)[:200])
            self._cleanup([ligand_mol, ligand_pdbqt, dock_pdbqt])
            return (idx, _FAIL_AFFINITY)
        try:
            aff_list = self.docking(
                self.receptor_file,
                ligand_mol,
                ligand_pdbqt,
                dock_pdbqt,
                seed=seed,
                docking_semaphore=sem,
            )
            affinity = aff_list[0] if aff_list else _FAIL_AFFINITY
        except Exception as e:
            logging.warning("docking error for molecule %s: %s", idx, str(e)[:200])
            affinity = _FAIL_AFFINITY
        finally:
            self._cleanup([ligand_mol, ligand_pdbqt, dock_pdbqt])
        return (idx, affinity)

    def _predict_single_seed(self, smiles_list, seed: int = 0):
        if not smiles_list:
            return []
        sem = DockingOracle._docking_semaphore
        max_workers = min(len(smiles_list), 200) if sem else 1
        indexed = list(enumerate(smiles_list))
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_molecule, x, seed): x[0] for x in indexed
            }
            for future in as_completed(futures):
                try:
                    idx, aff = future.result()
                    results[idx] = aff
                except Exception as e:
                    idx = futures[future]
                    logging.error("Unexpected error for molecule %s: %s", idx, e)
                    results[idx] = _FAIL_AFFINITY
        return [results[i] for i in range(len(smiles_list))]

    def predict(self, smiles_list, seed: int = 0):
        if not smiles_list:
            return []
        if self.target in _ANTITARGET_TARGETS:
            n_rep = int(os.environ.get("ANTITARGET_DOCK_REPEATS", "3"))
            n_rep = max(1, n_rep)
            agg = os.environ.get("ANTITARGET_DOCK_AGG", "max")
            if n_rep == 1:
                return self._predict_single_seed(smiles_list, seed)
            runs = [self._predict_single_seed(smiles_list, seed + k) for k in range(n_rep)]
            return _aggregate_antitarget_affinities(runs, agg)
        return self._predict_single_seed(smiles_list, seed)

    def _cleanup(self, paths):
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def __del__(self):
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            try:
                rmtree(self.temp_dir)
            except Exception:
                pass

