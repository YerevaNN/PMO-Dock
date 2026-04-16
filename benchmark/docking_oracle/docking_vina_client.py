"""
Shared HTTP client for DockingOracle service.

Expected service contract:
- POST {service_url}/predict/{target} with JSON {"smiles": [...], "seed"?: int}
- Response JSON contains {"scores": [...]} where each score is a docking affinity (typically negative).
- The service uses 99.9 to indicate a per-molecule docking failure (mapped to 0.0 by this client).
"""

from __future__ import annotations

import logging
from time import sleep
from typing import List, Union, Tuple

try:
    import requests
except Exception as e:
    requests = None  # type: ignore
    _requests_import_error = e

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def _as_list(smiles_list: Union[List[str], "np.ndarray"]) -> List[str]:
    if _HAS_NUMPY and isinstance(smiles_list, np.ndarray):
        return smiles_list.tolist()
    if not isinstance(smiles_list, list):
        return list(smiles_list)
    return smiles_list


class DockingOracleClient:
    def __init__(
        self,
        service_url: str,
        target: str,
        connect_timeout: int = 60,
        read_timeout: int = 1200,
    ):
        if requests is None:
            raise ImportError(
                "requests is not installed. Install it (e.g. `pip install requests`). "
                f"Original error: {_requests_import_error}"
            )
        service_url = service_url.rstrip("/")
        if not service_url.startswith(("http://", "https://")):
            service_url = f"http://{service_url}"
        self.service_url = service_url
        self.target = target
        self.predict_url = f"{self.service_url}/predict/{target}"
        self._timeout: Tuple[int, int] = (connect_timeout, read_timeout)

    def predict(
        self,
        smiles_list: Union[List[str], "np.ndarray"],
        seed: int = 0,
    ) -> List[float]:
        smiles_list = _as_list(smiles_list)
        max_retries = 5
        retry_sleep = 2
        last_exception = None

        for attempt in range(max_retries):
            try:
                payload = {"smiles": smiles_list}
                if seed is not None:
                    payload["seed"] = seed
                response = requests.post(
                    self.predict_url,
                    json=payload,
                    timeout=self._timeout,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                result = response.json()
                scores = result["scores"]
                return [0.0 if score == 99.9 else float(score) for score in scores]
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logging.warning(
                        "Request timeout (attempt %s/%s) for %s at %s. Retrying in %ss...",
                        attempt + 1,
                        max_retries,
                        self.target,
                        self.predict_url,
                        retry_sleep,
                    )
                    sleep(retry_sleep)
                    continue
                raise
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logging.warning(
                        "Connection error (attempt %s/%s) for %s at %s. Retrying in %ss...",
                        attempt + 1,
                        max_retries,
                        self.target,
                        self.service_url,
                        retry_sleep,
                    )
                    sleep(retry_sleep)
                    continue
                raise
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logging.warning(
                        "Request error (attempt %s/%s) for %s at %s: %s. Retrying in %ss...",
                        attempt + 1,
                        max_retries,
                        self.target,
                        self.predict_url,
                        e,
                        retry_sleep,
                    )
                    sleep(retry_sleep)
                    continue
                raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logging.warning(
                        "Unexpected error (attempt %s/%s) for %s: %s. Retrying in %ss...",
                        attempt + 1,
                        max_retries,
                        self.target,
                        e,
                        retry_sleep,
                    )
                    sleep(retry_sleep)
                    continue
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError(f"All {max_retries} retry attempts failed for {self.target} without exception.")

