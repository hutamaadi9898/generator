from __future__ import annotations

import base64
from typing import Any

import httpx

from app.schemas import RunpodJobState, RunpodSubmission


class RunpodClient:
    def __init__(
        self,
        api_key: str,
        api_base: str,
        client: httpx.Client | None = None,
        submission_mode: str = "plain",
        endpoint_function_names: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.client = client or httpx.Client(timeout=120)
        self.submission_mode = submission_mode.strip().lower() or "plain"
        self.endpoint_function_names = endpoint_function_names or {}

    def submit_job(self, endpoint_id: str, payload: dict[str, Any]) -> RunpodSubmission:
        if not endpoint_id:
            raise ValueError("RunPod endpoint id is not configured.")
        response = self.client.post(
            f"{self.api_base}/{endpoint_id}/run",
            headers=self._headers(),
            json={"input": self._build_input(endpoint_id, payload)},
        )
        response.raise_for_status()
        body = response.json()
        output, _ = self._extract_output(body.get("output"))
        return RunpodSubmission(
            job_id=str(body.get("id") or body.get("job_id") or ""),
            status=str(body.get("status") or "IN_QUEUE"),
            submitted_at=body.get("createdAt") or body.get("submitted_at"),
            artifacts=output,
        )

    def get_status(self, endpoint_id: str, job_id: str) -> RunpodJobState:
        response = self.client.get(
            f"{self.api_base}/{endpoint_id}/status/{job_id}",
            headers=self._headers(),
        )
        response.raise_for_status()
        body = response.json()
        output, extracted_error = self._extract_output(body.get("output"))
        return RunpodJobState(
            job_id=str(body.get("id") or job_id),
            status=str(body.get("status") or "IN_QUEUE"),
            output=output,
            error=str(body.get("error") or extracted_error or ""),
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _build_input(self, endpoint_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.submission_mode == "plain":
            return payload
        if self.submission_mode != "flash_function":
            raise ValueError(f"Unsupported RunPod submission mode: {self.submission_mode}")

        function_name = self.endpoint_function_names.get(endpoint_id, "").strip()
        if not function_name:
            raise ValueError(f"Missing RunPod function name for endpoint '{endpoint_id}'.")

        return {
            "function_name": function_name,
            "execution_type": "function",
            "serialization_format": "cloudpickle",
            "kwargs": {key: self._encode_arg(value) for key, value in payload.items()},
        }

    def _extract_output(self, output: Any) -> tuple[dict[str, Any], str]:
        if not isinstance(output, dict):
            return {}, ""

        if "success" not in output:
            return output, ""

        if not output.get("success"):
            return {}, str(output.get("error") or "")

        json_result = output.get("json_result")
        if isinstance(json_result, dict):
            return json_result, ""

        decoded_result = self._decode_result(output.get("result"))
        if isinstance(decoded_result, dict):
            return decoded_result, ""

        return {}, ""

    def _decode_result(self, value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return None

        try:
            import cloudpickle
        except ImportError:
            return None

        try:
            return cloudpickle.loads(base64.b64decode(value))
        except Exception:
            return None

    def _encode_arg(self, value: Any) -> str:
        try:
            import cloudpickle
        except ImportError as exc:
            raise RuntimeError(
                "cloudpickle is required for RUNPOD_SUBMISSION_MODE=flash_function."
            ) from exc

        return base64.b64encode(cloudpickle.dumps(value)).decode("utf-8")
