# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in queue deadlines for requests that have never started execution.

Set VLLM_MAX_QUEUE_AGE_SECONDS to a finite nonnegative number (0 disables
expiry). Select --scheduler-cls
vllm.v1.core.sched.queue_deadline_scheduler.QueueDeadlineScheduler, or select
AsyncQueueDeadlineScheduler from the same module for async scheduling.
The deadline starts at scheduler ingress and is checked on scheduling steps,
not by a timer. Requests blocked on remote KV transfers, grammar compilation,
or streaming input are excluded. Expiry uses the standard request error path;
the stop reason identifies a queue timeout without changing HTTP error mapping.
"""

import math
import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)
QUEUE_TIMEOUT_STOP_REASON = "queue_wait_deadline_exceeded"


class QueueDeadlineScheduler(Scheduler):
    """Expire never-started requests without interrupting active work."""

    def __init__(self, *args, **kwargs) -> None:
        raw_limit = os.environ.get("VLLM_MAX_QUEUE_AGE_SECONDS", "0")
        try:
            self.max_queue_age_seconds = float(raw_limit)
        except ValueError as exc:
            raise ValueError(
                "VLLM_MAX_QUEUE_AGE_SECONDS must be finite and nonnegative"
            ) from exc
        if (
            not math.isfinite(self.max_queue_age_seconds)
            or self.max_queue_age_seconds < 0
        ):
            raise ValueError(
                "VLLM_MAX_QUEUE_AGE_SECONDS must be finite and nonnegative"
            )
        super().__init__(*args, **kwargs)
        self._queued_at: dict[str, float] = {}
        self._expired: list[Request] = []

    def add_request(self, request: Request) -> None:
        new_request = request.request_id not in self.requests
        super().add_request(request)
        if new_request and self.max_queue_age_seconds:
            self._queued_at[request.request_id] = time.monotonic()

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if self.max_queue_age_seconds:
            now = time.monotonic()
            expired_ids = [
                request.request_id
                for queue in (self.waiting, self.kv_holding_waiting)
                for request in queue
                if request.status == RequestStatus.WAITING
                and request.num_computed_tokens == 0
                and request.num_output_tokens == 0
                and request.num_preemptions == 0
                and (queued_at := self._queued_at.get(request.request_id)) is not None
                and now - queued_at >= self.max_queue_age_seconds
            ]
            if expired_ids:
                self._expired.extend(
                    self.finish_requests(expired_ids, RequestStatus.FINISHED_ERROR)
                )
                logger.warning(
                    "Expired %d requests after %.3fs waiting to start execution",
                    len(expired_ids),
                    self.max_queue_age_seconds,
                )

        result = super().schedule(throttle_prefills)
        if self._queued_at:
            waiting_ids = {
                request.request_id
                for queue in (self.waiting, self.kv_holding_waiting)
                for request in queue
            }
            self._queued_at = {
                req_id: queued_at
                for req_id, queued_at in self._queued_at.items()
                if req_id in waiting_ids
            }
        return result

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        expired, self._expired = self._expired, []
        for request in expired:
            group = outputs.setdefault(request.client_index, EngineCoreOutputs())
            group.outputs.append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    stop_reason=QUEUE_TIMEOUT_STOP_REASON,
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                )
            )
        return outputs


class AsyncQueueDeadlineScheduler(QueueDeadlineScheduler, AsyncScheduler):
    """Queue deadlines with the standard async scheduler's execution behavior."""
