from __future__ import annotations

from dataclasses import asdict
from time import monotonic
from typing import Any, Awaitable, Callable, Dict, List, Optional
import asyncio
import json

from .mcp import MCPCall, MCPClient, MCPInvocationContext
from .memory import SharedMemory
from .protocol import WriteIntent, WritePolicy
from .reward import HierarchicalRewardModel
from .types import (
    ActionType,
    Budget,
    EpisodeTrace,
    GroupResult,
    GroupSpec,
    GroupStatus,
    MemoryObjectType,
    StepRecord,
    WriteReason,
    new_id,
    utc_now_iso,
)


ProgramFn = Callable[["GroupContext"], Awaitable[Any]]


class GroupContext:
    def __init__(self, runtime: "MCPRLMRuntime", group: GroupSpec) -> None:
        self._runtime = runtime
        self._group = group
        self._step_index = 0
        self._local_state: Dict[str, Any] = {}

    @property
    def group_id(self) -> str:
        return self._group.group_id

    @property
    def episode_id(self) -> str:
        return self._group.episode_id

    @property
    def goal(self) -> str:
        return self._group.goal

    @property
    def input_payload(self) -> Dict[str, Any]:
        return self._group.input_payload

    @property
    def local_state(self) -> Dict[str, Any]:
        return self._local_state

    def context_usage(self) -> float:
        payload_chars = len(json.dumps(self._group.input_payload, ensure_ascii=False, default=str))
        local_chars = len(json.dumps(self._local_state, ensure_ascii=False, default=str))
        total_chars = payload_chars + local_chars
        budget_chars = 24000
        return min(1.0, total_chars / float(budget_chars))

    async def read_memory(self, key: str) -> Any:
        async def op() -> Any:
            value = await self._runtime.memory.read_latest(key)
            return None if value is None else asdict(value)

        return await self._record(
            action_type=ActionType.READ_MEMORY,
            action_name="read_memory",
            payload={"key": key},
            op=op,
        )

    async def write_memory(
        self,
        *,
        key: str,
        object_type: MemoryObjectType,
        content: Any,
        reason: WriteReason,
        confidence: float = 1.0,
        expected_version: Optional[int] = None,
        force: bool = False,
    ) -> Any:
        async def op() -> Any:
            intent = WriteIntent(
                episode_id=self.episode_id,
                group_id=self.group_id,
                key=key,
                object_type=object_type,
                reason=reason,
                content=content,
                confidence=confidence,
                context_usage=self.context_usage(),
                expected_version=expected_version,
                force=force,
            )
            event = await self._runtime._commit_write(intent)
            return None if event is None else asdict(event)

        return await self._record(
            action_type=ActionType.WRITE_MEMORY,
            action_name="write_memory",
            payload={
                "key": key,
                "object_type": object_type.value,
                "reason": reason.value,
                "force": force,
            },
            op=op,
        )

    async def call_object(
        self,
        object_name: str,
        payload: Dict[str, Any],
        *,
        timeout_seconds: float = 30.0,
    ) -> Any:
        async def op() -> Any:
            self._runtime._reserve_object_calls(self._group, additional=1)
            result = await self._runtime.mcp_client.call(
                MCPCall(object_name=object_name, payload=payload, timeout_seconds=timeout_seconds),
                MCPInvocationContext(episode_id=self.episode_id, group_id=self.group_id),
            )
            if not result.ok:
                raise RuntimeError(f"MCP call failed on {object_name}: {result.error}")
            return result.output

        return await self._record(
            action_type=ActionType.CALL_OBJECT,
            action_name="call_object",
            payload={"object_name": object_name, "timeout_seconds": timeout_seconds},
            op=op,
        )

    async def call_objects(self, calls: List[MCPCall]) -> List[Any]:
        async def op() -> List[Any]:
            self._runtime._reserve_object_calls(self._group, additional=len(calls))
            results = await self._runtime.mcp_client.call_many(
                calls,
                MCPInvocationContext(episode_id=self.episode_id, group_id=self.group_id),
            )
            errors = [r for r in results if not r.ok]
            if errors:
                raise RuntimeError(f"{len(errors)} MCP calls failed")
            return [r.output for r in results]

        return await self._record(
            action_type=ActionType.CALL_OBJECT,
            action_name="call_objects",
            payload={"num_calls": len(calls)},
            op=op,
        )

    async def spawn_groups(self, specs: List[Dict[str, Any]]) -> List[str]:
        async def op() -> List[str]:
            planned = [
                {
                    "goal": str(spec.get("goal", "")),
                    "program": str(spec.get("program", "")),
                }
                for spec in specs
            ]
            await self._runtime._commit_internal_write(
                group=self._group,
                key=f"plan/{self.group_id}/spawn",
                object_type=MemoryObjectType.PLAN,
                reason=WriteReason.SPAWN_PREP,
                content={"planned_children": planned, "count": len(planned)},
            )

            existing_children = len(self._runtime.children.get(self.group_id, []))
            if existing_children + len(specs) > self._group.budget.max_children:
                raise RuntimeError("Child group budget exceeded")

            created_ids: List[str] = []
            for spec in specs:
                child_id = self._runtime._create_group(
                    episode_id=self.episode_id,
                    goal=str(spec["goal"]),
                    program=str(spec["program"]),
                    input_payload=dict(spec.get("input_payload", {})),
                    parent_group_id=self.group_id,
                    budget=spec.get("budget"),
                )
                self._runtime._start_group(child_id)
                created_ids.append(child_id)
            return created_ids

        return await self._record(
            action_type=ActionType.SPAWN_GROUPS,
            action_name="spawn_groups",
            payload={"num_specs": len(specs)},
            op=op,
        )

    async def join_groups(self, group_ids: List[str]) -> List[GroupResult]:
        async def op() -> List[GroupResult]:
            tasks = [self._runtime.group_tasks[group_id] for group_id in group_ids]
            results = await asyncio.gather(*tasks)
            await self._runtime._commit_internal_write(
                group=self._group,
                key=f"artifact/{self.group_id}/join",
                object_type=MemoryObjectType.ARTIFACT,
                reason=WriteReason.JOIN_RESULT,
                content={
                    "joined_groups": [r.group_id for r in results],
                    "statuses": [r.status.value for r in results],
                },
            )
            return results

        return await self._record(
            action_type=ActionType.JOIN_GROUPS,
            action_name="join_groups",
            payload={"group_ids": list(group_ids)},
            op=op,
        )

    async def finalize(self, output: Any) -> Any:
        async def op() -> Any:
            await self._runtime._commit_internal_write(
                group=self._group,
                key=f"final/{self.group_id}",
                object_type=MemoryObjectType.ARTIFACT,
                reason=WriteReason.FINALIZE,
                content={"output": output},
            )
            return output

        return await self._record(
            action_type=ActionType.FINALIZE,
            action_name="finalize",
            payload={},
            op=op,
        )

    async def _record(
        self,
        *,
        action_type: ActionType,
        action_name: str,
        payload: Dict[str, Any],
        op: Callable[[], Awaitable[Any]],
    ) -> Any:
        self._step_index += 1
        if self._step_index > self._group.budget.max_steps:
            raise RuntimeError("Step budget exceeded")

        started_at = utc_now_iso()
        ok = False
        result: Any = None
        error: Optional[str] = None
        try:
            result = await op()
            ok = True
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            ended_at = utc_now_iso()
            self._runtime._append_step(
                StepRecord(
                    step_id=new_id("step"),
                    episode_id=self.episode_id,
                    group_id=self.group_id,
                    step_index=self._step_index,
                    action_type=action_type,
                    action_name=action_name,
                    payload=payload,
                    result=result,
                    ok=ok,
                    started_at=started_at,
                    ended_at=ended_at,
                    error=error,
                )
            )


class MCPRLMRuntime:
    def __init__(
        self,
        *,
        program_registry: Any,
        mcp_client: MCPClient,
        memory: Optional[SharedMemory] = None,
        reward_model: Optional[HierarchicalRewardModel] = None,
        write_policy: Optional[WritePolicy] = None,
        max_group_concurrency: int = 64,
    ) -> None:
        self.program_registry = program_registry
        self.mcp_client = mcp_client
        self.memory = memory or SharedMemory()
        self.reward_model = reward_model or HierarchicalRewardModel()
        self.write_policy = write_policy or WritePolicy()
        self.max_group_concurrency = max_group_concurrency

        self._group_semaphore = asyncio.Semaphore(max_group_concurrency)
        self._reset_episode_state()

    @property
    def groups(self) -> Dict[str, GroupSpec]:
        return self._groups

    @property
    def children(self) -> Dict[str, List[str]]:
        return self._children

    @property
    def group_tasks(self) -> Dict[str, "asyncio.Task[GroupResult]"]:
        return self._group_tasks

    def _reset_episode_state(self) -> None:
        self._episode_id: Optional[str] = None
        self._groups: Dict[str, GroupSpec] = {}
        self._group_tasks: Dict[str, asyncio.Task[GroupResult]] = {}
        self._children: Dict[str, List[str]] = {}
        self._steps: List[StepRecord] = []
        self._group_step_count: Dict[str, int] = {}
        self._group_object_calls: Dict[str, int] = {}
        self._group_write_count: Dict[str, int] = {}
        self._group_last_write_time: Dict[str, float] = {}

    async def run_episode(
        self,
        *,
        goal: str,
        program: str,
        input_payload: Optional[Dict[str, Any]] = None,
        budget: Optional[Budget] = None,
    ) -> EpisodeTrace:
        self._reset_episode_state()
        episode_id = new_id("ep")
        self._episode_id = episode_id
        started_at = utc_now_iso()

        root_group_id = self._create_group(
            episode_id=episode_id,
            goal=goal,
            program=program,
            input_payload=dict(input_payload or {}),
            parent_group_id=None,
            budget=budget,
        )
        self._start_group(root_group_id)

        root_result = await self._group_tasks[root_group_id]

        while True:
            pending = [task for task in self._group_tasks.values() if not task.done()]
            if not pending:
                break
            await asyncio.gather(*pending)

        self._compute_total_rewards(root_group_id)
        ended_at = utc_now_iso()

        memory_events = await self.memory.export_events(episode_id=episode_id)
        root_group = self._groups[root_group_id]
        success = root_group.status == GroupStatus.SUCCEEDED and root_result.status == GroupStatus.SUCCEEDED

        return EpisodeTrace(
            episode_id=episode_id,
            root_group_id=root_group_id,
            goal=goal,
            started_at=started_at,
            ended_at=ended_at,
            success=success,
            root_output=root_group.result,
            groups=list(self._groups.values()),
            steps=list(self._steps),
            memory_events=memory_events,
        )

    def _create_group(
        self,
        *,
        episode_id: str,
        goal: str,
        program: str,
        input_payload: Dict[str, Any],
        parent_group_id: Optional[str],
        budget: Optional[Budget],
    ) -> str:
        group_id = new_id("grp")
        depth = 0
        if parent_group_id is not None:
            depth = self._groups[parent_group_id].depth + 1
            self._children.setdefault(parent_group_id, []).append(group_id)

        group = GroupSpec(
            group_id=group_id,
            episode_id=episode_id,
            goal=goal,
            program=program,
            parent_group_id=parent_group_id,
            depth=depth,
            budget=budget or Budget(),
            input_payload=input_payload,
        )
        self._groups[group_id] = group
        self._children.setdefault(group_id, [])
        self._group_step_count[group_id] = 0
        self._group_object_calls[group_id] = 0
        self._group_write_count[group_id] = 0
        return group_id

    def _start_group(self, group_id: str) -> None:
        self._group_tasks[group_id] = asyncio.create_task(self._run_group(group_id))

    async def _run_group(self, group_id: str) -> GroupResult:
        async with self._group_semaphore:
            group = self._groups[group_id]
            group.status = GroupStatus.RUNNING
            group.started_at = utc_now_iso()

            output: Any = None
            try:
                program_fn: ProgramFn = self.program_registry.get(group.program)
                output = await asyncio.wait_for(
                    program_fn(GroupContext(self, group)),
                    timeout=group.budget.max_wall_seconds,
                )
                group.result = output
                group.status = GroupStatus.SUCCEEDED
            except Exception as exc:
                group.error = str(exc)
                group.status = GroupStatus.FAILED
            finally:
                group.ended_at = utc_now_iso()

            immediate_reward = self.reward_model.immediate_reward(
                group=group,
                step_count=self._group_step_count.get(group_id, 0),
                write_count=self._group_write_count.get(group_id, 0),
                object_call_count=self._group_object_calls.get(group_id, 0),
            )
            group.immediate_reward = immediate_reward

            return GroupResult(
                group_id=group.group_id,
                status=group.status,
                output=group.result,
                reward=immediate_reward,
                child_group_ids=list(self._children.get(group.group_id, [])),
                error=group.error,
            )

    def _compute_total_rewards(self, group_id: str) -> float:
        child_ids = self._children.get(group_id, [])
        child_totals = [self._compute_total_rewards(child_id) for child_id in child_ids]
        group = self._groups[group_id]
        total = self.reward_model.total_reward(immediate=group.immediate_reward, child_totals=child_totals)
        group.total_reward = total
        return total

    def _append_step(self, step: StepRecord) -> None:
        self._steps.append(step)
        self._group_step_count[step.group_id] = self._group_step_count.get(step.group_id, 0) + 1

    def _reserve_object_calls(self, group: GroupSpec, *, additional: int) -> None:
        used = self._group_object_calls.get(group.group_id, 0)
        if used + additional > group.budget.max_object_calls:
            raise RuntimeError("Object call budget exceeded")
        self._group_object_calls[group.group_id] = used + additional

    async def _commit_write(self, intent: WriteIntent) -> Any:
        last_write_time = self._group_last_write_time.get(intent.group_id)
        event = await self.write_policy.commit(
            memory=self.memory,
            intent=intent,
            last_write_time=last_write_time,
        )
        if event is not None:
            self._group_write_count[intent.group_id] = self._group_write_count.get(intent.group_id, 0) + 1
            self._group_last_write_time[intent.group_id] = monotonic()
        return event

    async def _commit_internal_write(
        self,
        *,
        group: GroupSpec,
        key: str,
        object_type: MemoryObjectType,
        reason: WriteReason,
        content: Any,
    ) -> Any:
        intent = WriteIntent(
            episode_id=group.episode_id,
            group_id=group.group_id,
            key=key,
            object_type=object_type,
            reason=reason,
            content=content,
            force=True,
        )
        return await self._commit_write(intent)
