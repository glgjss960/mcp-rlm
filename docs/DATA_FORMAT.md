# MCP-RLM Data Format

## 1) Trace tables

### `episodes.jsonl`

- `episode_id`: string
- `root_group_id`: string
- `goal`: string
- `started_at`: ISO timestamp
- `ended_at`: ISO timestamp
- `success`: bool
- `root_output`: object
- `num_groups`: int
- `num_steps`: int
- `num_memory_events`: int

### `groups.jsonl`

- `group_id`: string
- `episode_id`: string
- `parent_group_id`: string or null
- `depth`: int
- `goal`: string
- `program`: string
- `status`: enum
- `budget`: object
- `input_payload`: object
- `result`: object
- `immediate_reward`: float
- `total_reward`: float
- `started_at`: ISO timestamp
- `ended_at`: ISO timestamp
- `error`: string or null

### `steps.jsonl`

- `step_id`: string
- `episode_id`: string
- `group_id`: string
- `step_index`: int
- `action_type`: enum
- `action_name`: string
- `payload`: object
- `result`: object
- `ok`: bool
- `started_at`: ISO timestamp
- `ended_at`: ISO timestamp
- `error`: string or null

### `memory_events.jsonl`

- `event_id`: string
- `episode_id`: string
- `group_id`: string
- `key`: string
- `object_type`: enum
- `reason`: enum
- `content`: object
- `confidence`: float
- `evidence_refs`: list[string]
- `parent_event_ids`: list[string]
- `version`: int
- `timestamp`: ISO timestamp

## 2) Cold-start training rows

### `cold_start_turns.jsonl`

- `episode_id`, `group_id`, `step_id`, `step_index`
- `history`: list of prior actions in same group
- `input_payload`: current action input
- `target_action_type`, `target_action_name`
- `target_result`
- `ok`

## 3) Agentic RL rows

### `agentic_rl.jsonl`

- `episode_id`
- `goal`
- `root_group_id`
- `success`
- `root_output`
- `groups`
- `steps`
- `memory_events`

## 4) Infra adapters

### `verl_warm_start.jsonl`

- `prompt`: serialized history + input
- `response`: serialized target action/result
- `metadata`: identifiers

### `openrlhf_episodes.jsonl`

- `query`: episode goal
- `response`: root output
- `reward`: root total reward
- `trajectory`: full episode structure
