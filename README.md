# Hoploy

Serving layer for [Hopwise](https://github.com/tail-unica/hopwise) recommendation models.

Hoploy wraps Hopwise models — which generate recommendations along real knowledge-graph paths — into a ready-to-use API service. Plugin authors define **domain-specific behaviour** by overriding a small set of hooks, while the framework handles checkpoint loading, tokenization, beam search, and API routing.

## Architecture overview

```
API request
    │
    ▼
┌──────────┐   distill()    ┌───────────┐   recommend()   ┌───────────────┐
│  Routes   │ ─────────────► │   Model   │ ───────────────► │ Beam search   │
│ /recommend│                │  wrapper  │                  │ + processors  │
│ /info     │ ◄───────────── │           │ ◄─────────────── │               │
│ /search   │   expand()     └───────────┘   raw output     └───────────────┘
└──────────┘                      │
                            config()  ──► logits / sequence processors
```

The pipeline (`Pipeline.run`) executes:

1. **`model.distill(**payload)`** — translate the API payload into Hopwise input tokens
2. **`model.config(**payload)`** — configure generation parameters (e.g. recommendation count)
3. **`processor.config(**payload)`** for each logits/sequence processor — configure restrictions
4. **`model.recommend(inputs)`** — beam-search generation (internal, not overridden)
5. **`model.expand(output)`** — transform raw model output into the API response schema

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/recommend` | Generate recommendations |
| `GET`  | `/info/{item}` | Item information lookup |
| `POST` | `/search` | Item search |
| `GET`  | `/health` | Health check |

Request/response schemas are plain Pydantic models defined in the plugin's schema module. Endpoint routing is declared in the plugin config.

## Writing a plugin

A plugin is a directory under `plugins/` containing:

```
plugins/my_domain/
    __init__.py          # entry point (imports trigger registration)
    model.py             # model wrapper with distill / config / expand
    processors.py        # logits & sequence processor overrides
    my_domain_schema.py  # Pydantic request/response schemas
    config.yaml          # plugin-specific configuration
```

### Plugin hooks

Plugin authors interact with the framework only through these hooks:

#### Model (`DefaultHopwiseWrapper` subclass)

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `distill(**payload)` | Convert API payload to Hopwise token sequences | Dataset IDs (e.g. item names, entity names) | `list[str]` of token sequences |
| `config(**payload)` | Set generation parameters from the request | API payload | `self` |
| `expand(values)` | Convert raw `(scores, recommendations, explanations)` to API response | Hopwise IDs | `dict` matching the response schema |

Methods you should **NOT** override: `recommend`, `encode`, `decode`.

#### Logits processor (`DefaultHopwiseLogitsProcessor` subclass)

Override `config(**payload)` to translate the API payload into constraints:

```python
def config(self, **payload):
    prev = payload.get("previous_recommendations")
    if prev:
        token_ids = id2tokenizer_token(self.dataset, prev, "item")
        self.set_previous_recommendations(token_ids)
    return self
```

Available methods to call from `config()`:
- `set_previous_recommendations(token_ids)` — mask already-recommended items
- `set_restrictions(hard_restrictions=..., soft_restrictions=...)` — entity/item restrictions
- `clear_restrictions()` — reset all restrictions

#### Sequence processor (`DefaultHopwiseSequenceScorePostProcessor` subclass)

Override `config(**payload)` to adjust scoring/filtering parameters.

### Utilities

The `hoploy.core.utils` module provides helpers so plugins never need to touch low-level tokenizer internals:

| Function | Description |
|----------|-------------|
| `hopwise_encode(dataset, value, token_type)` | Dataset ID → Hopwise token string (e.g. `"place_42"` → `"I7"`) |
| `hopwise_decode(dataset, token, real_token=False)` | Hopwise token string → dataset ID or human-readable name |
| `id2tokenizer_token(dataset, ids, token_type)` | Dataset IDs → tokenizer token IDs (for processor constraints) |

`token_type` is one of `"item"`, `"entity"`, `"relation"`, `"user"` (for `id2tokenizer_token`) or the corresponding `PathLanguageModelingTokenType.*.token` constant.

### Registration

Components are registered via decorators from `hoploy.core.registry`:

```python
from hoploy.core.registry import Wrapper, LogitsProcessor, SequenceProcessor

@Wrapper("my_model")
class MyWrapper(DefaultHopwiseWrapper): ...

@LogitsProcessor("my_logits_processor")
class MyLogitsProcessor(DefaultHopwiseLogitsProcessor): ...

@SequenceProcessor("my_sequence_processor")
class MySequenceProcessor(DefaultHopwiseSequenceScorePostProcessor): ...
```

Request/response schemas use `@Request` / `@Response` decorators to declare which endpoint they belong to:

```python
from hoploy.core.registry import Request, Response

@Request("recommend")
class MyRequest(BaseModel):
    user_id: str

@Response("recommend")
class MyResponse(BaseModel):
    recommendations: list[str]
```

### Configuration

Reference your registered components in `plugins/my_domain/config.yaml`:

```yaml
model:
  my_model_key:
    name: my_model
    device: "cuda"
    hopwise_checkpoint_file: "checkpoint/my_domain/..."
    dataset: "dataset/my_domain"

logits_processors:
  my_processor:
    name: my_logits_processor

sequence_processor:
  my_seq_processor:
    name: my_sequence_processor
```

And declare the plugin in `configs/default.yaml`:

```yaml
plugin:
  my_domain:
    name: my_domain
    path: "plugins/my_domain"
    schema:
      module: my_domain_schema
      get:
        info: info        # endpoint: pipeline handler method
      post:
        recommend: run
        search: search
```

## Quick start

```bash
pip install -e .
uvicorn hoploy.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up
```

API docs are available at `http://localhost:8000/docs`.

## License

See [LICENSE](LICENSE).
