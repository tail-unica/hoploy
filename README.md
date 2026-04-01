# Hoploy

Serving layer for [Hopwise](https://github.com/tail-unica/hopwise) recommendation models.

Hoploy wraps Hopwise models — which generate recommendations along real knowledge-graph paths — into a deployable REST API service. Plugin authors define domain-specific behaviour by implementing a small set of hooks; the framework handles checkpoint loading, tokenization, beam-search generation, logits/sequence processing, and API routing transparently.

## Intended use

Hoploy is designed as a reusable framework. A practitioner installs it as a dependency, writes a plugin directory for their domain, and launches the service with three lines:

```python
# main.py
from hoploy.core.config import Config
from hoploy.core.pipeline import Pipe
from hoploy.core.factory import factory

config   = Config("plugins/my_domain")
pipeline = Pipe(config)
app      = factory(pipeline, config)
```

`Config` loads and merges `hoploy`'s built-in defaults with the plugin configuration. `Pipe` instantiates the wrapper and all processors declared in the plugin. `factory` builds a FastAPI application with endpoints wired to the plugin schema, then exposes it via Uvicorn.

No framework internals need to be modified. Everything domain-specific lives entirely inside the plugin directory.

---

## Architecture overview

```
API request
    │
    ▼
┌──────────┐   distill()    ┌───────────┐   recommend()   ┌───────────────┐
│  Routes   │ ─────────────► │  Wrapper  │ ───────────────► │ Beam search   │
│ /recommend│                │           │                  │ + processors  │
│ /info     │ ◄───────────── │           │ ◄─────────────── │               │
│ /search   │   expand()     └───────────┘   raw output     └───────────────┘
└──────────┘                      │
                             handle()  ──► logits / sequence processors
```

The pipeline executes the following steps on each request:

1. `wrapper.distill(request)` — translate the API payload to Hopwise input token sequences.
2. `wrapper.handle(request)` — configure generation parameters for the current request.
3. `processor.handle(request)` for each logits and sequence processor — configure constraints.
4. `wrapper.recommend(inputs)` — beam-search generation (internal; not overrideable).
5. `wrapper.expand(output, request)` — map raw model output to the API response schema.

---

## Plugin structure

A plugin is a self-contained directory that is passed to `Config`:

```
plugins/my_domain/
    __init__.py           # entry point; imports trigger component registration
    model.py              # wrapper subclass: distill / handle / expand
    processors.py         # logits and sequence processor subclasses
    my_domain_schema.py   # Pydantic request/response models
    config.yaml           # plugin-specific configuration
```

All components are registered with the framework via decorators (`@Wrapper`, `@LogitsProcessor`, `@SequenceProcessor`) and referenced by name in `config.yaml`.

---

## Configuration levels

Plugin behaviour is controlled at three distinct levels. Moving from level 1 to level 3 provides progressively more control over the generation process.

### Level 1 — Wrapper and processors with `handle()`

This is the standard integration path. The plugin author implements `handle()` on the wrapper and on each processor to translate the incoming API request into generation state before beam search runs.

**Wrapper** (`DefaultHopwiseWrapper` subclass)

The wrapper has three required hooks:

| Method | Responsibility |
|---|---|
| `distill(request)` | Translate API payload into Hopwise token sequences passed to generation. |
| `handle(request)` | Configure generation parameters (e.g. number of recommendations, diversity). |
| `expand(output, request)` | Map the raw `(scores, item_ids, explanations)` tuple to the API response dict. |

Methods `recommend`, `encode`, and `decode` are implemented by the framework and must not be overridden.

```python
from hoploy.components import DefaultHopwiseWrapper
from hoploy.registry import Wrapper
from hoploy.core.utils import hopwise_encode

@Wrapper
class MyWrapper(DefaultHopwiseWrapper):

    def distill(self, request):
        return [
            "[BOS] " + hopwise_encode(self.dataset, item_id, "I")
            for item_id in request.input
        ]

    def handle(self, request):
        self.recommendation_count = request.n
        return self

    def expand(self, output, request):
        scores, item_ids, explanations = output
        return {"recommendations": item_ids, "scores": scores, "explanations": explanations}
```

**Logits processor** (`DefaultHopwiseLogitsProcessor` subclass)

The logits processor controls which tokens are allowed or scored at each generation step. `handle()` sets up restrictions from the request before generation begins. A second optional hook, `score_adjustment()`, can apply fine-grained score deltas at generation time.

```python
from hoploy.components import DefaultHopwiseLogitsProcessor
from hoploy.registry import LogitsProcessor
from hoploy.core.utils import id2tokenizer_token

@LogitsProcessor
class MyLogitsProcessor(DefaultHopwiseLogitsProcessor):

    def handle(self, request):
        if request.previous_recommendations:
            token_ids = id2tokenizer_token(
                self.dataset, request.previous_recommendations, "item"
            )
            self.set_previous_recommendations(token_ids)
        return self
```

Available methods callable from `handle()`:

| Method | Effect |
|---|---|
| `set_previous_recommendations(token_ids)` | Mask items the user has already seen. |
| `set_restrictions(hard_restrictions, soft_restrictions)` | Ban or penalise specific entities or items. |
| `clear_restrictions()` | Reset all restrictions. |

The optional `score_adjustment()` hook receives Hopwise IDs rather than tokenizer internals and returns score deltas:

```python
def score_adjustment(self, hopwise_current, hopwise_candidates):
    # hopwise_current and hopwise_candidates use "type:value" format,
    # e.g. "entity:SensoryFeature.NOISE.2.3" or "item:55"
    return {"item:55": float("-inf")}  # hard-ban item 55
```

**Sequence processor** (`DefaultHopwiseSequenceScorePostProcessor` subclass)

The sequence processor re-scores or filters complete generated sequences. `handle()` configures its parameters from the request.

```python
from hoploy.components import DefaultHopwiseSequenceScorePostProcessor
from hoploy.registry import SequenceProcessor

@SequenceProcessor
class MySequenceProcessor(DefaultHopwiseSequenceScorePostProcessor):

    def handle(self, request):
        # configure path boosting, filtering, etc.
        return self
```

---

### Level 2 — Custom generation with `ForcedLogitsProcessor` and `ForcedSequenceScorePostProcessor`

When the default beam-search constraints are insufficient, the framework provides two specialised base classes that allow the plugin to declare explicit relation-path patterns to force or boost during generation.

`ForcedLogitsProcessor` restricts the token vocabulary at each step so that generation can only follow declared relation sequences. `ForcedSequenceScorePostProcessor` re-scores complete paths by pattern matching the generated relation sequence against the declared patterns and applying multiplier boosts.

These classes still expose `handle()` as the request-wiring hook; the path-pattern logic is configured via `force_paths` in `config.yaml` and applied automatically.

---

### Level 3 — Full override with `process_scores_rec`

For advanced cases that cannot be expressed through score adjustments or path patterns, any logits processor subclass may override `process_scores_rec(scores, input_ids, ...)` directly. This is a low-level escape hatch that exposes the raw score tensor and should only be used when levels 1 and 2 are insufficient.

---

## Endpoint routing

Endpoints are declared in the plugin `config.yaml` under the `plugin.schema` key:

```yaml
plugin:
  schema:
    module: my_domain_schema
    get:
      /info/{item}: my_wrapper.info
    post:
      /recommend: run
      /search:    my_wrapper.search
```

The `run` handler invokes the full pipeline. All other values reference methods on the wrapper instance. The framework resolves request and response schemas automatically by convention (`<Name>Request` / `<Name>Response`) or via explicit class names.

---

## Request and response schemas with Pydantic

Schemas are plain [Pydantic](https://docs.pydantic.dev/) models. Pydantic is a data validation library for Python that uses type annotations and class attributes to describe the expected shape of incoming and outgoing data. At request time, the framework parses the raw JSON body through the request model, validates all types and constraints, and raises a `422 Unprocessable Entity` response automatically if validation fails. At response time, only fields declared in the response model are serialised.

Plugin authors define their schemas in the module referenced by `plugin.schema.module` in `config.yaml`. No registration decorators are required; the framework discovers schemas by class-name convention.

```python
# my_domain_schema.py
from pydantic import BaseModel, Field

class RecommendRequest(BaseModel):
    user_id: str = Field(description="Unique user identifier")
    n: int      = Field(default=5, ge=1, le=20)

class RecommendResponse(BaseModel):
    recommendations: list[str]
    scores: list[float]
    explanations: list[str]
```

Richer models — with enumerations, nested structures, cross-field validators via `@model_validator`, and `Field` constraints such as `ge`, `le`, `min_length` — are fully supported. See `plugins/autism/autism_schema.py` for a complete example using generic sensory-feature sets.

**Connecting the schema to Hopwise**

One of the responsibilities of `distill()` is to bridge the domain schema (where identifiers come from the request model) to the internal Hopwise token space. The translation utilities described below are the intended interface for this bridge.

---

## Hopwise ID typing

Hopwise encodes every element of the knowledge graph as a typed token string. Plugin authors encounter these tokens when implementing `distill()`, `expand()`, and processor hooks. The token format is a single uppercase letter followed by an integer index:

| Prefix | Type | Example |
|---|---|---|
| `I` | Item | `I42` |
| `E` | Entity (KG node) | `E17` |
| `R` | Relation (KG edge type) | `R3` |
| `U` | User | `U8` |

These indices are position-based and specific to the dataset loaded at runtime. They do not correspond directly to the original dataset identifiers (such as a CSV row number or an external ID string). All translation between dataset identifiers and Hopwise tokens must go through the utility functions in `hoploy.core.utils`.

| Function | Description |
|---|---|
| `hopwise_encode(dataset, value, token_type)` | Dataset ID string to Hopwise token (e.g. `"42"` to `"I7"`). |
| `hopwise_decode(dataset, token, real_token=False)` | Hopwise token to dataset ID; pass `real_token=True` for the human-readable label. |
| `id2tokenizer_token(dataset, ids, token_type)` | Batch of dataset ID strings to tokenizer-internal integer IDs; use this to feed processor constraint methods. |

`token_type` accepts the `PathLanguageModelingTokenType.*.token` constants from Hopwise (`"I"`, `"E"`, `"R"`, `"U"`).

In `score_adjustment()` and in generated sequences passed to `expand()`, Hopwise IDs are represented as `"type:value"` strings (e.g. `"entity:SensoryFeature.NOISE.2.3"` or `"item:55"`). These are the decoded, human-readable forms of the same token space and do not require further translation.

---

## Default endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/recommend` | Run the full recommendation pipeline. |
| `GET`  | `/info/{item}` | Item information lookup (delegated to `wrapper.info`). |
| `POST` | `/search` | Item search (delegated to `wrapper.search`). |
| `GET`  | `/health` | Liveness check. |

Additional endpoints are registered automatically for every route declared in the plugin `config.yaml`.

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
