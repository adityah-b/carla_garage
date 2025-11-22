import re
import ast
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Union, Literal


class Action(str, Enum):
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    MAINTAIN_SPEED = "maintain_speed"
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"
    BRAKE = "brake"
    WAIT_FOR = "wait_for"

    @staticmethod
    def from_token(token: str) -> "Action":
        t = token.strip().lower().replace("-", "_")
        t = re.sub(r"\s+", "_", t)
        try:
            return Action(t)
        except ValueError as e:
            raise ValueError(f"Unknown action: '{token}'") from e


# ---- Parameter payloads -----------------------------------------------------

@dataclass(frozen=True)
class LongitudinalParams:
    # NOTE: spd is strictly numeric (no macros)
    spd: Optional[float] = None
    t_head: Optional[float] = None
    f_dist: Optional[float] = None

@dataclass(frozen=True)
class LateralParams:
    gap_time: Optional[float] = None
    gap_dist: Optional[float] = None

@dataclass(frozen=True)
class WaitCond:
    spd: Optional[Dict[str, Any]] = None
    time: Optional[Dict[str, float]] = None
    tl_state: Optional[Dict[str, str]] = None
    actor_rel: Optional[Dict[str, Any]] = None
    gap_ok: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class WaitParams:
    # Only the mode ('all' or 'any') and a single condition list
    mode: Literal["all", "any"]
    conds: List[WaitCond]
    hold_s: float = 0.3
    timeout_s: float = 3.0
    on_timeout: str = "continue"

# Union of the three param block types (or None for BRAKE)
ParamBlock = Optional[Union[LongitudinalParams, LateralParams, WaitParams]]

@dataclass(frozen=True)
class Plan:
    actions: List[Action]         # High-level action sequence
    params: List[ParamBlock]      # Per-step param blocks aligned with actions

@dataclass(frozen=True)
class EgoPlan:
    raw_str: str
    plan: Plan
    reasoning: str

class EgoPlanParser:
    _HEADER_RE = re.compile(r"^(plan|low_level_actions|reasoning)\s*:\s*$",
                            re.IGNORECASE | re.MULTILINE)

    # Action header like: "1) decelerate:"
    _ACTION_HEADER_RE = re.compile(r"^\s*\d+\)\s*(?P<action_name>\w+)\s*:\s*$",
                                   re.MULTILINE)

    # ADD this new regex (finds the start of a param block right before its '{')
    _PARAM_KIND_RE = re.compile(
        r"(longitudinal_params|lateral_params|wait_params)\s*=\s*\{",
        re.DOTALL
    )

    # Simple key: value pairs for non-nested blocks (longitudinal/lateral)
    _KV_RE = re.compile(r"'(?P<key>\w+)'\s*:\s*(?P<val>[^,}]+)")

    @staticmethod
    def parse(raw_str: str) -> EgoPlan:
        text = raw_str.replace("\r\n", "\n").replace("\r", "\n")

        # Slice sections
        headers = [(m.group(1).lower(), m.start(), m.end())
                   for m in EgoPlanParser._HEADER_RE.finditer(text)]
        if len(headers) < 3:
            raise ValueError("Expected plan/low_level_actions/reasoning headers.")

        sections: Dict[str, str] = {}
        for i, (name, _s, e) in enumerate(headers):
            next_s = headers[i+1][1] if i+1 < len(headers) else len(text)
            sections[name] = text[e:next_s].strip()

        plan_str  = sections.get("plan", "")
        actions_s = sections.get("low_level_actions", "")
        reasoning = sections.get("reasoning", "")

        # Parse the CSV plan list (LLM summary)
        plan_actions_from_csv = EgoPlanParser._parse_plan_actions(plan_str)

        # Parse detailed low_level_actions into (actions, params) aligned lists
        actions_from_blocks, params_from_blocks = EgoPlanParser._parse_low_level_actions(actions_s)

        # Prefer block headers if mismatch; otherwise use the CSV
        if len(plan_actions_from_csv) == len(actions_from_blocks) and \
           all(a1 == a2 for a1, a2 in zip(plan_actions_from_csv, actions_from_blocks)):
            actions = plan_actions_from_csv
        else:
            actions = actions_from_blocks  # trust explicit block headers

        plan = Plan(actions=actions, params=params_from_blocks)
        return EgoPlan(raw_str=text, plan=plan, reasoning=reasoning)

    # ---- Section parsers ----------------------------------------------------

    @staticmethod
    def _parse_plan_actions(plan_str: str) -> List[Action]:
        if not plan_str:
            return []
        tokens = [p.strip() for p in plan_str.split(",") if p.strip()]
        return [Action.from_token(tok) for tok in tokens]

    # REPLACE the entire _parse_low_level_actions with this version
    @staticmethod
    def _parse_low_level_actions(actions_str: str) -> Tuple[List[Action], List[ParamBlock]]:
        if not actions_str:
            return [], []

        actions: List[Action] = []
        params: List[ParamBlock] = []

        headers = list(EgoPlanParser._ACTION_HEADER_RE.finditer(actions_str))
        for i, h in enumerate(headers):
            action_name = h.group("action_name")
            action = Action.from_token(action_name)
            start = h.end()
            end = headers[i+1].start() if i+1 < len(headers) else len(actions_str)
            block_text = actions_str[start:end]

            param_block: ParamBlock = None  # BRAKE will remain None

            # Scan for a param block start and extract the balanced body
            pos = 0
            while True:
                m = EgoPlanParser._PARAM_KIND_RE.search(block_text, pos)
                if not m:
                    break
                kind = m.group(1)
                open_idx = m.end() - 1  # points to '{'
                inner_body, after = EgoPlanParser._extract_braced_body(block_text, open_idx)

                if kind == "longitudinal_params":
                    kvs = EgoPlanParser._parse_body_kvs(inner_body)
                    param_block = LongitudinalParams(
                        spd=EgoPlanParser._coerce_float(kvs.get("spd")) if "spd" in kvs else None,
                        t_head=EgoPlanParser._coerce_float(kvs.get("t_head")),
                        f_dist=EgoPlanParser._coerce_float(kvs.get("f_dist")),
                    )
                elif kind == "lateral_params":
                    kvs = EgoPlanParser._parse_body_kvs(inner_body)
                    param_block = LateralParams(
                        gap_time=EgoPlanParser._coerce_float(kvs.get("gap_time")),
                        gap_dist=EgoPlanParser._coerce_float(kvs.get("gap_dist")),
                    )
                elif kind == "wait_params":
                    param_block = EgoPlanParser._parse_wait_params_block(inner_body)

                pos = after  # continue scanning in case an action block ever contains multiple assignments

            actions.append(action)
            params.append(param_block)

        return actions, params

    # ADD this helper inside EgoPlanParser
    @staticmethod
    def _extract_braced_body(src: str, open_brace_idx: int) -> Tuple[str, int]:
        """
        Given src and the index of '{', return (inner_text, end_idx_after_closing_brace).
        Handles nested braces and quoted strings with escapes.
        """
        if open_brace_idx >= len(src) or src[open_brace_idx] != "{":
            raise ValueError("Internal parser error: expected '{'.")
        i = open_brace_idx + 1
        start = i
        depth = 0
        while i < len(src):
            ch = src[i]
            if ch in ("'", '"'):
                q = ch
                i += 1
                # skip quoted string
                while i < len(src):
                    if src[i] == "\\":
                        i += 2  # skip escaped char
                        continue
                    if src[i] == q:
                        i += 1
                        break
                    i += 1
                continue
            if ch == "{":
                depth += 1
                i += 1
                continue
            if ch == "}":
                if depth == 0:
                    inner = src[start:i]
                    return inner, i + 1
                depth -= 1
                i += 1
                continue
            i += 1
        raise ValueError("Unbalanced braces in parameter block.")

    @staticmethod
    def _parse_body_kvs(body: str) -> Dict[str, str]:
        """
        Parse 'key': value pairs inside {...} for flat blocks.
        (Works for longitudinal/lateral where values are simple.)
        """
        out: Dict[str, str] = {}
        for m in EgoPlanParser._KV_RE.finditer(body):
            key = m.group("key")
            val = m.group("val").strip()
            # strip surrounding quotes if present
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1].strip()
            out[key] = val
        return out

    # ---- Coercion helpers ---------------------------------------------------
    @staticmethod
    def _coerce_float(v: Optional[str]) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # ---- wait_params parsing ------------------------------------------------

    @staticmethod
    def _parse_wait_params_block(body: str) -> WaitParams:
        """
        The wait_params body is a valid Python dict literal with nested lists/dicts.
        We safely parse it via ast.literal_eval and then validate/convert to dataclasses.
        Output WaitParams(mode='all'|'any', conds=[...], hold_s, timeout_s, on_timeout).
        """
        literal = "{" + body + "}"
        try:
            data = ast.literal_eval(literal)
        except Exception as e:
            raise ValueError(f"Failed to parse wait_params: {e}")

        if not isinstance(data, dict):
            raise ValueError("wait_params must be a dict")

        all_list = data.get("all")
        any_list = data.get("any")
        if bool(all_list) == bool(any_list):
            # Either both present or both missing -> invalid per spec
            raise ValueError("wait_params must contain exactly one of 'all' or 'any'")

        mode: Literal["all", "any"]
        raw_conds: List[Any]
        if all_list is not None:
            mode = "all"
            raw_conds = all_list
        else:
            mode = "any"
            raw_conds = any_list

        hold_s = float(data.get("hold_s", 0.3))
        timeout_s = float(data.get("timeout_s", 3.0))
        on_timeout = str(data.get("on_timeout", "continue"))

        def to_wait_cond(obj: Dict[str, Any]) -> WaitCond:
            if not isinstance(obj, dict) or len(obj) != 1:
                raise ValueError(f"Invalid wait condition: {obj}")
            (k, payload), = obj.items()
            if k not in {"spd", "time", "tl_state", "actor_rel", "gap_ok"}:
                raise ValueError(f"Unknown wait condition key: {k}")

            if k == "spd":
                # expect {'op': '>=', 'value': <number>}
                return WaitCond(spd={
                    "op": str(payload.get("op")),
                    "value": float(payload.get("value"))
                })
            if k == "time":
                return WaitCond(time={"seconds": float(payload.get("seconds"))})
            if k == "tl_state":
                return WaitCond(tl_state={"is": str(payload.get("is"))})
            if k == "actor_rel":
                # free-form payload; keep as-is but ensure it's a dict copy
                return WaitCond(actor_rel=dict(payload))
            if k == "gap_ok":
                return WaitCond(gap_ok={
                    "lane": str(payload.get("lane")),
                    "gap_time": float(payload.get("gap_time")),
                    "gap_dist": float(payload.get("gap_dist")),
                })
            return WaitCond()

        conds = [to_wait_cond(c) for c in (raw_conds or [])]

        # Guardrail; allow but you may clamp later
        if hold_s > timeout_s:
            pass

        return WaitParams(
            mode=mode,
            conds=conds,
            hold_s=hold_s,
            timeout_s=timeout_s,
            on_timeout=on_timeout
        )