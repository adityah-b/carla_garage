import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class EgoPlan:
    raw_str: str
    plan: List[str]
    low_level_actions: List[Dict]
    reasoning: str

class EgoPlanParser:
    _HEADER_RE = re.compile(r"^(plan|low_level_actions|reasoning)\s*:\s*$",
                            re.IGNORECASE | re.MULTILINE)

    # Action header like: "1) decelerate:"
    _ACTION_HEADER_RE = re.compile(r"^\s*\d+\)\s*(?P<action_name>\w+)\s*:\s*$",
                                   re.MULTILINE)

    # Blocks like: "longitudinal_params = { ... }" or "lateral_params = { ... }"
    _BLOCK_RE = re.compile(r"(?P<kind>longitudinal_params|lateral_params)\s*=\s*\{\s*(?P<body>.*?)\s*\}",
                           re.DOTALL)

    # Key/val pairs inside the braces; tolerant of quotes, spaces, newlines
    _KV_RE = re.compile(r"'(?P<key>\w+)'\s*:\s*(?P<val>[^,}]+)")

    @staticmethod
    def parse(raw_str: str) -> EgoPlan:
        text = raw_str.replace("\r\n", "\n").replace("\r", "\n")

        headers = [(m.group(1).lower(), m.start(), m.end())
                   for m in EgoPlanParser._HEADER_RE.finditer(text)]
        if len(headers) < 3:
            raise ValueError("Expected plan/low_level_actions/reasoning headers.")

        sections = {}
        for i, (name, _s, e) in enumerate(headers):
            next_s = headers[i+1][1] if i+1 < len(headers) else len(text)
            sections[name] = text[e:next_s].strip()

        plan_str  = sections.get("plan", "")
        actions_s = sections.get("low_level_actions", "")
        reasoning = sections.get("reasoning", "")

        plan = EgoPlanParser._parse_plan(plan_str)
        low_level_actions = EgoPlanParser._parse_low_level_actions(actions_s)

        return EgoPlan(
            raw_str=text,
            plan=plan,
            low_level_actions=low_level_actions,
            reasoning=reasoning
        )

    @staticmethod
    def _parse_plan(plan_str: str) -> List[str]:
        if not plan_str:
            return []
        # tolerate newlines and extra spaces around commas
        return [p.strip() for p in re.split(r",", plan_str) if p.strip()]

    @staticmethod
    def _parse_low_level_actions(actions_str: str) -> List[Dict]:
        if not actions_str:
            return []

        # Find each action header, then slice to next header/end
        actions: List[Dict] = []
        headers = list(EgoPlanParser._ACTION_HEADER_RE.finditer(actions_str))
        for i, h in enumerate(headers):
            name = h.group("action_name")
            start = h.end()
            end = headers[i+1].start() if i+1 < len(headers) else len(actions_str)
            block_text = actions_str[start:end]

            # Find one or more param blocks within this action chunk
            action_dict: Dict[str, Dict] = {"action_name": name}
            for m in EgoPlanParser._BLOCK_RE.finditer(block_text):
                kind = m.group("kind")
                body = m.group("body")
                kvs = EgoPlanParser._parse_body_kvs(body)

                if kind == "longitudinal_params":
                    lp = {}
                    if "spd" in kvs:    lp["spd"] = EgoPlanParser._coerce(kvs["spd"])
                    if "t_head" in kvs: lp["t_head"] = EgoPlanParser._coerce(kvs["t_head"])
                    if "f_dist" in kvs: lp["f_dist"] = EgoPlanParser._coerce(kvs["f_dist"])
                    if lp: action_dict["longitudinal_params"] = lp

                elif kind == "lateral_params":
                    lt = {}
                    if "gap_time" in kvs: lt["gap_time"] = EgoPlanParser._coerce(kvs["gap_time"])
                    if "gap_dist" in kvs: lt["gap_dist"] = EgoPlanParser._coerce(kvs["gap_dist"])
                    if lt: action_dict["lateral_params"] = lt

            actions.append(action_dict)

        return actions

    @staticmethod
    def _parse_body_kvs(body: str) -> Dict[str, str]:
        """Parse 'key': value pairs inside {...} regardless of line breaks."""
        out: Dict[str, str] = {}
        for m in EgoPlanParser._KV_RE.finditer(body):
            key = m.group("key")
            val = m.group("val").strip()
            # strip surrounding quotes if present
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1].strip()
            out[key] = val
        return out

    @staticmethod
    def _coerce(v: str) -> Union[float, str]:
        try:
            return float(v)
        except ValueError:
            return v
