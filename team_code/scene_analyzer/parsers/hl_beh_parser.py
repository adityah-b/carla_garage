import re
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class HighLevelBehaviour:
    raw_str : str
    scenario : str
    key_actors : List[Dict[str, Optional[str]]]   # changed from Optional[Dict]
    reasoning : str

class HighLevelBehaviourParser:
    _HEADER_RE = re.compile(r"^(scenario|key_actors|reasoning)\s*:\s*$",
                            re.IGNORECASE | re.MULTILINE)

    _ACTOR_BLOCK_RE = re.compile(
        r"""
        ^\s*\d+\)\s*actor\s*=\s*\{\s*
        (?P<body>.*?)
        ^\s*\}\s*,?\s*$
        """,
        re.MULTILINE | re.DOTALL | re.VERBOSE
    )

    _KV_RE = re.compile(
        r"""
        ' (?P<key>\w+) ' \s* : \s*
        (?P<val>
            None
            | -?\d+(?:\.\d+)?
            | ' [^']* '
            | "[^"]*"
            | [A-Za-z_][A-Za-z0-9_]*
        )
        """,
        re.VERBOSE
    )

    @staticmethod
    def parse(raw_str: str) -> HighLevelBehaviour:
        text = raw_str.replace("\r\n", "\n").replace("\r", "\n")

        headers = [(m.group(1).lower(), m.start(), m.end())
                   for m in HighLevelBehaviourParser._HEADER_RE.finditer(text)]
        if len(headers) < 3:
            raise ValueError("Could not parse the input string - expected scenario/key_actors/reasoning headers.")

        sections = {}
        for i, (name, _start, end) in enumerate(headers):
            next_start = headers[i + 1][1] if i + 1 < len(headers) else len(text)
            sections[name] = text[end:next_start].strip()

        scenario  = sections.get("scenario", "")
        key_block = sections.get("key_actors", "")
        reasoning = sections.get("reasoning", "")

        key_actors = HighLevelBehaviourParser._parse_key_actors(key_block)

        return HighLevelBehaviour(
            raw_str=text,
            scenario=scenario,
            key_actors=key_actors,
            reasoning=reasoning
        )

    @staticmethod
    def _parse_key_actors(key_actors_str: str) -> List[Dict[str, Optional[str]]]:
        if not key_actors_str:
            return []

        actors: List[Dict[str, Optional[str]]] = []

        for m in HighLevelBehaviourParser._ACTOR_BLOCK_RE.finditer(key_actors_str):
            body = m.group("body")

            kvs = {}
            for kv in HighLevelBehaviourParser._KV_RE.finditer(body):
                k = kv.group("key")
                v = kv.group("val").strip()
                if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                    v = v[1:-1].strip()
                kvs[k] = v

            if "id" not in kvs or "type" not in kvs:
                continue

            traffic_raw = kvs.get("traffic")
            traffic = None if traffic_raw in (None, "None") else str(traffic_raw)

            actors.append({
                "id": int(kvs["id"]),
                "type": str(kvs["type"]),
                "traffic": traffic
            })

        return actors
