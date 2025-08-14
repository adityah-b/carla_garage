from typing import Any

class BaseFormatter:
    @classmethod
    def fmt(cls, v: Any, precision: int = 2) -> str:
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(cls._fmt_float(x, precision) for x in v) + "]"
        return cls._fmt_float(v, precision)

    @staticmethod
    def _fmt_float(x: float, p: int) -> str:
        return f"{x:.{p}f}"