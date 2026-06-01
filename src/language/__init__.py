"""Natural-language command parsing (spec §9 Phase 6)."""

from .command_parser import RuleCommandParser, parse_command
from .schema import (
    CommandIntent,
    ConstraintSpec,
    IntentType,
    RelationSpec,
    TargetSpec,
)

__all__ = [
    "CommandIntent",
    "ConstraintSpec",
    "IntentType",
    "RelationSpec",
    "TargetSpec",
    "RuleCommandParser",
    "parse_command",
]
