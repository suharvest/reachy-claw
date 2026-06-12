"""Parse SKILL.md into sections for progressive tool loading.

SKILL.md format:
    # Title
    Description
    ## Container (e.g. "Tools")
    ### Group Name
    Group description.
    #### `tool_name`
    Tool description.
    - `param` (type): description

Each ### group becomes a skill section that can be loaded on demand.
The LLM starts with a lightweight index and calls load_skill(name)
to activate a group's tools.
Sections (## headings) without any #### tool definitions are skipped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_H2_RE = re.compile(r"^##\s+(.+)$")
_H3_RE = re.compile(r"^###\s+(.+)$")
_H4_RE = re.compile(r"^####\s+`(\w+)`$")
_PARAM_RE = re.compile(r"^-\s+`(\w+)`\s+\((\w+(?:,\s*\w+)?)\):\s*(.+)$")

_TYPE_MAP = {"float": "number", "int": "integer", "integer": "integer",
             "bool": "boolean", "boolean": "boolean"}


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, dict] = field(default_factory=dict)

    def to_ollama(self) -> dict:
        """Convert to Ollama /api/chat tools format."""
        props = {
            k: {"type": v["type"], "description": v["description"]}
            for k, v in self.parameters.items()
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": [],
                },
            },
        }


@dataclass
class SkillSection:
    key: str           # slug for load_skill(), e.g. "settings"
    heading: str       # original ### heading
    description: str   # brief description for index
    tools: list[ToolDef] = field(default_factory=list)
    api_base: str = "" # e.g. "http://localhost:3260" — tool calls POST here

    def index_line(self) -> str:
        tool_names = ", ".join(t.name for t in self.tools)
        return f"- {self.key}: {self.description} [{tool_names}]"

    def to_ollama_tools(self) -> list[dict]:
        return [t.to_ollama() for t in self.tools]


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_skill_md(content: str) -> list[SkillSection]:
    """Parse SKILL.md content into skill sections (### groups with #### tools)."""
    sections: list[SkillSection] = []
    title_desc = ""
    cur_group: SkillSection | None = None
    cur_tool: ToolDef | None = None
    in_code_block = False

    for line in content.split("\n"):
        # Track code blocks to skip content
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # ## heading: reset group (new top-level container)
        if _H2_RE.match(line):
            if cur_tool and cur_group:
                cur_group.tools.append(cur_tool)
                cur_tool = None
            if cur_group:
                sections.append(cur_group)
                cur_group = None
            continue

        # ### heading: new skill group
        m = _H3_RE.match(line)
        if m:
            if cur_tool and cur_group:
                cur_group.tools.append(cur_tool)
                cur_tool = None
            if cur_group:
                sections.append(cur_group)
            cur_group = SkillSection(
                key=_slugify(m.group(1)), heading=m.group(1), description="",
            )
            continue

        # #### `tool_name`: tool definition
        m = _H4_RE.match(line)
        if m and cur_group:
            if cur_tool:
                cur_group.tools.append(cur_tool)
            cur_tool = ToolDef(name=m.group(1), description="")
            continue

        # Parameter line
        m = _PARAM_RE.match(line)
        if m and cur_tool:
            pname, ptype, pdesc = m.groups()
            cur_tool.parameters[pname] = {
                "type": _TYPE_MAP.get(ptype.strip(), "string"),
                "description": pdesc,
            }
            continue

        # Plain text — description
        stripped = line.strip()
        if (stripped and not stripped.startswith(("#", "-", "|", "```", "*"))
                and not stripped.startswith("[")):
            if cur_tool and not cur_tool.description:
                cur_tool.description = stripped
            elif cur_group and not cur_tool and not cur_group.description:
                cur_group.description = stripped
            elif not cur_group and not title_desc:
                title_desc = stripped

    # Flush
    if cur_tool and cur_group:
        cur_group.tools.append(cur_tool)
    if cur_group:
        sections.append(cur_group)

    # Filter sections with no tools, fill missing descriptions
    result = [s for s in sections if s.tools]
    for s in result:
        if not s.description:
            s.description = s.heading
    return result


def load_skill_file(path: str | Path) -> list[SkillSection]:
    """Load and parse a single SKILL.md file. Returns empty list if not found."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Skill file not found: {p}")
        return []
    sections = parse_skill_md(p.read_text())
    logger.info(
        "Loaded %d skill sections from %s: %s",
        len(sections), p.name,
        ", ".join(f"{s.key}({len(s.tools)} tools)" for s in sections),
    )
    return sections


def _load_skill_config(skill_dir: Path) -> dict:
    """Load config.yaml from a skill directory. Returns empty dict if not found."""
    config_file = skill_dir / "config.yaml"
    if not config_file.is_file():
        return {}
    try:
        import yaml
        return yaml.safe_load(config_file.read_text()) or {}
    except Exception as e:
        logger.warning(f"Failed to parse {config_file}: {e}")
        return {}


def _check_health(url: str, timeout: float = 3.0) -> bool:
    """Synchronous health check — returns True if the URL is reachable."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def load_skill_dir(skill_dir: str | Path) -> list[SkillSection]:
    """Scan a directory for skill subdirs, each containing a SKILL.md.

    Expected structure:
        skills/
          reachy-robot/
            SKILL.md              # always loaded
          sensecraft/
            SKILL.md
            config.yaml           # health_check: http://...

    Skills with a config.yaml health_check are only loaded if the URL is
    reachable. Skills without config.yaml are always loaded.
    """
    d = Path(skill_dir)
    if not d.is_dir():
        logger.debug(f"Skill directory not found: {d}")
        return []
    all_sections: list[SkillSection] = []
    seen_keys: set[str] = set()
    for sub in sorted(d.iterdir()):
        skill_file = sub / "SKILL.md"
        if not skill_file.is_file():
            continue

        # Health check
        config = _load_skill_config(sub)
        health_url = config.get("health_check", "")
        if health_url:
            timeout = config.get("timeout", 3.0)
            if not _check_health(health_url, timeout):
                logger.info("Skill '%s' skipped: %s unreachable", sub.name, health_url)
                continue
            logger.info("Skill '%s' health check passed: %s", sub.name, health_url)

        sections = parse_skill_md(skill_file.read_text())
        # Attach api_base from config to all sections in this skill
        api_base = config.get("api_base", "").rstrip("/")
        for s in sections:
            s.api_base = api_base
        # Deduplicate keys across files
        for s in sections:
            if s.key in seen_keys:
                suffix = 2
                while f"{s.key}_{suffix}" in seen_keys:
                    suffix += 1
                s.key = f"{s.key}_{suffix}"
            seen_keys.add(s.key)
        all_sections.extend(sections)
        logger.info(
            "Loaded %s: %s",
            skill_file,
            ", ".join(f"{s.key}({len(s.tools)} tools)" for s in sections),
        )
    if all_sections:
        logger.info(
            "Total: %d skill sections, %d tools from %s",
            len(all_sections),
            sum(len(s.tools) for s in all_sections),
            d,
        )
    return all_sections


def build_skill_index(sections: list[SkillSection]) -> str:
    """Build the system prompt snippet listing available skills."""
    if not sections:
        return ""
    lines = [
        "You have skills that extend your abilities. "
        "Call load_skill(name) to activate a skill before using its tools.",
        "Available skills:",
    ]
    lines.extend(s.index_line() for s in sections)
    return "\n".join(lines)


# The meta-tool definition for Ollama
LOAD_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": (
            "Load a skill to gain access to its tools. "
            "You must call this before using any skill-specific tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name from the available list",
                },
            },
            "required": ["name"],
        },
    },
}
