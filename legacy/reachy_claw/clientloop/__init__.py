"""Client-loop migration slice for clawd-reachy-mini.

Additive proof that ovs_agent's CLIENT-SIDE tool runner
(``ovs_agent.tools.runner.stream_with_tools``) can drive the
LLM <-> tool loop for a Reachy companion robot, mirroring the
``apps/voice_arm`` reference app.

This package is intentionally isolated from the existing
``reachy_claw`` app / plugins — it imports ovs_agent's framework, not
reachy's own text-only LLM stack.
"""
