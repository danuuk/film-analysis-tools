"""Study execution: plan, dispatch, checkpoint, resume — in process.

The legacy orchestrator worked but drove every stage by building argv strings and shelling
out, because each analyzer was an argparse program. That constraint is why it never became
more than a stage runner. Rule 1 (no argument parsing outside ``cli``) is what makes
in-process dispatch possible here.

Its state and resume semantics were sound and are carried forward; its subprocess layer is
not.
"""

from __future__ import annotations

__all__: list[str] = []
