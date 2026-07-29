# Archive records

One-shot campaign work is **documented, then left readable at the legacy pin** rather than
ported. These records exist so the reasoning survives without the code having to.

Each record answers four questions:

1. **What was it asking?** The hypothesis, in the terms the campaign used.
2. **How did it try to answer?** Stages, controls, objective, evidence.
3. **What did it conclude?** Quoted from the artifacts where possible, not remembered.
4. **What is worth carrying forward?** Mechanisms, cautions, or nothing.

Records state plainly which claims were **verified from artifacts** and which are **inferred
from code**, because a campaign that ended in an unresolved state is common and pretending
otherwise is the failure mode this system is being rebuilt to avoid.

Written records are named in [`ledger.toml`](../../ledger.toml); the outstanding ones are
tracked by a ratchet in `tests/migration/test_ledger.py` so an archive group cannot quietly
appear without one.
