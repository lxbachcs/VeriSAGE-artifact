# Known limitations

Status at publication, 5 November 2026. This package preserves the experimental
prototype and stored results used by the paper. The following findings remain
open. The smoke tests establish agreement on their tested synthetic cases.

## C4 does not propagate declared dependencies

`Service.taint` declares input-to-output dependencies, but `verify` and
`verify_all` do not propagate them. An output without an explicit field label
is treated as public, even when it depends on a secret input. Neither API
therefore implements the dependency-reachability guarantee of the formal design.

Run this example from the repository root:

```python
from core import Service, Goal, verify, verify_all
copy = Service("copySecret", "copy", {"secret"}, {"derived"},
               lat_ub=0, lat_mean=0, taint={"derived": {"secret"}})
sink = Service("publicLog", "log", {"derived"}, {"done"},
               lat_ub=0, lat_mean=0, sink_field="derived", sink_clearance="public")
goal = Goal({"secret"}, {"done"}, set(), [], 100,
            init_level={"secret": "secret"})
print(verify([copy, sink], goal))
print(verify_all([copy, sink], goal))
```

The current code returns `ok=True` and no witnesses. The declared secret flow
should be rejected. No service is executed by this example. General privacy
enforcement requires correcting both verification paths and validating them
against independently labelled dependency graphs.

## Implementation and simulation scope

- C1 checks availability of field names. It does not implement JSON Schema subtyping.
- C2 checks propositional facts. Full SMT discharge is timed separately.
- Policy guards are Boolean values. The prototype does not execute Rego policies.
- The dependency false-positive experiment in `exp6_deptaint.py` uses a separate
  synthetic model. Its Boolean rejection simulation is not a call to `core.verify`.
- QWS response times form a bootstrap population for latency simulations. They are
  not joint invocation traces of the evaluated workflows.
- The reduced C1+C2 ablation is internal. It is not an evaluation of Clover.

## Report corrections and remaining implementation work

The revised extended report now distinguishes the full formal design from the
implemented checks. It corrects the repair bound to allow a final successful
proposal after at most N failing proposals, adds the nonnegative path-slack
condition to the DAG Hoeffding expression, and identifies the C1+C2 experiment
as an internal ablation rather than an evaluation of Clover. It also labels the
dependency experiments as separate synthetic simulations.

These are documentation and mathematical-statement corrections. The prototype's
C4 propagation defect remains unresolved, and no new experiments or verifier
algorithm changes are claimed. Matching stored numbers does not resolve it.
