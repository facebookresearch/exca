# Claim Inheritance — cut content

Append-only. Verbatim content removed from `claim-inheritance-design.md`, with a
one-line note on why.

## Open question: single-task array ids

Cut: only matters if `Opt-self-job` is adopted, which `UC-nested-chain` now puts in
doubt.

> Unconfirmed: whether a single-task `executor.batch()` yields an array-style
> (`"<array>_<task>"`) or plain job id. Confirm on-cluster before relying on
> `Opt-self-job`.

## Opt-reject-config

Cut: rejected — the config is legitimate, it should work.

> Guard at validation or first dispatch when a chain and its last resolved step both
> carry a `_concurrent` backend on the same folder. Two resource requests for one
> cache cell is a contradiction, and the outer's worker requests a second allocation
> for work it was already allocated for. `Parallel._unify_infra` sets the precedent
> by refusing mismatched backends across itself and its steps.

## Opt-prefix-split

Cut: rejected — redefines what the chain's infra covers.

> When the chain has infra and its last resolved step has its own, the chain's
> backend covers `steps[:-1]` (a distinct prefix `step_uid`) and the last step owns
> the shared cell. Honors both resource specs, no double allocation, no self-claim.
> Costs a cache cell for the intermediate.

## Opt-outer-wins

Cut: rejected — silently discards the tail's resource request, usually the expensive
one.

> Chain's backend runs everything; the last step's infra degrades to caching only.

## Opt-no-shared-identity

Cut: rejected — drops a deliberate design property.

> Give the chain its own uid segment so it no longer shares a cell with its last
> step. Kills the whole class of bugs, but duplicates the final result on disk.

## Workaround: infra-free outer chain

Cut: different topology, so it does not address `UC-tail`.

> Nest instead of flatten, leaving the outer chain infra-free:
> `Chain(steps=[Chain(steps=[Preprocess()], infra=...), Train(infra=...)])`.
> Verified working — the inner chain takes the prefix uid, the tail takes the shared
> cell, and nobody self-claims.
