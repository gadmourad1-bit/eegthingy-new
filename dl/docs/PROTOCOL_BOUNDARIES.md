# Protocol boundaries and result-joining rules

## The non-negotiable rule

Numbers may share a report, but they may be ranked or contrasted as peers only
when model, inputs, participants, folds, seeds, training recipe, selection,
reset/refit, and test-use contracts match. A result from one row below cannot
fill a missing cell in another.

The only completed five-dataset architecture leaderboard distributed with
this release is the **43-model common raw-trial recipe**. “All models” in that
context means all 43 members of the frozen common roster, not every registry
entry, historical procedure, author-recipe reference, transfer condition,
control, or experimental candidate in the repository.

## Track matrix

| Track | Scientific question | Eligibility and unit | Evidence status in this release | Comparison boundary |
|---|---|---|---|---|
| Common raw-trial recipe | How do 43 fixed architecture/configuration factories compare under one recipe? | Five opened datasets; 448 participant/fold units; five seeds; 96,320 jobs | **Complete, exactly audited, and bundled** | Rank only the 43 common rows. TCFormer is the prespecified context comparator. |
| Gauge Gate 1 | Does one frozen experimental continuation justify disjoint Gate 2? | 17 prespecified opened participants, one candidate, bound common-reference predictions | **Complete negative gate; not in common roster** | Compare only under its gate contract. Failed breadth criterion forbids promotion. |
| CHSD screens/robustness amendment | Do complex HSD representations clear frozen promotion criteria? | Separate screened and disjoint-subject subsets | **Negative development evidence documented; not registered/common** | Report as a screen, never as a common-grid row. |
| Author-recipe references | How do adapted TCFormer and FBCNet behave under training recipes closer to their publications? | Same eligible datasets but method-specific optimization; 4,480 planned records | Formal runner is included; **no completed author-recipe result is distributed or claimed by this release** | Compare the two fixed author-recipe references only inside their own table. Do not substitute these values for common TCFormer/FBCNet. |
| GeoAdapt harmonized-v2 | How do geometric architectures/procedures behave with their SPD inputs? | Binary-eligible L/B/C/P; 6,495 planned records | Runner/protocol work is included, but the harmonized-v2 grid is **unrun** and no score is distributed | Compare only matching GeoAdapt conditions and eligible datasets. |
| Local binary procedures | How do CAMEO, HemiParity, PARITY-Fuse, and ORBIT-v3 behave with paired-view/procedure-specific training? | Binary L/B/C/P; BNCI2014-001 is principled N/A; 8,780 planned records | Frozen Local Exp4 procedure artifacts exist historically; the all-dataset formal procedure grid is **unrun** | Keep one separate procedure table; historical Cho/legacy-A scores use different protocols. |
| Deterministic classical controls | How do tangent/Riemannian/EA-FBCSP controls perform without stochastic seed aliases? | Five datasets; one seedless record per participant/fold; 1,344 planned records | A separate CPU runner is included, but no completed control grid is distributed or claimed | No invented five-seed replication; compare within the control table or exact paired contexts. |
| HemiQ harmonized-v2 | How does the dataset-specific reflection-odd field behave on supplied BNCI2014-004 bipolar signals? | BNCI2014-004 only; 45 planned records | Historical development/one-time confirmation evidence exists; the harmonized-v2 grid is **unrun** and raw historical predictions are excluded from this clean bundle | N/A elsewhere. Never treat the three derivations as point electrodes. |
| Native CardinalFBMS transfer | Does a source checkpoint help on native Cho/PhysioNet target montages? | Cho S16–S52 and PhysioNet S1–S54; five conditions; 8,675 records | A prior complete artifact was independently verified; **not bundled into the common ranking** | Compare only the five matched transfer conditions. |
| Native CardinalFBC transfer | Same transfer question for predecessor FBC conditions | Same native target cohorts; four conditions; 6,940 total when complete | Only limited predecessor coverage is documented; full matched table is **incomplete** | No available-case aggregate and no substitution from common CardinalFBC. |
| CardinalSplineDualView | Does fixed geometry transport plus native/canonical fusion help target adaptation? | Native Cho/PhysioNet transfer cohorts | Prototype only; no promoted aggregate writer/result | Blocked, not zero and not a common cell. |
| Sealed confirmation | Does a frozen selected method replicate on untouched cohorts? | PhysioNet S55–S109, Lee2019, Weibo2014, Zhou2016 under a future protocol | **Not run and not authorized by this release** | Must remain absent until lawful, prespecified unsealing. |

“Not bundled” is not a performance statement. It means that this clean
common-grid result package does not supply a canonical same-protocol aggregate
for that track. Consult the named track's own immutable artifact and audit
before making any claim about completion.

## Common-grid comparison contract

The common roster is immutable. It contains 28 in-house configurations and 15
external configurations. Adding Gauge, CHSD, a procedure model, a new variant,
or an author-recipe result would change the cardinality and invalidate the
formal plan identity.

The observed equal-dataset leader is selected descriptively from this roster.
The winner-versus-TCFormer interval is not confirmatory because the winner was
selected using the same outcomes. No p-value or null-hypothesis decision is
reported. A small numerical lead or an interval that crosses zero must not be
restated as equivalence, non-inferiority, or superiority.

## Coverage states

Use exactly these meanings when building broader tables:

- `completed`: the exact named track has a validated result for the cell;
- `planned_incomplete`: the cell is eligible and prespecified but the exact
  track is not complete;
- `blocked`: the scientific cell may be meaningful, but an adapter,
  configuration identity, checkpoint, provenance decision, or approval is
  missing; and
- `not_applicable`: the method cannot answer that cell's frozen task, such as
  a binary-only parity procedure on four-class BNCI2014-001.

Never encode planned, blocked, or not-applicable cells as zero. Never impute
them into an overall average. Do not mark a cell completed using a result from
a differently preprocessed, differently split, historical, author-recipe,
deployment, transfer, or candidate-screen artifact.

## Names are not join keys

Some display names recur across tracks. Common TCFormer and author-recipe
TCFormer are intentional aliases for one architecture under different
training procedures. Common FBCNet and author-recipe FBCNet have the same
relationship. CardinalFBMS from scratch in the common recipe is not the
pretrained CardinalFBMS transfer procedure. ORBIT-v1–v5 share one classifier
class but depend on distinct configurations.

Join results only with a stable registry ID plus the exact track and artifact
digest. Do not join by display name, Python class, family name, runtime
condition, or nearest available score.

## Participant and metric boundaries

- The participant is the independent unit. Folds, trials, and seeds are
  repeated measures.
- Common overall balanced accuracy is an equal average of five dataset means,
  not a trial-pooled micro score.
- Trial-micro accuracy, NLL, Brier, and ECE may be shown separately, but they
  weight cohorts by trial count and repeated seeds.
- Four-class and binary class semantics differ. Cross-dataset macro AUROC and
  other pooled classification metrics are intentionally undefined in the
  trial-micro table.
- Runtime reflects shared-workstation conditions and is an engineering
  measurement, not a scientific performance tie-breaker.

## Claims allowed by the current common bundle

It is accurate to say that the exact common grid completed 96,320 jobs with
no missing, failed, extra, partial, or live-claim jobs, and that
`cardinal_fbc_compactdyn_scale025_extended` was the observed descriptive
leader under the frozen common recipe.

It is not accurate to call that model independently confirmed, globally best,
state of the art, clinically validated, proven useful for disabled or
paralyzed people, or definitively superior to TCFormer. It is also not
accurate to call Gauge a successful promoted model: Gauge failed its frozen
Gate 1 even though its overall subset mean was numerically favorable.
