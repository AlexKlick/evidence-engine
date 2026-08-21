# ADR-0002: Rights-aware adapters, default-deny

Date: 2026-08-21
Status: accepted

## Context

The founding doc's central architectural conclusion: the durable asset is a
rights-aware evidence engine, not a scraped corpus. The same record may be
technically retrievable from three sources while being usable for three
different sets of purposes. Platform terms (Google, YouTube, Reddit) restrict
scraping, aggregation, commercial use, external sharing, AI training and
deletion handling in source-specific ways. "robots allowed" is not equivalent
to "all uses allowed" (RFC 9309).

## Decision

1. `config/source_policies.yaml` is the first-class entitlement registry:
   one row per source with booleans populated from the actual source contract
   (`collect_enabled`, `store_derived`, `store_raw`, `aggregate`,
   `local_inference`, `external_inference`, `model_training`,
   `redisplay_raw_content`, `commercial_use`, `deletion_propagates`,
   `retention_days`) plus a human-readable `disabled_reason` and `notes`.
2. The policy gate is enforced in code at two points: adapter `collect()`
   (template method) and pipeline stages that consume evidence for a purpose
   (aggregation, inference, training, redisplay).
3. **Default-deny**: a source missing from the registry can never be
   collected; an unknown purpose denies; ambiguity denies.
4. Every `source_run` and `evidence_event` snapshots
   `rights_profile` + `policy_version` at fetch time, so policy changes never
   retroactively reinterpret old data — reprocessing is explicit.
5. Lineage-aware deletion: when retention expires or an upstream deletion
   propagates, the deletion service walks
   `evidence → segments → claims → cluster memberships → hypothesis evidence`
   and invalidates dependents (`deletion/service.py`).

## Consequences

- Adding Reddit/YouTube later = adding an entitlement row + implementing the
  adapter, with the gate already enforcing their contract quirks
  (Reddit: deletion sync + no training; YouTube: no cross-channel aggregation
  until reviewed).
- Slightly more ceremony per record (rights snapshot on every row) — accepted
  cost; it is the audit trail that makes the evidence store trustworthy.
- Scoring has a hard `rights_clear` gate: an idea whose lineage contains a
  non-permitted source cannot reach the paid-validation band regardless of
  score.
