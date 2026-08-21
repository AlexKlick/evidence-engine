# ADR-0001: Local-first collection and inference

Date: 2026-08-21
Status: accepted

## Context

The founding design doc recommends contracted SERP providers, Google Ads API,
YouTube Data API and Reddit API as collection sources. As of 2026-08-21 none of
those entitlements exist for this project: Google Custom Search JSON is closed
to new customers (transition ends 2027-01-01); a commercial SERP provider
requires procurement and contract review; YouTube's developer policy restricts
API-Data aggregation and prohibits scraping; Reddit's Developer Terms require a
separate agreement for commercial use and derived-data revenue.

The workstation already runs a self-hosted SearXNG on loopback `:8018`
(keyless, JSON API), a text-main OpenAI-compatible lane on `:18000`
(Qwen3.8-27B) and an embedding lane on `:6900` (Qwen3-Embedding-0.6B, 1024-dim).

## Decision

1. v1 collects live **only** through the local SearXNG adapter, under a
   personal-research entitlement (`commercial_use: false`,
   `external_inference: false`, `store_raw: false`).
2. All NLP (embeddings, pain-claim extraction, classification) runs on
   loopback model lanes. No collected text is sent to third-party hosted APIs.
3. Every other adapter ships as a gated stub wired to the entitlement
   registry; enabling one is a config + credential/contract change, not a code
   change.

## Consequences

- $0 external spend; the pipeline is runnable today and testable offline.
- Demand signals are weaker than official Google Ads keyword metrics
  (no search volume / competition / bid ranges). When a Google Ads
  entitlement lands, its adapter fills `search_volume`-class features without
  schema changes.
- SearXNG upstream-engine terms are not reviewed for commercial reuse; the
  posture stays personal research until that review happens. The registry
  records this so downstream consumers can check `commercial_use`.
