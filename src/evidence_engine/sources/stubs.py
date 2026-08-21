"""Gated stub adapters — entitlement rows exist, none enabled.

Each class documents what its future live implementation must honor.
Enabling one is a config change in source_policies.yaml plus credentials;
the collection code is the work item.
"""

from __future__ import annotations

from evidence_engine.sources.base import GatedStubAdapter


class GoogleAdsAdapter(GatedStubAdapter):
    """KeywordPlanIdeaService: keyword ideas + historical monthly-search,
    competition and bid-range metrics (the doc's greenest Google path).

    Live implementation must:
      * use an approved developer token + OAuth customer ID;
      * respect the 1 request/second cap on keyword-planning methods;
      * cache aggressively — metrics do not need minute-level refresh;
      * emit keyword_metrics records (search_volume/competition/bid_range),
        not SERP results.
    """

    name = "google_ads"
    version = "0.1.0"


class SerpProviderAdapter(GatedStubAdapter):
    """Contracted commercial SERP provider (SerpApi / DataForSEO / Bright Data).

    Live implementation must:
      * read provider credentials from env, never config;
      * store provenance (provider, request id) on each evidence record;
      * honor the provider's contractual permitted uses, deletion and
        retention terms (deletion_propagates: true);
      * remember: provider usage does not by itself establish platform
        authorization for commercial redistribution.
    """

    name = "serp_provider"
    version = "0.1.0"


class YouTubeAdapter(GatedStubAdapter):
    """YouTube Data API — approval/rights gated.

    Live implementation must (per the founding doc's YouTube modes):
      * stay in creator-owned / partner-licensed / discovery-only mode;
      * never aggregate across channels for market scoring until that use
        case is cleared (rights.aggregate stays false until then);
      * never depend on arbitrary transcript scraping — captions.download
        requires edit permission on the video;
      * respect quota (search.list: 50 items/call, 100 calls/day default).
    """

    name = "youtube"
    version = "0.1.0"


class RedditAdapter(GatedStubAdapter):
    """Reddit OAuth API — commercial agreement required first.

    Live implementation must:
      * authenticate via OAuth with a descriptive user agent;
      * honor 100 QPM/client (10-minute average) and X-Ratelimit-* headers;
      * run deletion sync: remove deleted content + deleted-user identifiers,
        including de-identified retention (deletion_propagates: true);
      * keep model_training: false — no AI training on Reddit data without
        permission;
      * route extraction through local inference only.
    """

    name = "reddit"
    version = "0.1.0"


class WebCrawlerAdapter(GatedStubAdapter):
    """Polite crawler for independent domains discovered via SERPs.

    Live implementation must:
      * resolve per-domain policy BEFORE fetching (robots.txt is a crawler
        control, not authorization — RFC 9309);
      * robots.txt cache -> host policy -> token bucket -> conditional GET ->
        exponential backoff -> bounded retries -> content-type/size limits;
      * canonicalize and minimize PII on ingestion;
      * never use proxies to defeat rate limits, CAPTCHAs or bans.
    """

    name = "web_crawler"
    version = "0.1.0"


STUB_ADAPTERS: dict[str, type[GatedStubAdapter]] = {
    cls.name: cls  # type: ignore[misc]
    for cls in (
        GoogleAdsAdapter,
        SerpProviderAdapter,
        YouTubeAdapter,
        RedditAdapter,
        WebCrawlerAdapter,
    )
}
