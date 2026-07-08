# Embedding-model A/B — retrieval quality

Pure-semantic (vector-only) search on a labeled fixture. `hit@1` = expected file is the top result; `hit@k` = expected file anywhere in top-k.

## Summary

| metric | gemma | qwen |
| --- | --- | --- |
| model_name | google/embeddinggemma-300m | Qwen/Qwen3-Embedding-4B |
| embedding_dimension | 768 | 2560 |
| backend | torch | torch |
| hit@1 rate | 1.0 | 1.0 |
| hit@k rate | 1.0 | 1.0 |
| mean latency (ms) | 196.0 | 563.6 |
| index seconds | 52.17 | 305.54 |
| chunks | 8 | 8 |

## Per-query, side by side

### Q1: How do we know when an access token is about to expire and needs renewal?
*expected:* `oauth_token_cache.py`

**gemma** ✅ (hit@1=True, hit@k=True, 255.2ms)
- `1` `src/oauth_token_cache.py` (score=0.5679, token_needs_refresh) — def token_needs_refresh(self, now): return self._token is None or now + self._refresh_margin >= self._expires_at
- `2` `src/oauth_token_cache.py` (score=0.4892, OAuthTokenCache) — class OAuthTokenCache: """Stores bearer tokens and refreshes them before expiry.""" def __init__(self, refresh_margin_se
- `3` `src/oauth_token_cache.py` (score=0.417, store_token) — def store_token(self, token, expires_at): self._token = token self._expires_at = expires_at
- `4` `src/oauth_token_cache.py` (score=0.3905, __init__) — def __init__(self, refresh_margin_seconds=60): self._token = None self._expires_at = datetime.min self._refresh_margin =
- `5` `README.md` (score=0.119, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca

**qwen** ✅ (hit@1=True, hit@k=True, 403.5ms)
- `1` `src/oauth_token_cache.py` (score=0.6839, token_needs_refresh) — def token_needs_refresh(self, now): return self._token is None or now + self._refresh_margin >= self._expires_at
- `2` `src/oauth_token_cache.py` (score=0.6269, OAuthTokenCache) — class OAuthTokenCache: """Stores bearer tokens and refreshes them before expiry.""" def __init__(self, refresh_margin_se
- `3` `src/oauth_token_cache.py` (score=0.5581, store_token) — def store_token(self, token, expires_at): self._token = token self._expires_at = expires_at
- `4` `src/oauth_token_cache.py` (score=0.5752, __init__) — def __init__(self, refresh_margin_seconds=60): self._token = None self._expires_at = datetime.min self._refresh_margin =
- `5` `README.md` (score=0.4228, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca

### Q2: Where do we keep API bearer credentials so we don't re-authenticate on every call?
*expected:* `oauth_token_cache.py`

**gemma** ✅ (hit@1=True, hit@k=True, 233.8ms)
- `1` `src/oauth_token_cache.py` (score=0.5086, OAuthTokenCache) — class OAuthTokenCache: """Stores bearer tokens and refreshes them before expiry.""" def __init__(self, refresh_margin_se
- `2` `src/oauth_token_cache.py` (score=0.3957, store_token) — def store_token(self, token, expires_at): self._token = token self._expires_at = expires_at
- `3` `src/oauth_token_cache.py` (score=0.3512, __init__) — def __init__(self, refresh_margin_seconds=60): self._token = None self._expires_at = datetime.min self._refresh_margin =
- `4` `src/oauth_token_cache.py` (score=0.3188, token_needs_refresh) — def token_needs_refresh(self, now): return self._token is None or now + self._refresh_margin >= self._expires_at
- `5` `README.md` (score=0.1944, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca

**qwen** ✅ (hit@1=True, hit@k=True, 654.8ms)
- `1` `src/oauth_token_cache.py` (score=0.5413, OAuthTokenCache) — class OAuthTokenCache: """Stores bearer tokens and refreshes them before expiry.""" def __init__(self, refresh_margin_se
- `2` `src/oauth_token_cache.py` (score=0.5082, store_token) — def store_token(self, token, expires_at): self._token = token self._expires_at = expires_at
- `3` `src/oauth_token_cache.py` (score=0.4599, __init__) — def __init__(self, refresh_margin_seconds=60): self._token = None self._expires_at = datetime.min self._refresh_margin =
- `4` `README.md` (score=0.4677, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca
- `5` `src/oauth_token_cache.py` (score=0.4059, token_needs_refresh) — def token_needs_refresh(self, now): return self._token is None or now + self._refresh_margin >= self._expires_at

### Q3: Check whether the money a customer actually paid matches the sum of what they were billed.
*expected:* `invoice_reconciler.py`

**gemma** ✅ (hit@1=True, hit@k=True, 219.4ms)
- `1` `src/invoice_reconciler.py` (score=0.5645, reconcile_invoice_totals) — def reconcile_invoice_totals(invoice_lines, payment_amount): """Compare summed invoice line amounts to the received paym
- `2` `README.md` (score=0.1814, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca
- `3` `src/oauth_token_cache.py` (score=0.0868, token_needs_refresh) — def token_needs_refresh(self, now): return self._token is None or now + self._refresh_margin >= self._expires_at
- `4` `src/noise.py` (score=0.0676, render_markdown_table) — def render_markdown_table(rows): return "\n".join("|" + "|".join(map(str, row)) + "|" for row in rows)
- `5` `src/oauth_token_cache.py` (score=0.0486, OAuthTokenCache) — class OAuthTokenCache: """Stores bearer tokens and refreshes them before expiry.""" def __init__(self, refresh_margin_se

**qwen** ✅ (hit@1=True, hit@k=True, 561.6ms)
- `1` `src/invoice_reconciler.py` (score=0.7683, reconcile_invoice_totals) — def reconcile_invoice_totals(invoice_lines, payment_amount): """Compare summed invoice line amounts to the received paym
- `2` `README.md` (score=0.3445, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca
- `3` `src/oauth_token_cache.py` (score=0.2403, token_needs_refresh) — def token_needs_refresh(self, now): return self._token is None or now + self._refresh_margin >= self._expires_at
- `4` `src/warehouse_router.py` (score=0.2247, choose_cold_chain_shipping_route) — def choose_cold_chain_shipping_route(package): """Route refrigerated medicine through a cold-chain carrier.""" if packag
- `5` `src/noise.py` (score=0.2144, render_markdown_table) — def render_markdown_table(rows): return "\n".join("|" + "|".join(map(str, row)) + "|" for row in rows)

### Q4: Detect a mismatch between billed line items and the settlement amount received.
*expected:* `invoice_reconciler.py`

**gemma** ✅ (hit@1=True, hit@k=True, 249.8ms)
- `1` `src/invoice_reconciler.py` (score=0.4973, reconcile_invoice_totals) — def reconcile_invoice_totals(invoice_lines, payment_amount): """Compare summed invoice line amounts to the received paym
- `2` `README.md` (score=0.2073, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca
- `3` `src/noise.py` (score=0.0713, render_markdown_table) — def render_markdown_table(rows): return "\n".join("|" + "|".join(map(str, row)) + "|" for row in rows)
- `4` `src/oauth_token_cache.py` (score=0.0554, token_needs_refresh) — def token_needs_refresh(self, now): return self._token is None or now + self._refresh_margin >= self._expires_at
- `5` `src/warehouse_router.py` (score=0.0459, choose_cold_chain_shipping_route) — def choose_cold_chain_shipping_route(package): """Route refrigerated medicine through a cold-chain carrier.""" if packag

**qwen** ✅ (hit@1=True, hit@k=True, 863.6ms)
- `1` `src/invoice_reconciler.py` (score=0.6393, reconcile_invoice_totals) — def reconcile_invoice_totals(invoice_lines, payment_amount): """Compare summed invoice line amounts to the received paym
- `2` `README.md` (score=0.3678, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca
- `3` `src/oauth_token_cache.py` (score=0.2128, token_needs_refresh) — def token_needs_refresh(self, now): return self._token is None or now + self._refresh_margin >= self._expires_at
- `4` `src/warehouse_router.py` (score=0.2078, choose_cold_chain_shipping_route) — def choose_cold_chain_shipping_route(package): """Route refrigerated medicine through a cold-chain carrier.""" if packag
- `5` `src/oauth_token_cache.py` (score=0.1864, __init__) — def __init__(self, refresh_margin_seconds=60): self._token = None self._expires_at = datetime.min self._refresh_margin =

### Q5: Make sure temperature-sensitive pharmaceutical shipments travel via a refrigerated carrier.
*expected:* `warehouse_router.py`

**gemma** ✅ (hit@1=True, hit@k=True, 82.7ms)
- `1` `src/warehouse_router.py` (score=0.4719, choose_cold_chain_shipping_route) — def choose_cold_chain_shipping_route(package): """Route refrigerated medicine through a cold-chain carrier.""" if packag
- `2` `README.md` (score=0.1272, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca
- `3` `src/oauth_token_cache.py` (score=0.1073, OAuthTokenCache) — class OAuthTokenCache: """Stores bearer tokens and refreshes them before expiry.""" def __init__(self, refresh_margin_se
- `4` `src/oauth_token_cache.py` (score=0.0675, __init__) — def __init__(self, refresh_margin_seconds=60): self._token = None self._expires_at = datetime.min self._refresh_margin =
- `5` `src/oauth_token_cache.py` (score=0.0556, store_token) — def store_token(self, token, expires_at): self._token = token self._expires_at = expires_at

**qwen** ✅ (hit@1=True, hit@k=True, 409.1ms)
- `1` `src/warehouse_router.py` (score=0.7984, choose_cold_chain_shipping_route) — def choose_cold_chain_shipping_route(package): """Route refrigerated medicine through a cold-chain carrier.""" if packag
- `2` `README.md` (score=0.4913, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca
- `3` `src/oauth_token_cache.py` (score=0.2625, __init__) — def __init__(self, refresh_margin_seconds=60): self._token = None self._expires_at = datetime.min self._refresh_margin =
- `4` `src/oauth_token_cache.py` (score=0.2576, token_needs_refresh) — def token_needs_refresh(self, now): return self._token is None or now + self._refresh_margin >= self._expires_at
- `5` `src/invoice_reconciler.py` (score=0.2386, reconcile_invoice_totals) — def reconcile_invoice_totals(invoice_lines, payment_amount): """Compare summed invoice line amounts to the received paym

### Q6: Pick a delivery method that keeps perishable goods cold while in transit.
*expected:* `warehouse_router.py`

**gemma** ✅ (hit@1=True, hit@k=True, 135.4ms)
- `1` `src/warehouse_router.py` (score=0.5065, choose_cold_chain_shipping_route) — def choose_cold_chain_shipping_route(package): """Route refrigerated medicine through a cold-chain carrier.""" if packag
- `2` `README.md` (score=0.1155, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca
- `3` `src/oauth_token_cache.py` (score=0.1058, OAuthTokenCache) — class OAuthTokenCache: """Stores bearer tokens and refreshes them before expiry.""" def __init__(self, refresh_margin_se
- `4` `src/oauth_token_cache.py` (score=0.1007, store_token) — def store_token(self, token, expires_at): self._token = token self._expires_at = expires_at
- `5` `src/oauth_token_cache.py` (score=0.0728, __init__) — def __init__(self, refresh_margin_seconds=60): self._token = None self._expires_at = datetime.min self._refresh_margin =

**qwen** ✅ (hit@1=True, hit@k=True, 488.8ms)
- `1` `src/warehouse_router.py` (score=0.7273, choose_cold_chain_shipping_route) — def choose_cold_chain_shipping_route(package): """Route refrigerated medicine through a cold-chain carrier.""" if packag
- `2` `README.md` (score=0.5177, Quality Fixture) — # Quality Fixture This miniature project is intentionally built to test semantic code search. It contains OAuth token ca
- `3` `src/oauth_token_cache.py` (score=0.2802, __init__) — def __init__(self, refresh_margin_seconds=60): self._token = None self._expires_at = datetime.min self._refresh_margin =
- `4` `src/oauth_token_cache.py` (score=0.2526, store_token) — def store_token(self, token, expires_at): self._token = token self._expires_at = expires_at
- `5` `src/invoice_reconciler.py` (score=0.2475, reconcile_invoice_totals) — def reconcile_invoice_totals(invoice_lines, payment_amount): """Compare summed invoice line amounts to the received paym
