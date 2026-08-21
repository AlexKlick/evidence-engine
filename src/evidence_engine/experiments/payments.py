"""Payments provider — stub until a real provider is entitled + configured.

Do NOT wire a hosted checkout before the source rights posture supports it
(commercial_use review). This stub fails loudly rather than half-working.
"""

from __future__ import annotations

from typing import Any, Protocol


class PaymentsProvider(Protocol):
    def create_deposit_checkout(self, idea_id: str, amount_cents: int) -> str: ...


class UnconfiguredPayments:
    """Explicit no-op provider: `available=False`, raises on use."""

    available = False
    reason = (
        "No payments provider configured. Wiring one is a rights + business "
        "decision (see docs/design/evidence-to-revenue.md, pricing section). "
        "Until then the experiment factory emits landing drafts + deposit "
        "CTAs without live checkout."
    )

    def create_deposit_checkout(self, idea_id: str, amount_cents: int) -> str:
        raise NotImplementedError(self.reason)


def get_payments_provider(_config: dict[str, Any] | None = None) -> PaymentsProvider:
    """Return the configured provider — only the stub exists in v1."""
    return UnconfiguredPayments()
