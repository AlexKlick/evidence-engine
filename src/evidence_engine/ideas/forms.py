"""Product forms: one pain -> many forms (never assume every pain is SaaS)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormSpec:
    form: str
    pitch_template: str
    pricing_mechanism: str
    smallest_paid_test: str
    mvp_sketch: str


FORMS: list[FormSpec] = [
    FormSpec(
        form="software",
        pitch_template="Software that {outcome}",
        pricing_mechanism="hybrid_base_plus_usage",
        smallest_paid_test="Landing page with visible price + refundable deposit",
        mvp_sketch="Single-workflow wedge app with checkout on the core loop",
    ),
    FormSpec(
        form="productized_service",
        pitch_template="Done-for-you service that {outcome}",
        pricing_mechanism="fixed_scope_pilot",
        smallest_paid_test="2-week fixed-scope paid pilot sold on the outcome",
        mvp_sketch="Checklist + internal tools; human fulfillment, exceptions logged",
    ),
    FormSpec(
        form="template_info_product",
        pitch_template="Template/playbook that {outcome}",
        pricing_mechanism="one_time",
        smallest_paid_test="Preorder page for the template pack",
        mvp_sketch="Documented workflow + spreadsheet/notion system",
    ),
    FormSpec(
        form="integration",
        pitch_template="One-click {incumbent} integration that {outcome}",
        pricing_mechanism="per_connection_subscription",
        smallest_paid_test="Paid beta for 5 users of the exact X->Y sync",
        mvp_sketch="Narrow X->Y sync with exception queue",
    ),
    FormSpec(
        form="managed_workflow",
        pitch_template="Managed workflow that {outcome} every week",
        pricing_mechanism="monthly_retainer",
        smallest_paid_test="First month retainer, outcome-guaranteed",
        mvp_sketch="Scheduled pipeline + human review step",
    ),
    FormSpec(
        form="marketplace",
        pitch_template="Marketplace matching buyers who need {outcome}",
        pricing_mechanism="take_rate",
        smallest_paid_test="Manually broker 3 transactions, take a cut",
        mvp_sketch="Concierge brokerage over email/spreadsheet",
    ),
]


def _outcome(hypothesis) -> str:
    job = getattr(hypothesis, "job", None) or "the recurring job"
    pain = getattr(hypothesis, "pain", None) or "removes the manual work"
    return f"handles: {job} — {pain}"


def generate_forms(hypothesis) -> list[dict[str, str]]:
    """All six forms for one hypothesis (scoring picks the winners)."""
    incumbent = getattr(hypothesis, "current_paid_alternative", None) or "A->B"
    return [
        {
            "form": spec.form,
            "pitch": spec.pitch_template.format(
                outcome=_outcome(hypothesis), incumbent=incumbent
            ),
            "pricing_mechanism": spec.pricing_mechanism,
            "smallest_paid_test": spec.smallest_paid_test,
            "mvp_sketch": spec.mvp_sketch,
        }
        for spec in FORMS
    ]
