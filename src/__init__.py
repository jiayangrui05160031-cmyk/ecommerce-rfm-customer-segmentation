"""ecommerce-rfm-customer-segmentation — public package entry.

Library-style entry points (for notebooks and downstream apps):

    from src import analyze, profile_inference
    result = analyze(my_df, mapping=SchemaMapping(...),
                      profile="retail",
                      steps=["features", "cluster", "clv"])

    # Or: just point it at any DataFrame and let it guess.
    result = analyze(my_df)            # uses profile_inference()
    result = analyze(my_df, "retail")  # short form, retail profile

The CLI is run_modern.py at the repo root.
"""
from src.data_sources import (
    Dataset, BaseDataSource,
    SchemaMapping, DomainProfile,
    retail_mapping, retail_profile, donation_profile, PROFILES,
    profile_inference,
    load_dataset, available_sources,
)
from src.pipeline import run_pipeline, PipelineResult, ALL_STEPS, precompute_customer_events


def analyze(
    df,
    mapping=None,
    profile="retail",
    *,
    steps=None,
    cfg=None,
    out_dir=None,
    nba_out=None,
    skip_agents_for_speed=False,
):
    """One-call analysis for any entity-event DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form event log. Required columns depend on profile
        (see SchemaMapping.required() for the retail profile).
    mapping : SchemaMapping, optional
        Declares which column plays which role. If omitted AND
        profile is a string, profile_inference is run to guess it.
    profile : str | DomainProfile
        Either a profile name from PROFILES (default: "retail")
        or a DomainProfile instance.
    steps : list[str], optional
        Subset of ALL_STEPS. None = run everything.
    cfg : AppConfig, optional
        Pass an explicit config (defaults to the project config.yaml).
    out_dir, nba_out : Path, optional
        Where to write the HTML report and the NBA CSV.

    Returns
    -------
    PipelineResult
    """
    # Resolve profile first (string or instance)
    if isinstance(profile, str):
        if profile not in PROFILES:
            raise ValueError(
                "Unknown profile '%s'. Available: %s" % (profile, list(PROFILES))
            )
        prof = PROFILES[profile]
    else:
        prof = profile

    # Resolve mapping (explicit > guessed)
    if mapping is None:
        mapping, _debug = profile_inference(df)

    # Validate: required columns must exist
    missing = [c for c in mapping.required() if c not in df.columns]
    if missing:
        raise ValueError(
            "Mapping requires columns %s but DataFrame has %s. "
            "Pass an explicit SchemaMapping or fix the data." % (missing, list(df.columns))
        )
    # Required dtype coercion
    import pandas as pd
    df = df.copy()
    df[mapping.timestamp] = pd.to_datetime(df[mapping.timestamp], errors="coerce")
    df[mapping.value] = pd.to_numeric(df[mapping.value], errors="coerce")
    bad = df[mapping.timestamp].isna() | df[mapping.value].isna()
    if bad.any():
        df = df[~bad].reset_index(drop=True)

    # Build an ad-hoc Dataset and run the unified pipeline
    ds = Dataset(
        name="adhoc",
        transactions=df,
        meta={"source": "analyze()", "profile": prof.name},
        mapping=mapping,
        profile=prof,
    )
    return run_pipeline(
        ds, steps=steps, cfg=cfg,
        out_dir=out_dir, nba_out=nba_out,
        skip_agents_for_speed=skip_agents_for_speed,
    )


__all__ = [
    "analyze",
    "SchemaMapping", "DomainProfile",
    "retail_mapping", "retail_profile", "donation_profile", "PROFILES",
    "profile_inference",
    "Dataset", "load_dataset", "available_sources",
    "run_pipeline", "PipelineResult", "ALL_STEPS", "precompute_customer_events",
]
