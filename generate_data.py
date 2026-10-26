"""Generate a reproducible synthetic sales-lead dataset for demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 2500
DATA_PATH = Path("data/sales_leads.csv")

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    lead_source = rng.choice(
        ["Website", "Referral", "LinkedIn", "Email Campaign", "Cold Call"],
        N_ROWS, p=[0.28, 0.18, 0.18, 0.20, 0.16]
    )
    industry = rng.choice(
        ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing"],
        N_ROWS
    )
    company_size = rng.choice(["Small", "Medium", "Large"], N_ROWS, p=[0.45, 0.35, 0.20])
    region = rng.choice(["North", "South", "East", "West"], N_ROWS)
    touchpoints = rng.integers(1, 13, N_ROWS)
    days_to_contact = rng.integers(0, 31, N_ROWS)
    website_visits = rng.poisson(4, N_ROWS).clip(0, 20)
    email_engagement = rng.integers(0, 11, N_ROWS)
    lead_score = rng.integers(10, 101, N_ROWS)
    deal_value = np.round(rng.lognormal(mean=8.2, sigma=0.65, size=N_ROWS), 2)

    # Build a plausible signal so the demo can learn meaningful patterns.
    log_odds = (
        -2.0
        + 0.22 * touchpoints
        - 0.075 * days_to_contact
        + 0.13 * website_visits
        + 0.16 * email_engagement
        + 0.035 * (lead_score - 50)
        + np.where(lead_source == "Referral", 0.75, 0)
        + np.where(lead_source == "Website", 0.35, 0)
        + np.where(company_size == "Large", 0.35, 0)
        + np.where(industry == "Technology", 0.25, 0)
        + rng.normal(0, 0.75, N_ROWS)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    converted = rng.binomial(1, probability)

    df = pd.DataFrame({
        "lead_source": lead_source,
        "industry": industry,
        "company_size": company_size,
        "region": region,
        "touchpoints": touchpoints,
        "days_to_contact": days_to_contact,
        "website_visits": website_visits,
        "email_engagement": email_engagement,
        "lead_score": lead_score,
        "deal_value": deal_value,
        "converted": converted,
    })

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} synthetic lead records to {DATA_PATH}")
    print(f"Conversion rate: {df['converted'].mean():.1%}")

if __name__ == "__main__":
    main()
