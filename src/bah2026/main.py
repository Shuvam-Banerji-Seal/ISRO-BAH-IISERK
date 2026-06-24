"""
Bharatiya Antariksh Hackathon 2026 — Challenge 15
Forecasting and Nowcasting of Solar Flares using combined
Soft and Hard X-ray data from Aditya-L1 (SoLEXS + HEL1OS)

Author: IISER Kolkata Team
"""


def main():
    print("=" * 60)
    print("  BAH 2026 — Challenge 15: Solar Flare Prediction")
    print("  Aditya-L1: SoLEXS (soft X-ray) + HEL1OS (hard X-ray)")
    print("=" * 60)
    print("\nPipeline stages:")
    print("  1. Download data from ISSDC PRADAN portal")
    print("  2. Read & preprocess SoLEXS/HEL1OS light curves")
    print("  3. Nowcasting: real-time flare detection")
    print("  4. Forecasting: time-series prediction with lead time")
    print("  5. Visualization dashboard with alerts")
    print("\nSee AGENTS.md for the full problem statement.")


if __name__ == "__main__":
    main()
