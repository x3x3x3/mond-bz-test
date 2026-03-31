"""
SPARC Rotation Curve Data Read Debug
Verify correct loading of MassModels_Lelli2016c.mrt
"""
import pandas as pd

# ==================================================
# STANDARD FILE PATHS
# ==================================================
SPARC_GALAXY_TABLE = "data/SPARC_Lelli2016c.mrt"
SPARC_ROTATION_TABLE = "data/MassModels_Lelli2016c.mrt"

def main():
    print("="*70)
    print("SPARC ROTATION CURVE DATA READ DEBUG")
    print("="*70)

    # ----------------------
    # 1. Read Galaxy Table (Inc, Q flags)
    # ----------------------
    print("\n[1/2] Reading Galaxy Table...")
    df_gal = pd.read_csv(
        SPARC_GALAXY_TABLE,
        skiprows=98,
        sep=r'\s+',
        header=None,
        engine='python'
    )
    df_gal = pd.DataFrame({
        "Galaxy": df_gal[0].str.strip(),
        "Inc": df_gal[5],
        "Q": df_gal[17]
    }).dropna()
    print(f"  Loaded {len(df_gal)} galaxies with Inc/Q flags")

    # ----------------------
    # 2. Read Rotation Curve Table (official byte positions)
    # ----------------------
    print("\n[2/2] Reading Rotation Curve Table...")
    # Official column positions (1-based → 0-based conversion)
    # From Lelli+2016 official documentation
    colspecs_rot = [
        (0, 11),    # Galaxy name (1-11)
        (12, 18),   # Radius R (kpc, 13-18)
        (19, 25),   # Observed velocity Vobs (km/s, 20-25)
        (26, 32),   # Velocity error errV (km/s, 27-32)
        (33, 39),   # Gas velocity Vgas (km/s, 34-39)
        (40, 46),   # Disk velocity Vdisk (km/s, M/L=1, 41-46)
        (47, 53)    # Bulge velocity Vbul (km/s, M/L=1, 48-53)
    ]
    col_names_rot = ["Galaxy", "R_kpc", "Vobs_km_s", "errV_km_s", "Vgas_km_s", "Vdisk_km_s", "Vbul_km_s"]

    df_rot = pd.read_fwf(
        SPARC_ROTATION_TABLE,
        colspecs=colspecs_rot,
        skiprows=3,  # Skip 3 header lines
        names=col_names_rot,
        na_values=["---", "99.99"]
    )
    df_rot["Galaxy"] = df_rot["Galaxy"].str.strip()
    df_rot = df_rot.dropna()

    print(f"  Loaded {len(df_rot)} radial points")
    print(f"  Unique galaxies in rotation table: {df_rot['Galaxy'].nunique()}")
    print(f"\nFirst 5 rotation points:")
    print(df_rot.head())

    # ----------------------
    # Verify merge
    # ----------------------
    df_merged = pd.merge(df_rot, df_gal, on="Galaxy", how="inner")
    print(f"\nMerged dataset: {len(df_merged)} points, {df_merged['Galaxy'].nunique()} galaxies")
    print("="*70)

if __name__ == "__main__":
    main()
