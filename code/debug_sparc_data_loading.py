"""
SPARC Rotation Curve Data Read - FULLY FIXED
"""
import pandas as pd

SPARC_GALAXY_TABLE = "data/SPARC_Lelli2016c.mrt"
SPARC_ROTATION_TABLE = "data/MassModels_Lelli2016c.mrt"

def main():
    print("="*70)
    print("SPARC ROTATION CURVE - FULLY FIXED READ")
    print("="*70)

    # ----------------------
    # 1. Read Galaxy Table
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
        "Inc_deg": df_gal[5],
        "Q_flag": df_gal[17]
    }).dropna()
    print(f"  Loaded {len(df_gal)} galaxies")

    # ----------------------
    # 2. Read Rotation Curve - FULLY FIXED
    # ----------------------
    print("\n[2/2] Reading Rotation Curve Table...")
    colspecs_rot = [
        (0, 11),    # Galaxy name
        (12, 18),   # Radius R (kpc)
        (19, 25),   # Observed velocity Vobs (km/s)
        (26, 32),   # Velocity error errV (km/s)
        (33, 39),   # Gas velocity Vgas (km/s)
        (40, 46),   # Disk velocity Vdisk (km/s, M/L=1)
        (47, 53)    # Bulge velocity Vbul (km/s, M/L=1)
    ]
    col_names_rot = ["Galaxy", "R_kpc", "Vobs_km_s", "errV_km_s", "Vgas_km_s", "Vdisk_km_s", "Vbul_km_s"]

    # 跳过前10行
    df_rot = pd.read_fwf(
        SPARC_ROTATION_TABLE,
        colspecs=colspecs_rot,
        skiprows=10,
        names=col_names_rot,
        na_values=["---", "99.99", "=====", "-----"]  # 把分隔线设为NA
    )
    df_rot["Galaxy"] = df_rot["Galaxy"].str.strip()
    df_rot = df_rot.dropna()

    # 强制把所有数值列转成float
    numeric_cols = ["R_kpc", "Vobs_km_s", "errV_km_s", "Vgas_km_s", "Vdisk_km_s", "Vbul_km_s"]
    for col in numeric_cols:
        df_rot[col] = pd.to_numeric(df_rot[col], errors="coerce")
    df_rot = df_rot.dropna()

    print(f"  Loaded {len(df_rot)} radial points")
    print(f"  Unique galaxies: {df_rot['Galaxy'].nunique()}")
    print(f"\n数据类型检查：")
    print(df_rot.dtypes)
    print(f"\n前5行真实数据：")
    print(df_rot.head())

    # ----------------------
    # Verify merge
    # ----------------------
    df_merged = pd.merge(df_rot, df_gal, on="Galaxy", how="inner")
    df_clean = df_merged[
        (df_merged["Inc_deg"] >= 20) &
        (df_merged["Q_flag"] <= 2)
    ]
    print(f"\n{len(df_clean)} 点，{df_clean['Galaxy'].nunique()} 星系")
    print("="*70)

if __name__ == "__main__":
    main()
