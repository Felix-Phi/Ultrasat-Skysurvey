import os
import warnings
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Modules"))

def main():
    print("main() function is starting...")

    RV_values = [1, 2, 3.1]
    dfs = {}

    for rv in RV_values:
        folder_path = f"Fits/rv{rv}"
        files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        df_all = pd.DataFrame()
        for idx, file in enumerate(files, start=1):
            df_part = pd.read_csv(file)
            print(f"R_V={rv}, File {idx}/{len(files)}: {len(df_part)} rows")
            df_all = pd.concat([df_all, df_part], ignore_index=True)
        dfs[rv] = df_all.reset_index(drop=True)

    df = pd.DataFrame({
        'RV1 chi2 ZTF': dfs[1]['chi2 ZTF'],
        'RV2 chi2 ZTF': dfs[2]['chi2 ZTF'],
        'RV3.1 chi2 ZTF': dfs[3.1]['chi2 ZTF'],
        'RV1 chi2 combi': dfs[1]['chi2 combi'],
        'RV2 chi2 combi': dfs[2]['chi2 combi'],
        'RV3.1 chi2 combi': dfs[3.1]['chi2 combi'],
        'RV1 chi2dof ZTF': dfs[1]['chi2dof ZTF'],
        'RV2 chi2dof ZTF': dfs[2]['chi2dof ZTF'],
        'RV3.1 chi2dof ZTF': dfs[3.1]['chi2dof ZTF'],
        'RV1 chi2dof combi': dfs[1]['chi2dof combi'],
        'RV2 chi2dof combi': dfs[2]['chi2dof combi'],
        'RV3.1 chi2dof combi': dfs[3.1]['chi2dof combi']
    })

    limiter = 0.3
    c1 = abs(df['RV1 chi2 ZTF'] - df['RV2 chi2 ZTF']) <= limiter * df['RV1 chi2 ZTF']
    c2 = abs(df['RV2 chi2 ZTF'] - df['RV3.1 chi2 ZTF']) <= limiter * df['RV2 chi2 ZTF']
    c3 = abs(df['RV3.1 chi2 ZTF'] - df['RV1 chi2 ZTF']) <= limiter*2 * df['RV3.1 chi2 ZTF']
    c4 = abs(df['RV1 chi2 combi'] - df['RV2 chi2 combi']) <= limiter* df['RV1 chi2 combi']
    c5 = abs(df['RV2 chi2 combi'] - df['RV3.1 chi2 combi']) <= limiter* df['RV2 chi2 combi']
    c6 = abs(df['RV3.1 chi2 combi'] - df['RV1 chi2 combi']) <= limiter*2* df['RV3.1 chi2 combi']
    c7 = abs(df['RV3.1 chi2dof ZTF'] - df['RV3.1 chi2dof combi']) <= limiter*df['RV3.1 chi2dof ZTF']
    df_filtered = df[c1 & c2 & c3 & c4 & c5 & c6 & c7].reset_index(drop=True)
    print(f"Anzahl gefilterter Einträge: {len(df_filtered)}")

    # Relative Unterschiede basierend auf reduzierten chi2 (mit SEM statt STD)
    long_rows = []
    for rv in RV_values:
        z_dof = df_filtered[f"RV{rv} chi2dof ZTF"]
        c_dof = df_filtered[f"RV{rv} chi2dof combi"]
        rel_red = c_dof - z_dof
        tmp = pd.DataFrame({
            "RV": rv,
            "chi2dof_ZTF": z_dof,
            "chi2dof_combi": c_dof,
            "rel_red": rel_red
        })
        long_rows.append(tmp)

    df_rel = pd.concat(long_rows, ignore_index=True)

    # SEM berechnen statt STD
    def sem(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))

    stats = df_rel.groupby('RV')['rel_red'].agg(['mean', sem, 'median', lambda x: x.quantile(0.75)-x.quantile(0.25)])
    stats.columns = ['mean', 'SEM', 'median', 'IQR']
    print(stats)

    plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size': 18})
    plt.errorbar(stats.index, stats['mean'], yerr=stats['SEM'], fmt='-o', linewidth=4,markersize=10, capsize=5, color="tab:gray",
                 label='Difference of ZTF+ULTRASAT and ZTF only (± SEM)')
    plt.xlabel(r"$R_V$ value of fit (simulated $R_V=3.1$)")
    plt.ylabel(r"Mean difference in $\chi ^2$/DoF")
    #plt.title("Relative improvement in reduced $\chi^2$ (ULTRASAT vs ZTF)")
    #plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fits/results/relative_reduced_chi2_sem.pdf")
    print("Saved relative reduced chi2 plot with SEM to Fits/results/relative_reduced_chi2_sem.pdf")


    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 18})

    RV_values = [1, 2, 3.1]

    # Summen für chi2 berechnen
    chi2_ztf_sums = [df_filtered[f"RV{rv} chi2 ZTF"].sum() for rv in RV_values]
    chi2_combi_sums = [df_filtered[f"RV{rv} chi2 combi"].sum() for rv in RV_values]

    plt.plot(RV_values, chi2_ztf_sums, label="ZTF", linewidth=4)
    plt.plot(RV_values, chi2_combi_sums, label="ZTF + ULTRASAT", linewidth=4)

    plt.plot([], [], " ", label="Total # of SNeIa fits = "+str(len(df_filtered)))
    plt.plot([], [], " ", label="Total # of DoF ZTF = "+str(round(sum(df_filtered["RV3.1 chi2 ZTF"]/df_filtered["RV3.1 chi2dof ZTF"]))))
    plt.plot([], [], " ", label="Total # of DoF ZTF+ULTRASAT = "+str(round(sum(df_filtered["RV3.1 chi2 combi"]/df_filtered["RV3.1 chi2dof combi"]))))

    plt.xlabel(f"$R_V$ value of fit (simulated $R_V = 3.1$")
    plt.ylabel("Sum of all "+r"$\chi ^2$")
    plt.ylim(27500, 33100)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fits/results/chi2_dust.pdf")

    # Plot Chi²/DoF Mittelwerte
    plt.figure(figsize=(10, 7))

    chi2dof_ztf_means = [df_filtered[f"RV{rv} chi2dof ZTF"].mean() for rv in RV_values]
    chi2dof_combi_means = [df_filtered[f"RV{rv} chi2dof combi"].mean() for rv in RV_values]

    plt.plot(RV_values, chi2dof_ztf_means, label="ZTF", linewidth=4)
    plt.plot(RV_values, chi2dof_combi_means, label="ZTF + ULTRASAT", linewidth=4)

    plt.plot([], [], " ", label="Total # of SNeIa fits = "+str(len(df_filtered)))
    plt.plot([], [], " ", label="Total # of DoF ZTF = "+str(round(sum(df_filtered["RV3.1 chi2 ZTF"]/df_filtered["RV3.1 chi2dof ZTF"]))))
    plt.plot([], [], " ", label="Total # of DoF ZTF+ULTRASAT = "+str(round(sum(df_filtered["RV3.1 chi2 combi"]/df_filtered["RV3.1 chi2dof combi"]))))

    plt.xlabel(f"$R_V$ value of fit (simulated $R_V = 3.1$")
    plt.ylabel("Mean "+r"$\chi ^2$/DoF")
    plt.ylim(1.37, 1.52)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fits/results/chi2_dof_dust.pdf")
if __name__ == "__main__":
    main()