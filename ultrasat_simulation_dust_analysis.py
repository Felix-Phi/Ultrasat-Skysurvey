import yaml
import os
import hashlib
import pickle
import warnings
import matplotlib.pyplot as plt
import skysurvey
import pandas as pd
import glob
from datetime import datetime
warnings.filterwarnings("ignore", category=FutureWarning, module="skysurvey")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Modules"))



def main():
    print("main() function is starting...")
    # Load the configuration file
    RV_value = [1, 2, 3.1]
    dfs = {}

    for i in RV_value:
        folder_path = "Fits/rv" + str(i)
        text_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        
        dfs[i] = pd.DataFrame()  # Initialisiere ein leeres DataFrame
        k=0
        for file in text_files:
            k=k+1
            df = pd.read_csv(file)
            print(str(i)+","+str(k)+"="+str(len(df)))
            dfs[i] = pd.concat([dfs[i], df], ignore_index=True)
        dfs[i].reset_index(drop=True)
    print(len(dfs[1]),len(dfs[2]),len(dfs[3.1]))

    df=pd.DataFrame({
        "index RV1": dfs[1]["index"],
    #    "index RV2": dfs[2]["index"],
    #    "index RV3.1": dfs[3.1]["index"],
        "RV1 chi2 ZTF": dfs[1]["chi2 ZTF"],
        "RV2 chi2 ZTF": dfs[2]["chi2 ZTF"],
        "RV3.1 chi2 ZTF": dfs[3.1]["chi2 ZTF"],
        "RV1 chi2 combi": dfs[1]["chi2 combi"],
        "RV2 chi2 combi": dfs[2]["chi2 combi"],
        "RV3.1 chi2 combi": dfs[3.1]["chi2 combi"],
        "RV1 chi2dof ZTF": dfs[1]["chi2dof ZTF"],
        "RV2 chi2dof ZTF": dfs[2]["chi2dof ZTF"],
        "RV3.1 chi2dof ZTF": dfs[3.1]["chi2dof ZTF"],
        "RV1 chi2dof combi": dfs[1]["chi2dof combi"],
        "RV2 chi2dof combi": dfs[2]["chi2dof combi"],
        "RV3.1 chi2dof combi": dfs[3.1]["chi2dof combi"]
    })

    print(df)
    limiter=0.3
    condition1 = abs(df['RV1 chi2 ZTF'] - df['RV2 chi2 ZTF']) <= limiter * df['RV1 chi2 ZTF']
    condition2 = abs(df['RV2 chi2 ZTF'] - df['RV3.1 chi2 ZTF']) <= limiter * df['RV2 chi2 ZTF']
    condition3 = abs(df['RV3.1 chi2 ZTF'] - df['RV1 chi2 ZTF']) <= limiter*2 * df['RV3.1 chi2 ZTF']
    condition4 = abs(df['RV1 chi2 combi'] - df['RV2 chi2 combi']) <= limiter* df['RV1 chi2 combi']
    condition5 = abs(df['RV2 chi2 combi'] - df['RV3.1 chi2 combi']) <= limiter* df['RV2 chi2 combi']
    condition6 = abs(df['RV3.1 chi2 combi'] - df['RV1 chi2 combi']) <= limiter*2* df['RV3.1 chi2 combi']
    condition7 = abs(df['RV3.1 chi2dof ZTF']- df['RV3.1 chi2dof combi']) <= limiter*df['RV3.1 chi2dof ZTF']
    df=df[condition1 & condition2 & condition3 & condition4 & condition5 & condition6 & condition7]
    #df=df[(df >= 1e-10).all(axis=1)]
    #df=df[(df.drop(columns=['index RV1',"index RV2","index RV3.1"]) <2).all(axis=1)]
    print(df)
    print(len(df))

    plt.figure()
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 18})
    plt.plot([1,2,3.1], [sum(df["RV1 chi2 ZTF"]),sum(df["RV2 chi2 ZTF"]),sum(df["RV3.1 chi2 ZTF"])], label="ZTF",linewidth=4)
    plt.plot([1,2,3.1], [sum(df["RV1 chi2 combi"]),sum(df["RV2 chi2 combi"]),sum(df["RV3.1 chi2 combi"])], label="ZTF + ULTRASAT",linewidth=4)
    plt.plot([], [], " ", label="Total # of SNeIa fits = "+str(len(df)))
    plt.plot([], [], " ", label="Total # of DoF ZTF = "+str(round(sum(df["RV3.1 chi2 ZTF"]/df["RV3.1 chi2dof ZTF"]))))
    plt.plot([], [], " ", label="Total # of DoF ZTF+ULTRASAT = "+str(round(sum(df["RV3.1 chi2 combi"]/df["RV3.1 chi2dof combi"]))))
    plt.xlabel(f"$R_V$ value of fit. (simulated $R_V = 3.1$)")
    plt.ylabel("Sum of all "+r"$\chi ^2$")
    plt.ylim(27500,33100)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fits/results/chi2_dust.pdf")

    plt.figure()
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 18})
    plt.plot([1,2,3.1], [sum(df["RV1 chi2dof ZTF"]),sum(df["RV2 chi2dof ZTF"]),sum(df["RV3.1 chi2dof ZTF"])], label="ZTF",linewidth=4)
    plt.plot([1,2,3.1], [sum(df["RV1 chi2dof combi"]),sum(df["RV2 chi2dof combi"]),sum(df["RV3.1 chi2dof combi"])], label="ZTF + ULTRASAT",linewidth=4)
    plt.plot([], [], " ", label="Total # of SNeIa fits = "+str(len(df)))
    plt.plot([], [], " ", label="Total # of DoF ZTF = "+str(round(sum(df["RV3.1 chi2 ZTF"]/df["RV3.1 chi2dof ZTF"]))))
    plt.plot([], [], " ", label="Total # of DoF ZTF+ULTRASAT = "+str(round(sum(df["RV3.1 chi2 combi"]/df["RV3.1 chi2dof combi"]))))
    plt.xlabel(f"$R_V$ value of fit. (simulated $R_V = 3.1$)")
    plt.ylabel("Sum of all "+r"$\chi ^2$/DoF")
    plt.ylim(220,242.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fits/results/chi2_dof_dust.pdf")

    df["RV1-RV2"]=(df["RV1 chi2 combi"]-df["RV2 chi2 combi"])/df["RV1 chi2 combi"]
    df["RV1-RV3.1"]=(df["RV1 chi2 combi"]-df["RV3.1 chi2 combi"])/df["RV1 chi2 combi"]
    sorteddf=df.sort_values(by="RV1-RV3.1")
    print(sorteddf)
    totaldof= sum(df["RV3.1 chi2 combi"]/df["RV3.1 chi2dof combi"])
    print(totaldof)


if __name__ == "__main__":
    main()