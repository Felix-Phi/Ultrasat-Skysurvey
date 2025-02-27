import pandas as pd
from tqdm import tqdm
import sncosmo
from astropy.table import Table
import warnings
from plotting import generate_unique_filename
import os
import matplotlib.pyplot as plt
from datetime import datetime

def prepare_light_curve_table(df):
    """
    Prepares the light curve data as an Astropy Table for sncosmo.

    Args:
        df (pandas.DataFrame): Light curve data.

    Returns:
        astropy.table.Table: Light curve table formatted for sncosmo.
    """
    data = {
        "time": df["time"].values,
        "flux": df["flux"].values,
        "fluxerr": df["fluxerr"].values,
        "band": df["band"].values,
        "zp": df["zp"].values,
        "zpsys": df["zpsys"].values,
    }
    return Table(data)


def perform_light_curve_fit(lc_table, model, redshift,t0=50000,x0=1,x1=0,c=0):
    """
    Fits the light curve data using sncosmo, with error handling.

    Args:
        lc_table (astropy.table.Table): Light curve data table.
        model (sncosmo.Model): sncosmo model to fit.
        redshift (float): Redshift value.

    Returns:
        tuple: Fit result and fitted model, or (None, None) if fitting fails.
    """
    model.set(z=redshift,t0=t0,x0=x0,x1=x1,c=c)
    params = ["t0", "x0", "x1", "c"]
    bounds = {"x1": (-3, 3), "c": (-0.3, 0.3)}

    try:
        # Perform the fit
        result, fitted_model = sncosmo.fit_lc(
            lc_table, model, params, modelcov=True, bounds=bounds
        )

        # Validate the result for NaN values
        if any(pd.isna(result.parameters)):
            raise RuntimeError(f"Fit result contains NaN values: {result.parameters}")

        return result, fitted_model

    except (RuntimeError, ValueError) as e:
        # Log the error and return None
        print(f"Error during fit: {e}")
        return None, None



def build_results_table(combined_lightcurves,ztf_lightcurves, sniadata, sorted_indices):
    """
    Builds a results table with fit parameters for combined and ZTF datasets.

    Args:
        combined_lightcurves (skysurvey.DataSet): Combined dataset (ULTRASAT + ZTF).
        ztf_lightcurves (skysurvey.DataSet): ZTF dataset.
        sniadata (pandas.DataFrame): Original supernova data.
        sorted_indices (list): Sorted indices of combined observations.

    Returns:
        pandas.DataFrame: Table of fit results.
    """
    # Initialize the results table
    columns = [
        "index", "z", "t0", "x0 real", "x1 real", "c real",
        "x0 combi", "x0err combi", "x1 combi", "x1err combi", "c combi", "cerr combi",
        "x0 ZTF", "x0err ZTF", "x1 ZTF", "x1err ZTF", "c ZTF", "cerr ZTF",
        "Suc combi", "Suc ZTF"
    ]
    results_table = pd.DataFrame(columns=columns)
    warnings.filterwarnings("ignore", category=UserWarning)
    unique_dir_name = f"Fits/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(unique_dir_name, exist_ok=True)

    print(f"Created directory: {unique_dir_name}")

    for k, indexx in enumerate(tqdm(sorted_indices, desc="Fitting light curves")):
        
        # ZTF dataset fit
        df_ztf = ztf_lightcurves.data.loc[indexx]
        lc_table_ztf = prepare_light_curve_table(df_ztf)

        model_ztf = sncosmo.Model(source="QinanSalt3")
        z = sniadata["z"][indexx]
        result_ztf, fitted_model_ztf = perform_light_curve_fit(lc_table_ztf, model_ztf, z)
        
        if result_ztf is None:
            print(f"Skipping index {indexx} due to failed ZTF fit.")
            continue
        


        # Combined dataset fit
        df_combined = combined_lightcurves.data.loc[indexx]
        lc_table_combined = prepare_light_curve_table(df_combined)

        model_combined = sncosmo.Model(source="QinanSalt3")
        
        t0=result_ztf.parameters[1]
        x0=result_ztf.parameters[2]
        x1=result_ztf.parameters[3]
        c=result_ztf.parameters[4]
        result_combined, fitted_model_combined= perform_light_curve_fit(lc_table_combined, model_combined, z,t0,x0,x1,c)

        if result_combined is None:
            print(f"Skipping index {indexx} due to failed combined fit.")
            continue




        
        # Generate annotations for the plot
        Text1 = f"The simulated parameters are:\n" + \
                r"$t_0$ = " + f"{round(sniadata['t0'][indexx], 2)}\n" + \
                r"$x_0$ = " + f"{round(sniadata['x0'][indexx], 5)}\n" + \
                r"$x_1$ = " + f"{round(sniadata['x1'][indexx], 4)}\n" + \
                r"$c$   = " + f"{round(sniadata['c'][indexx], 4)}"

        Text2 = "Differences to the simulated parameters are:" +\
                "\n"+r"$\Delta (t_0)$ = " + f"{round(result_combined.parameters[1] - sniadata['t0'][indexx], 5)}"+" or " + str(round((result_combined.parameters[1]-sniadata[indexx:indexx+1]["t0"].iloc[0])/result_combined.errors["t0"],1))+r"$\cdot \Delta(t_0)$"+\
                "\n"+r"$\Delta (x_0)$ = " + f"{round(result_combined.parameters[2] - sniadata['x0'][indexx], 5)}"+" or " + str(round((result_combined.parameters[2]-sniadata[indexx:indexx+1]["x0"].iloc[0])/result_combined.errors["x0"],1))+r"$\cdot \Delta(x_0)$"+\
                "\n"+r"$\Delta (x_1)$ = " + f"{round(result_combined.parameters[3] - sniadata['x1'][indexx], 5)}"+" or " + str(round((result_combined.parameters[3]-sniadata[indexx:indexx+1]["x1"].iloc[0])/result_combined.errors["x1"],1))+r"$\cdot \Delta(x_1)$"+\
                "\n"+r"$\Delta (c)$   = " + f"{round(result_combined.parameters[4] - sniadata['c'][indexx], 5)}"+" or " + str(round((result_combined.parameters[4]-sniadata[indexx:indexx+1]["c"].iloc[0])/result_combined.errors["c"],1))+r"$\cdot \Delta(c)$"
        
        Text3 = "Differences to the simulated parameters are:" +\
                "\n"+r"$\Delta (t_0)$ = " + f"{round(result_ztf.parameters[1] - sniadata['t0'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[1]-sniadata[indexx:indexx+1]["t0"].iloc[0])/result_ztf.errors["t0"],1))+r"$\cdot \Delta(t_0)$"+\
                "\n"+r"$\Delta (x_0)$ = " + f"{round(result_ztf.parameters[2] - sniadata['x0'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[2]-sniadata[indexx:indexx+1]["x0"].iloc[0])/result_ztf.errors["x0"],1))+r"$\cdot \Delta(x_0)$"+\
                "\n"+r"$\Delta (x_1)$ = " + f"{round(result_ztf.parameters[3] - sniadata['x1'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[3]-sniadata[indexx:indexx+1]["x1"].iloc[0])/result_ztf.errors["x1"],1))+r"$\cdot \Delta(x_1)$"+\
                "\n"+r"$\Delta (c)$   = " + f"{round(result_ztf.parameters[4] - sniadata['c'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[4]-sniadata[indexx:indexx+1]["c"].iloc[0])/result_ztf.errors["c"],1))+r"$\cdot \Delta(c)$"
        

        # Plot the fitted light curve. ZTF and Ultrasat combined
        plt.figure()
        plt.figure(figsize=(7, 7))
        sncosmo.plot_lc(lc_table_combined, model=fitted_model_combined, errors=result_combined.errors, zp=28.1, zpsys='ab')
        plt.subplots_adjust(bottom=0.3)
        plt.figtext(0.1, 0.05, Text1, ha='left', va='center', fontsize=10)
        plt.figtext(0.5, 0.05, Text2, ha='left', va='center', fontsize=10)

        # Save the plot with the index in the filename
        filename = os.path.join(unique_dir_name, f"Fit_Index_{indexx}_combined.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
        
        # Plot the fitted light curve. Only ZTF.
        plt.figure()
        sncosmo.plot_lc(lc_table_ztf, model=fitted_model_ztf, errors=result_ztf.errors, zp=28.1, zpsys='ab')
        plt.subplots_adjust(bottom=0.3)
        plt.figtext(0.1, 0.05, Text1, ha='left', va='center', fontsize=10)
        plt.figtext(0.5, 0.05, Text3, ha='left', va='center', fontsize=10)

        # Save the plot with the index in the filename
        filename = os.path.join(unique_dir_name, f"Fit_Index_{indexx}_ztf.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()  # Close the plot to free memory


        # Append results to the table
        results_table.loc[k] = {
            "index": indexx,
            "z": sniadata["z"][indexx],
            "t0": round(sniadata["t0"][indexx], 1),
            "x0 real": sniadata["x0"][indexx],
            "x1 real": sniadata["x1"][indexx],
            "c real": sniadata["c"][indexx],
            "x0 combi": result_combined.parameters[2],
            "x0err combi": result_combined.errors["x0"],
            "x1 combi": round(result_combined.parameters[3], 3),
            "x1err combi": round(result_combined.errors["x1"], 4),
            "c combi": round(result_combined.parameters[4], 4),
            "cerr combi": round(result_combined.errors["c"], 5),
            "Suc combi": str(result_combined.success),
            "x0 ZTF": result_ztf.parameters[2],
            "x0err ZTF": result_ztf.errors["x0"],
            "x1 ZTF": round(result_ztf.parameters[3], 3),
            "x1err ZTF": round(result_ztf.errors["x1"], 4),
            "c ZTF": round(result_ztf.parameters[4], 4),
            "cerr ZTF": round(result_ztf.errors["c"], 5),
            "Suc ZTF": str(result_ztf.success),
        }

    # Filter out unsuccessful fits
    results_table.set_index("index", inplace=True)
    results_table = results_table[
        (results_table["Suc ZTF"] == "True") & (results_table["Suc combi"] == "True")
    ]

    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    base_filename = ("light_curve_fit_results_"+str(len(results_table))+"_entries.csv")
    unique_filename = generate_unique_filename(results_dir, base_filename)

    print(f"Saving results to {unique_filename}...")
    results_table.to_csv(unique_filename)

    return results_table

def create_comparison_table(fit_results_table):
    """
    Creates a comparison table with differences and deltas for fitted parameters.

    Args:
        fit_results_table (pandas.DataFrame): Table with fit results.
        save_results (bool): Whether to save the comparison table to the 'results' directory.

    Returns:
        pandas.DataFrame: Table of differences and deltas.
    """
    # Initialize the comparison table
    comparison_table = pd.DataFrame(columns=["z"])
    comparison_table["z"] = fit_results_table["z"]

    # Calculate differences and deltas for x0
    comparison_table["Del x0 combi"] = round(
        (fit_results_table["x0 combi"] - fit_results_table["x0 real"]) /
        fit_results_table["x0err combi"], 1
    )
    comparison_table["Del x0 ZTF"] = round(
        (fit_results_table["x0 ZTF"] - fit_results_table["x0 real"]) /
        fit_results_table["x0err ZTF"], 1
    )
    comparison_table["Dif x0 combi"] = abs(fit_results_table["x0 real"] - fit_results_table["x0 combi"])
    comparison_table["Dif x0 ZTF"] = abs(fit_results_table["x0 real"] - fit_results_table["x0 ZTF"])
    comparison_table["compare x0"] = comparison_table["Dif x0 ZTF"] - comparison_table["Dif x0 combi"]

    # Calculate differences and deltas for x1
    comparison_table["Del x1 combi"] = round(
        (fit_results_table["x1 combi"] - fit_results_table["x1 real"]) /
        fit_results_table["x1err combi"], 1
    )
    comparison_table["Del x1 ZTF"] = round(
        (fit_results_table["x1 ZTF"] - fit_results_table["x1 real"]) /
        fit_results_table["x1err ZTF"], 1
    )
    comparison_table["Dif x1 combi"] = abs(fit_results_table["x1 real"] - fit_results_table["x1 combi"])
    comparison_table["Dif x1 ZTF"] = abs(fit_results_table["x1 real"] - fit_results_table["x1 ZTF"])
    comparison_table["compare x1"] = comparison_table["Dif x1 ZTF"] - comparison_table["Dif x1 combi"]

    # Calculate differences and deltas for c
    comparison_table["Del c combi"] = round(
        (fit_results_table["c combi"] - fit_results_table["c real"]) /
        fit_results_table["cerr combi"], 1
    )
    comparison_table["Del c ZTF"] = round(
        (fit_results_table["c ZTF"] - fit_results_table["c real"]) /
        fit_results_table["cerr ZTF"], 1
    )
    comparison_table["Dif c combi"] = abs(fit_results_table["c real"] - fit_results_table["c combi"])
    comparison_table["Dif c ZTF"] = abs(fit_results_table["c real"] - fit_results_table["c ZTF"])
    comparison_table["compare c"] = comparison_table["Dif c ZTF"] - comparison_table["Dif c combi"]

    # Save results if required
    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    base_filename = "parameter_comparison_table_"+str(len(comparison_table))+"_entries.csv"
    unique_filename = generate_unique_filename(results_dir, base_filename)

    print(f"Saving comparison table to {unique_filename}...")
    comparison_table.to_csv(unique_filename, index=False)

    return comparison_table

#____________________________________________________
#DUST

def perform_light_curve_fit_dust(lc_table, model, redshift,t0=50000,x0=1,x1=0,ebv=0,rv=4):
    """
    Fits the light curve data using sncosmo, with error handling.

    Args:
        lc_table (astropy.table.Table): Light curve data table.
        model (sncosmo.Model): sncosmo model to fit.
        redshift (float): Redshift value.

    Returns:
        tuple: Fit result and fitted model, or (None, None) if fitting fails.
    """
    model.set(z=redshift,t0=t0,x0=x0,x1=x1,hostebv=ebv,hostr_v=rv,c=0)
    
    params = ["t0", "x0", "x1","hostebv"]
    bounds = {"x1": (-3, 3),"hostebv":(0,1)}

    try:
        # Perform the fit
        result, fitted_model = sncosmo.fit_lc(
            lc_table, model, params, modelcov=True, bounds=bounds
        )

        # Validate the result for NaN values
        if any(pd.isna(result.parameters)):
            raise RuntimeError(f"Fit result contains NaN values: {result.parameters}")

        return result, fitted_model

    except (RuntimeError, ValueError) as e:
        # Log the error and return None
        print(f"Error during fit: {e}")
        return None, None


def perform_light_curve_fit_dust_withrv(lc_table, model, redshift,t0=50000,x0=1,x1=0,ebv=0,rv=4):
    """
    Fits the light curve data using sncosmo, with error handling.

    Args:
        lc_table (astropy.table.Table): Light curve data table.
        model (sncosmo.Model): sncosmo model to fit.
        redshift (float): Redshift value.

    Returns:
        tuple: Fit result and fitted model, or (None, None) if fitting fails.
    """
    model.set(z=redshift,t0=t0,x0=x0,x1=x1,hostebv=ebv,hostr_v=rv,c=0)
    
    params = ["t0", "x0", "x1", "hostr_v","hostebv"]
    bounds = {"x1": (-3, 3), "hostr_v": (0, 6),"hostebv":(0,1)}

    try:
        # Perform the fit
        result, fitted_model = sncosmo.fit_lc(
            lc_table, model, params, modelcov=True, bounds=bounds
        )

        # Validate the result for NaN values
        if any(pd.isna(result.parameters)):
            raise RuntimeError(f"Fit result contains NaN values: {result.parameters}")

        return result, fitted_model

    except (RuntimeError, ValueError) as e:
        # Log the error and return None
        print(f"Error during fit: {e}")
        return None, None

def build_results_table_dust(combined_lightcurves,ztf_lightcurves, sniadata, sorted_indices,model,rv,folder="Results"):
    """
    Builds a results table with fit parameters for combined and ZTF datasets.

    Args:
        combined_lightcurves (skysurvey.DataSet): Combined dataset (ULTRASAT + ZTF).
        ztf_lightcurves (skysurvey.DataSet): ZTF dataset.
        sniadata (pandas.DataFrame): Original supernova data.
        sorted_indices (list): Sorted indices of combined observations.

    Returns:
        pandas.DataFrame: Table of fit results.
    """
    # Initialize the results table
    columns = [
        "index", "z", "t0", "x0 real", "x1 real", "rv real", "ebv real",
        "x0 combi", "x0err combi", "x1 combi", "x1err combi", "ebv combi", "ebverr combi",
        "x0 ZTF", "x0err ZTF", "x1 ZTF", "x1err ZTF", "ebv ZTF", "ebverr ZTF",
        "Suc combi", "Suc ZTF","chi2dof combi","chi2dof ZTF","chi2 combi","chi2 ZTF"
    ]
    results_table = pd.DataFrame(columns=columns)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    unique_dir_name ="Fits/"+folder+f"/lightcurves/dust_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(unique_dir_name, exist_ok=True)

    print(f"Created directory: {unique_dir_name}")

    for k, indexx in enumerate(tqdm(sorted_indices, desc="Fitting light curves")):
        
        # ZTF dataset fit
        df_ztf = ztf_lightcurves.data.loc[indexx]
        lc_table_ztf = prepare_light_curve_table(df_ztf)

        z = sniadata["z"][indexx]
        if rv is None:
            rv=sniadata['hostr_v'][indexx]
        result_ztf, fitted_model_ztf = perform_light_curve_fit_dust(lc_table_ztf, model, z,rv=rv)
        
        if result_ztf is None:
            print(f"Skipping index {indexx} due to failed ZTF fit.")
            continue
        if result_ztf.ndof==0:
            print(f"Skipping index {indexx} due to failed combined fit(ndof=0).")
            continue

        
        # Combined dataset fit
        df_combined = combined_lightcurves.data.loc[indexx]
        lc_table_combined = prepare_light_curve_table(df_combined)

        t0=result_ztf.parameters[1]
        x0=result_ztf.parameters[2]
        x1=result_ztf.parameters[3]
        ebv=result_ztf.parameters[5]
        result_combined, fitted_model_combined= perform_light_curve_fit_dust(lc_table_combined, model, z,t0,x0,x1,ebv,rv=rv)

        if result_combined is None:
            print(f"Skipping index {indexx} due to failed combined fit.")
            continue
        if result_combined.ndof==0:
            print(f"Skipping index {indexx} due to failed combined fit(ndof=0).")
            continue

        # Generate annotations for the plot
        Text1 = f"The simulated parameters are:\n" + \
                r"$t_0$ = " + f"{round(sniadata['t0'][indexx], 2)}\n" + \
                r"$x_0$ = " + f"{round(sniadata['x0'][indexx], 5)}\n" + \
                r"$x_1$ = " + f"{round(sniadata['x1'][indexx], 4)}\n" + \
                r"$E(B-V)$ = " + f"{round(sniadata['hostebv'][indexx], 4)}\n\n"
        
        Text2 = "Differences to the simulated parameters are:" +\
                "\n"+r"$\Delta (t_0)$ = " + f"{round(result_combined.parameters[1] - sniadata['t0'][indexx], 5)}"+" or " + str(round((result_combined.parameters[1]-sniadata[indexx:indexx+1]["t0"].iloc[0])/result_combined.errors["t0"],1))+r"$\cdot \Delta(t_0)$"+\
                "\n"+r"$\Delta (x_0)$ = " + f"{round(result_combined.parameters[2] - sniadata['x0'][indexx], 5)}"+" or " + str(round((result_combined.parameters[2]-sniadata[indexx:indexx+1]["x0"].iloc[0])/result_combined.errors["x0"],1))+r"$\cdot \Delta(x_0)$"+\
                "\n"+r"$\Delta (x_1)$ = " + f"{round(result_combined.parameters[3] - sniadata['x1'][indexx], 5)}"+" or " + str(round((result_combined.parameters[3]-sniadata[indexx:indexx+1]["x1"].iloc[0])/result_combined.errors["x1"],1))+r"$\cdot \Delta(x_1)$"+\
                "\n"+r"$\Delta (E(B-V))$ = " + f"{round(result_combined.parameters[5] - sniadata['hostebv'][indexx], 5)}"+" or " + str(round((result_combined.parameters[5]-sniadata[indexx:indexx+1]["hostebv"].iloc[0])/result_combined.errors["hostebv"],1))+r"$\cdot \Delta(E(B-V))$"+\
                "\n"+r"$chisq/dof$ = " + f"{round(result_combined.chisq/result_ztf.ndof, 5)}"+r" or $chisq calc$ = "+f"{round(sncosmo.chisq(lc_table_combined,fitted_model_combined,modelcov=False), 5)}"
        
        Text3 = "Differences to the simulated parameters are:" +\
                "\n"+r"$\Delta (t_0)$ = " + f"{round(result_ztf.parameters[1] - sniadata['t0'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[1]-sniadata[indexx:indexx+1]["t0"].iloc[0])/result_ztf.errors["t0"],1))+r"$\cdot \Delta(t_0)$"+\
                "\n"+r"$\Delta (x_0)$ = " + f"{round(result_ztf.parameters[2] - sniadata['x0'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[2]-sniadata[indexx:indexx+1]["x0"].iloc[0])/result_ztf.errors["x0"],1))+r"$\cdot \Delta(x_0)$"+\
                "\n"+r"$\Delta (x_1)$ = " + f"{round(result_ztf.parameters[3] - sniadata['x1'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[3]-sniadata[indexx:indexx+1]["x1"].iloc[0])/result_ztf.errors["x1"],1))+r"$\cdot \Delta(x_1)$"+\
                "\n"+r"$\Delta (E(B-V))$ = " + f"{round(result_ztf.parameters[5] - sniadata['hostebv'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[5]-sniadata[indexx:indexx+1]["hostebv"].iloc[0])/result_ztf.errors["hostebv"],1))+r"$\cdot \Delta(E(B-V))$"+\
                "\n"+r"$chisq/dof$ = " + f"{round(result_ztf.chisq/result_ztf.ndof, 5)}"+r" or $chisq calc$ = "+f"{round(sncosmo.chisq(lc_table_ztf,fitted_model_ztf,modelcov=False), 5)}"
        

        # Plot the fitted light curve. ZTF and Ultrasat combined
        plt.figure()
        sncosmo.plot_lc(lc_table_combined, model=fitted_model_combined, errors=result_combined.errors, zp=28.1, zpsys='ab',xfigsize=13)
        plt.subplots_adjust(bottom=0.3)
        plt.figtext(0.1, 0.05, Text1, ha='left', va='center', fontsize=15)
        plt.figtext(0.5, 0.05, Text2, ha='left', va='center', fontsize=15)

        # Save the plot with the index in the filename
        filename = os.path.join(unique_dir_name, f"Fit_Index_{indexx}_combined.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
        
        # Plot the fitted light curve. Only ZTF.
        plt.figure()
        sncosmo.plot_lc(lc_table_ztf, model=fitted_model_ztf, errors=result_ztf.errors, zp=28.1, zpsys='ab',xfigsize=13)
        plt.subplots_adjust(bottom=0.3)
        plt.figtext(0.1, 0.05, Text1, ha='left', va='center', fontsize=15)
        plt.figtext(0.5, 0.05, Text3, ha='left', va='center', fontsize=15)

        # Save the plot with the index in the filename
        filename = os.path.join(unique_dir_name, f"Fit_Index_{indexx}_ztf.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()  # Close the plot to free memory


        # Append results to the table
        results_table.loc[k] = {
            "index": indexx,
            "z": sniadata["z"][indexx],
            "t0": round(sniadata["t0"][indexx], 1),
            "x0 real": sniadata["x0"][indexx],
            "x1 real": sniadata["x1"][indexx],
            "rv real": sniadata["hostr_v"][indexx],
            "ebv real": sniadata["hostebv"][indexx],
            "x0 combi": result_combined.parameters[2],
            "x0err combi": result_combined.errors["x0"],
            "x1 combi": round(result_combined.parameters[3], 3),
            "x1err combi": round(result_combined.errors["x1"], 4),
            "ebv combi": round(result_combined.parameters[5], 4),
            "ebverr combi": round(result_combined.errors["hostebv"], 5),
            "Suc combi": str(result_combined.success),
            "chi2dof combi": sncosmo.chisq(lc_table_combined,fitted_model_combined,modelcov=False)/result_combined.ndof,
            "chi2 combi": sncosmo.chisq(lc_table_combined,fitted_model_combined,modelcov=False),
            "x0 ZTF": result_ztf.parameters[2],
            "x0err ZTF": result_ztf.errors["x0"],
            "x1 ZTF": round(result_ztf.parameters[3], 3),
            "x1err ZTF": round(result_ztf.errors["x1"], 4),
            "ebv ZTF": round(result_ztf.parameters[5], 4),
            "ebverr ZTF": round(result_ztf.errors["hostebv"], 5),
            "Suc ZTF": str(result_ztf.success),
            "chi2dof ZTF": sncosmo.chisq(lc_table_ztf,fitted_model_ztf,modelcov=False)/result_ztf.ndof,
            "chi2 ZTF": sncosmo.chisq(lc_table_ztf,fitted_model_ztf,modelcov=False)
        }

    # Filter out unsuccessful fits
    results_table.set_index("index", inplace=True)
    #results_table = results_table[
    #    (results_table["Suc ZTF"] == "True") & (results_table["Suc combi"] == "True")
    #]

    results_dir = "Fits/"+folder
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    base_filename = ("light_curve_dust_fit_results_rv"+str(rv)+"_"+str(len(results_table))+"_entries.csv")
    unique_filename = generate_unique_filename(results_dir, base_filename)

    print(f"Saving results to {unique_filename}...")
    results_table.to_csv(unique_filename)

    return results_table

def build_results_table_dust_withrv(combined_lightcurves,ztf_lightcurves, sniadata, sorted_indices,model):
    """
    Builds a results table with fit parameters for combined and ZTF datasets.

    Args:
        combined_lightcurves (skysurvey.DataSet): Combined dataset (ULTRASAT + ZTF).
        ztf_lightcurves (skysurvey.DataSet): ZTF dataset.
        sniadata (pandas.DataFrame): Original supernova data.
        sorted_indices (list): Sorted indices of combined observations.

    Returns:
        pandas.DataFrame: Table of fit results.
    """
    # Initialize the results table
    columns = [
        "index", "z", "t0", "x0 real", "x1 real", "rv real", "ebv real",
        "x0 combi", "x0err combi", "x1 combi", "x1err combi", "rv combi", "rverr combi", "ebv combi", "ebverr combi",
        "x0 ZTF", "x0err ZTF", "x1 ZTF", "x1err ZTF", "rv ZTF", "rverr ZTF", "ebv ZTF", "ebverr ZTF",
        "Suc combi", "Suc ZTF"
    ]
    results_table = pd.DataFrame(columns=columns)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    unique_dir_name = f"Fits/dust_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(unique_dir_name, exist_ok=True)

    print(f"Created directory: {unique_dir_name}")

    for k, indexx in enumerate(tqdm(sorted_indices, desc="Fitting light curves")):
        
        # ZTF dataset fit
        df_ztf = ztf_lightcurves.data.loc[indexx]
        lc_table_ztf = prepare_light_curve_table(df_ztf)

        z = sniadata["z"][indexx]
        result_ztf, fitted_model_ztf = perform_light_curve_fit_dust(lc_table_ztf, model, z)
        
        if result_ztf is None:
            print(f"Skipping index {indexx} due to failed ZTF fit.")
            continue

        
        # Combined dataset fit
        df_combined = combined_lightcurves.data.loc[indexx]
        lc_table_combined = prepare_light_curve_table(df_combined)

        t0=result_ztf.parameters[1]
        x0=result_ztf.parameters[2]
        x1=result_ztf.parameters[3]
        ebv=result_ztf.parameters[5]
        rv=result_ztf.parameters[6]
        result_combined, fitted_model_combined= perform_light_curve_fit_dust(lc_table_combined, model, z,t0,x0,x1,ebv,rv)

        if result_combined is None:
            print(f"Skipping index {indexx} due to failed combined fit.")
            continue

        # Generate annotations for the plot
        Text1 = f"The simulated parameters are:\n" + \
                r"$t_0$ = " + f"{round(sniadata['t0'][indexx], 2)}\n" + \
                r"$x_0$ = " + f"{round(sniadata['x0'][indexx], 5)}\n" + \
                r"$x_1$ = " + f"{round(sniadata['x1'][indexx], 4)}\n" + \
                r"$E(B-V)$ = " + f"{round(sniadata['hostebv'][indexx], 4)}\n" + \
                r"$R_V$   = " + f"{round(sniadata['hostr_v'][indexx], 4)}"

        Text2 = "Differences to the simulated parameters are:" +\
                "\n"+r"$\Delta (t_0)$ = " + f"{round(result_combined.parameters[1] - sniadata['t0'][indexx], 5)}"+" or " + str(round((result_combined.parameters[1]-sniadata[indexx:indexx+1]["t0"].iloc[0])/result_combined.errors["t0"],1))+r"$\cdot \Delta(t_0)$"+\
                "\n"+r"$\Delta (x_0)$ = " + f"{round(result_combined.parameters[2] - sniadata['x0'][indexx], 5)}"+" or " + str(round((result_combined.parameters[2]-sniadata[indexx:indexx+1]["x0"].iloc[0])/result_combined.errors["x0"],1))+r"$\cdot \Delta(x_0)$"+\
                "\n"+r"$\Delta (x_1)$ = " + f"{round(result_combined.parameters[3] - sniadata['x1'][indexx], 5)}"+" or " + str(round((result_combined.parameters[3]-sniadata[indexx:indexx+1]["x1"].iloc[0])/result_combined.errors["x1"],1))+r"$\cdot \Delta(x_1)$"+\
                "\n"+r"$\Delta (E(B-V))$ = " + f"{round(result_combined.parameters[5] - sniadata['hostebv'][indexx], 5)}"+" or " + str(round((result_combined.parameters[5]-sniadata[indexx:indexx+1]["hostebv"].iloc[0])/result_combined.errors["hostebv"],1))+r"$\cdot \Delta(E(B-V))$"+\
                "\n"+r"$\Delta (R_V)$   = " + f"{round(result_combined.parameters[6] - sniadata['hostr_v'][indexx], 5)}"+" or " + str(round((result_combined.parameters[6]-sniadata[indexx:indexx+1]["hostr_v"].iloc[0])/result_combined.errors["hostr_v"],1))+r"$\cdot \Delta(R_V)$"
        
        Text3 = "Differences to the simulated parameters are:" +\
                "\n"+r"$\Delta (t_0)$ = " + f"{round(result_ztf.parameters[1] - sniadata['t0'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[1]-sniadata[indexx:indexx+1]["t0"].iloc[0])/result_ztf.errors["t0"],1))+r"$\cdot \Delta(t_0)$"+\
                "\n"+r"$\Delta (x_0)$ = " + f"{round(result_ztf.parameters[2] - sniadata['x0'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[2]-sniadata[indexx:indexx+1]["x0"].iloc[0])/result_ztf.errors["x0"],1))+r"$\cdot \Delta(x_0)$"+\
                "\n"+r"$\Delta (x_1)$ = " + f"{round(result_ztf.parameters[3] - sniadata['x1'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[3]-sniadata[indexx:indexx+1]["x1"].iloc[0])/result_ztf.errors["x1"],1))+r"$\cdot \Delta(x_1)$"+\
                "\n"+r"$\Delta (E(B-V))$ = " + f"{round(result_ztf.parameters[5] - sniadata['hostebv'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[5]-sniadata[indexx:indexx+1]["hostebv"].iloc[0])/result_ztf.errors["hostebv"],1))+r"$\cdot \Delta(E(B-V))$"+\
                "\n"+r"$\Delta (R_V)$   = " + f"{round(result_ztf.parameters[6] - sniadata['hostr_v'][indexx], 5)}"+" or " + str(round((result_ztf.parameters[6]-sniadata[indexx:indexx+1]["hostr_v"].iloc[0])/result_ztf.errors["hostr_v"],1))+r"$\cdot \Delta(R_V)$"
        

        # Plot the fitted light curve. ZTF and Ultrasat combined
        plt.figure()
        sncosmo.plot_lc(lc_table_combined, model=fitted_model_combined, errors=result_combined.errors, zp=28.1, zpsys='ab')
        plt.subplots_adjust(bottom=0.3)
        plt.figtext(0.1, 0.05, Text1, ha='left', va='center', fontsize=10)
        plt.figtext(0.5, 0.05, Text2, ha='left', va='center', fontsize=10)

        # Save the plot with the index in the filename
        filename = os.path.join(unique_dir_name, f"Fit_Index_{indexx}_combined.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
        
        # Plot the fitted light curve. Only ZTF.
        plt.figure()
        sncosmo.plot_lc(lc_table_ztf, model=fitted_model_ztf, errors=result_ztf.errors, zp=28.1, zpsys='ab')
        plt.subplots_adjust(bottom=0.3)
        plt.figtext(0.1, 0.05, Text1, ha='left', va='center', fontsize=10)
        plt.figtext(0.5, 0.05, Text3, ha='left', va='center', fontsize=10)

        # Save the plot with the index in the filename
        filename = os.path.join(unique_dir_name, f"Fit_Index_{indexx}_ztf.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()  # Close the plot to free memory


        # Append results to the table
        results_table.loc[k] = {
            "index": indexx,
            "z": sniadata["z"][indexx],
            "t0": round(sniadata["t0"][indexx], 1),
            "x0 real": sniadata["x0"][indexx],
            "x1 real": sniadata["x1"][indexx],
            "rv real": sniadata["hostr_v"][indexx],
            "ebv real": sniadata["hostebv"][indexx],
            "x0 combi": result_combined.parameters[2],
            "x0err combi": result_combined.errors["x0"],
            "x1 combi": round(result_combined.parameters[3], 3),
            "x1err combi": round(result_combined.errors["x1"], 4),
            "rv combi": round(result_combined.parameters[6], 4),
            "rverr combi": round(result_combined.errors["hostr_v"], 5),
            "ebv combi": round(result_combined.parameters[5], 4),
            "ebverr combi": round(result_combined.errors["hostebv"], 5),
            "Suc combi": str(result_combined.success),
            "x0 ZTF": result_ztf.parameters[2],
            "x0err ZTF": result_ztf.errors["x0"],
            "x1 ZTF": round(result_ztf.parameters[3], 3),
            "x1err ZTF": round(result_ztf.errors["x1"], 4),
            "rv ZTF": round(result_ztf.parameters[6], 4),
            "rverr ZTF": round(result_ztf.errors["hostr_v"], 5),
            "ebv ZTF": round(result_ztf.parameters[5], 4),
            "ebverr ZTF": round(result_ztf.errors["hostebv"], 5),
            "Suc ZTF": str(result_ztf.success),
        }

    # Filter out unsuccessful fits
    results_table.set_index("index", inplace=True)
    results_table = results_table[
        (results_table["Suc ZTF"] == "True") & (results_table["Suc combi"] == "True")
    ]

    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    base_filename = ("light_curve_dust_fit_results_"+str(len(results_table))+"_entries.csv")
    unique_filename = generate_unique_filename(results_dir, base_filename)

    print(f"Saving results to {unique_filename}...")
    results_table.to_csv(unique_filename)

    return results_table

def create_comparison_table_dust_withrv(fit_results_table):
    """
    Creates a comparison table with differences and deltas for fitted parameters.

    Args:
        fit_results_table (pandas.DataFrame): Table with fit results.

    Returns:
        pandas.DataFrame: Table of differences and deltas.
    """
    # Initialize the comparison table
    comparison_table = pd.DataFrame(columns=["z"])
    comparison_table["z"] = fit_results_table["z"]

    # Calculate differences and deltas for x0
    comparison_table["Del x0 combi"] = round(
        (fit_results_table["x0 combi"] - fit_results_table["x0 real"]) /
        fit_results_table["x0err combi"], 1
    )
    comparison_table["Del x0 ZTF"] = round(
        (fit_results_table["x0 ZTF"] - fit_results_table["x0 real"]) /
        fit_results_table["x0err ZTF"], 1
    )
    comparison_table["Dif x0 combi"] = abs(fit_results_table["x0 real"] - fit_results_table["x0 combi"])
    comparison_table["Dif x0 ZTF"] = abs(fit_results_table["x0 real"] - fit_results_table["x0 ZTF"])
    comparison_table["compare x0"] = comparison_table["Dif x0 ZTF"] - comparison_table["Dif x0 combi"]

    # Calculate differences and deltas for x1
    comparison_table["Del x1 combi"] = round(
        (fit_results_table["x1 combi"] - fit_results_table["x1 real"]) /
        fit_results_table["x1err combi"], 1
    )
    comparison_table["Del x1 ZTF"] = round(
        (fit_results_table["x1 ZTF"] - fit_results_table["x1 real"]) /
        fit_results_table["x1err ZTF"], 1
    )
    comparison_table["Dif x1 combi"] = abs(fit_results_table["x1 real"] - fit_results_table["x1 combi"])
    comparison_table["Dif x1 ZTF"] = abs(fit_results_table["x1 real"] - fit_results_table["x1 ZTF"])
    comparison_table["compare x1"] = comparison_table["Dif x1 ZTF"] - comparison_table["Dif x1 combi"]

    # Calculate differences and deltas for rv
    comparison_table["Del rv combi"] = round(
        (fit_results_table["rv combi"] - fit_results_table["rv real"]) /
        fit_results_table["rverr combi"], 1
    )
    comparison_table["Del rv ZTF"] = round(
        (fit_results_table["rv ZTF"] - fit_results_table["rv real"]) /
        fit_results_table["rverr ZTF"], 1
    )
    comparison_table["Dif rv combi"] = abs(fit_results_table["rv real"] - fit_results_table["rv combi"])
    comparison_table["Dif rv ZTF"] = abs(fit_results_table["rv real"] - fit_results_table["rv ZTF"])
    comparison_table["compare rv"] = comparison_table["Dif rv ZTF"] - comparison_table["Dif rv combi"]

    # Calculate differences and deltas for ebv
    comparison_table["Del ebv combi"] = round(
        (fit_results_table["ebv combi"] - fit_results_table["ebv real"]) /
        fit_results_table["ebverr combi"], 1
    )
    comparison_table["Del ebv ZTF"] = round(
        (fit_results_table["ebv ZTF"] - fit_results_table["ebv real"]) /
        fit_results_table["ebverr ZTF"], 1
    )
    comparison_table["Dif ebv combi"] = abs(fit_results_table["ebv real"] - fit_results_table["ebv combi"])
    comparison_table["Dif ebv ZTF"] = abs(fit_results_table["ebv real"] - fit_results_table["ebv ZTF"])
    comparison_table["compare ebv"] = comparison_table["Dif ebv ZTF"] - comparison_table["Dif ebv combi"]

    # Save results
    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    base_filename = "parameter_dust_comparison_table_" + str(len(comparison_table)) + "_entries.csv"
    unique_filename = generate_unique_filename(results_dir, base_filename)

    print(f"Saving comparison table to {unique_filename}...")
    comparison_table.to_csv(unique_filename, index=False)

    return comparison_table

def create_comparison_table_dust(fit_results_table):
    """
    Creates a comparison table with differences and deltas for fitted parameters.

    Args:
        fit_results_table (pandas.DataFrame): Table with fit results.

    Returns:
        pandas.DataFrame: Table of differences and deltas.
    """
    # Initialize the comparison table
    comparison_table = pd.DataFrame(columns=["z"])
    comparison_table["z"] = fit_results_table["z"]

    # Calculate differences and deltas for x0
    comparison_table["Del x0 combi"] = round(
        (fit_results_table["x0 combi"] - fit_results_table["x0 real"]) /
        fit_results_table["x0err combi"], 1
    )
    comparison_table["Del x0 ZTF"] = round(
        (fit_results_table["x0 ZTF"] - fit_results_table["x0 real"]) /
        fit_results_table["x0err ZTF"], 1
    )
    comparison_table["Dif x0 combi"] = abs(fit_results_table["x0 real"] - fit_results_table["x0 combi"])
    comparison_table["Dif x0 ZTF"] = abs(fit_results_table["x0 real"] - fit_results_table["x0 ZTF"])
    comparison_table["compare x0"] = comparison_table["Dif x0 ZTF"] - comparison_table["Dif x0 combi"]

    # Calculate differences and deltas for x1
    comparison_table["Del x1 combi"] = round(
        (fit_results_table["x1 combi"] - fit_results_table["x1 real"]) /
        fit_results_table["x1err combi"], 1
    )
    comparison_table["Del x1 ZTF"] = round(
        (fit_results_table["x1 ZTF"] - fit_results_table["x1 real"]) /
        fit_results_table["x1err ZTF"], 1
    )
    comparison_table["Dif x1 combi"] = abs(fit_results_table["x1 real"] - fit_results_table["x1 combi"])
    comparison_table["Dif x1 ZTF"] = abs(fit_results_table["x1 real"] - fit_results_table["x1 ZTF"])
    comparison_table["compare x1"] = comparison_table["Dif x1 ZTF"] - comparison_table["Dif x1 combi"]


    # Calculate differences and deltas for ebv
    comparison_table["Del ebv combi"] = round(
        (fit_results_table["ebv combi"] - fit_results_table["ebv real"]) /
        fit_results_table["ebverr combi"], 1
    )
    comparison_table["Del ebv ZTF"] = round(
        (fit_results_table["ebv ZTF"] - fit_results_table["ebv real"]) /
        fit_results_table["ebverr ZTF"], 1
    )
    comparison_table["Dif ebv combi"] = abs(fit_results_table["ebv real"] - fit_results_table["ebv combi"])
    comparison_table["Dif ebv ZTF"] = abs(fit_results_table["ebv real"] - fit_results_table["ebv ZTF"])
    comparison_table["compare ebv"] = comparison_table["Dif ebv ZTF"] - comparison_table["Dif ebv combi"]

    # Save results
    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    base_filename = "parameter_dust_comparison_table_" + str(len(comparison_table)) + "_entries.csv"
    unique_filename = generate_unique_filename(results_dir, base_filename)

    print(f"Saving comparison table to {unique_filename}...")
    comparison_table.to_csv(unique_filename, index=False)

    return comparison_table

