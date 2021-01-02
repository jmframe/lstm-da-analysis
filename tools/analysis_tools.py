"""
Tools for analyzing the LSTM Data Assimilation results across the CAMELS basins
"""

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import metrics
import signatures
import corrstats #https://github.com/psinger/CorrelationStats/blob/master/corrstats.py

import scipy
import scipy.stats as st
import statsmodels as sm

# For the regression.
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def DIFFS(diffs_inputs):
    (basin_list, 
          ensemble_metrics,
          control, 
          test,
          use_metrics, 
          optimal,
          percent) = diffs_inputs
    diffs = np.full([len(basin_list),len(use_metrics)], np.nan)
    percent_diffs = np.full([len(basin_list),len(use_metrics)], np.nan)
    for m, metric in enumerate(use_metrics):
        if optimal[m] == 1:
            diffs[:,m] = ensemble_metrics[metric, test] - \
                               ensemble_metrics[metric, control]
            percent_diffs[:,m] = diffs[:,m] / np.abs(ensemble_metrics[metric, control])
        elif optimal[m] == 0:
            diffs[:,m] = np.abs(ensemble_metrics[metric,control]) - \
                                np.abs(ensemble_metrics[metric, test])
            percent_diffs[:,m] = diffs[:,m] / np.abs(ensemble_metrics[metric, control])
    if percent:
        return percent_diffs
    else:
        return diffs

def PLOT_MAPS(diffs_inputs, diffs, plot_inputs): 
    (basin_list, 
         ensemble_metrics,
         control, 
         test,
         use_metrics, 
         optimal,
         percent) = diffs_inputs
    (display_bounds_from_control,
         display_colors_from_control, 
         disp_bounds,
         diff_bounds,
         plot_lons, 
         plot_lats, 
         use_metric_names) = plot_inputs
    
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(len(use_metrics),3)

    for m, metric in enumerate(use_metrics):

        ax0 = fig.add_subplot(gs[m,:2])
        im = ax0.scatter(plot_lons, plot_lats,
                        c=diffs[:,m],
                        s=20,
                        cmap=display_colors_from_control[control],
                        vmin=-diff_bounds[m], vmax=diff_bounds[m])
        ax0.set_title(use_metric_names[m])
        clims = im.get_clim()

        # colorbar
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax1 = fig.add_subplot(gs[m,2])    
        ax1.plot(disp_bounds[m], disp_bounds[m], 'k--', lw=0.6)
        for b, basin in enumerate(basin_list):
            basin_color = im.to_rgba(diffs[b,m])
            ax1.scatter(ensemble_metrics[metric, control][b], 
                        ensemble_metrics[metric, test][b],
                        s=5,
                        color=im.to_rgba(diffs[b,m]))
        ax1.set_xlabel(control)
        ax1.set_ylabel(test)
        ax1.set_xlim(display_bounds_from_control[control][m])
        ax1.set_ylim(display_bounds_from_control[test][m])
        ax1.grid()

    plt.tight_layout()
    plt.show()
    plt.close()

def COUNT(diffs_inputs, diffs, 
          threshold=1, verbose=True):
    (basin_list, 
          ensemble_metrics,
          control, 
          test,
          use_metrics, 
          optimal,
          percent) = diffs_inputs
    for m, metric in enumerate(use_metrics):
        count_improved = 0
        count_detriment = 0
        count_total = 0
        for i, ival in enumerate(diffs[:,m]):
            if ensemble_metrics[metric, 'base_model'][i] < threshold:
                count_total += 1
                if ival > 0:
                    count_improved+= 1
                if ival < 0:
                    count_detriment+= 1
        if verbose:
            print(metric)
            print('Number of improved basins = {}, {:.2f}%'.format(count_improved, 
                                                                   count_improved/count_total))
            print('Number of detrimented basins = {}, {:.2f}%'.format(count_detriment, 
                                                                      count_detriment/count_total))
        else:
            if metric == 'NSE':
                return count_improved/count_total

# Correlation of base model performance with model variations
def CORR(corr_inputs):
    (ensemble_metrics, control, model_types, metrics) = corr_inputs
    corrs = np.full([len(model_types), len(metrics)], np.nan)
    for t, test in enumerate(model_types):
        for m, metric in enumerate(metrics):
            test_m = np.array(ensemble_metrics[metric, test])
            control_m = np.array(ensemble_metrics[metric, control])
            corrs[t, m] = st.stats.pearsonr(control_m, test_m)[0]
    return pd.DataFrame(data=corrs, index=model_types, columns=metrics)


# Calculate the flow catagories of observations
def FLOW_CAT(basin_list, flow_categories, observations, date_range):
    flow_dates = {fc:{b:[] for b in basin_list} for fc in flow_categories}

    for ib, b in enumerate(basin_list):  
        df = observations[b]

        for i_date, today in enumerate(date_range):
            # For a baseline also evaluate the whole record
            flow_dates['all'][b].append(today)

            # Rising and falling limbs
            if i_date>0:
                yesterday =  date_range[i_date-1]
                diff_1d = df.loc[today]-df.loc[yesterday]
                if diff_1d < 0:
                    flow_dates['fall'][b].append(today)
                if diff_1d > 0:
                    flow_dates['rise'][b].append(today)

            # Above or below Median
            if df.loc[today] > np.median(df):
                flow_dates['above_mid'][b].append(today)
            else:
                flow_dates['below_mid'][b].append(today)

            # Above or below Mean
            if df.loc[today] > np.mean(df):
                flow_dates['above_mean'][b].append(today)
            else:
                flow_dates['below_mean'][b].append(today)

            # Above or below 20th/80th percentile
            if df.loc[today] > np.percentile(df, 80):
                flow_dates['above_80'][b].append(today)
            elif df.loc[today] < np.percentile(df, 20):
                flow_dates['below_20'][b].append(today) 

    return flow_dates

def PLOT_FREQ(basin_list, ensemble_metrics, metrics, model_types, met_lims):
    yvalues = list(range(len(basin_list)))
    for i, _ in enumerate(yvalues):
        yvalues[i] = yvalues[i]/len(yvalues)
    
    for m, metric in enumerate(metrics):
        x = [np.array(ensemble_metrics[metric, model]) for model in model_types]
        
        for imod, X in enumerate(x):
            plotdata = X
            plotdata = np.sort(plotdata[~pd.isnull(plotdata)])
            plt.plot(plotdata,  yvalues[:len(plotdata)], label=model_types[imod], lw=2)
            plt.title(metric)
            plt.ylabel('Frequency')
        plt.xlim(met_lims[metric])
        plt.grid()
        plt.legend()
        plt.show()
        plt.close()



def FLOW_METRICS(observations,
                 simulations,
                 calc_metrics,
                 basin_list,
                 model_types,
                 flow_categories, flow_dates):
    nmets = len(calc_metrics)
    nbasins = len(basin_list)
    nmodels = len(model_types)
    obs_model_types = ['obs'] + list(model_types)
    
    met_mat = {fc:np.full([nbasins,nmodels,nmets],np.nan) for fc in flow_categories}

    flow_mat = {fc:{mt:[] for mt in obs_model_types} for fc in flow_categories}

    did_not_calculate = []

    # Loop through the basins and calculate the performance metric.
    for ib, b in enumerate(basin_list):   #['02216180']):
        # Keep a record of any infinities that come out of performance metric calculations.
        basin_inf = {fc:[] for fc in model_types}

        # Loop through the flow categories, and calculate the performance metrics.
        # Any flow category that is split up by percentile 
        for fc in flow_categories:
            
            flow_set = flow_dates[fc][b]
            
            #Check for NaNs
            # Start by checking if there is a nan somewhere in the obs or sims
            if observations[b].isnull().values.any():
                flow_set = [fs for fs in flow_set if not pd.isna(observations[b].loc[fs])]
            if simulations[b].isnull().values.any():
                # If there is a NaN somewhere, then check each value and replace 
                for imt, model_type in enumerate(model_types):
                    flow_set = [fs for fs in flow_set if not pd.isna(simulations[b][model_type].loc[fs])]
            
            xobs = xr.DataArray(observations[b].loc[flow_set]).rename({'dim_0': 'date'})
            flow_mat[fc]['obs'].extend(list(observations[b].loc[flow_set]))

            # Do not calculate the metrics on the basins that have a mean close to zero.
            # This will lead to -inf or very low (-100xxx) Nash-Sutcliffe values.
            if np.mean(observations[b].loc[flow_set]) > 0:
                is_calc_metrics = True
            else:
                is_calc_metrics = False
                did_not_calculate.append([b,fc])

            # 1) combined flows from all basins into one dictionary, for overall metrics
            # 2) Calculate the metrics for each model type.
            for imt, model_type in enumerate(model_types):
                
                # Combine all the basins so the metrics can be calculated across them all at once.
                # Model must have at least some criteria of values to calculate the metrics
                if len(list(simulations[b][model_type].loc[flow_set])) >10:
                    flow_mat[fc][model_type].extend(list(simulations[b][model_type].loc[flow_set]))
                else:
                    did_not_calculate.append([b,fc, model_type])
                    
                # Now just put the curent basin's flow into an xarray to get the metrics for this basin alone.
                xsim = xr.DataArray(simulations[b][model_type].loc[flow_set]).rename({'dim_0': 'date'})
                
                if is_calc_metrics:
                    met_dict = metrics.calculate_metrics(xobs, xsim, calc_metrics)
                    for imet, met_name in enumerate(calc_metrics):
                        if np.isinf(met_dict[met_name]):
                                basin_inf[model_type].append([b, met_name])
                        met_mat[fc][ib,imt,imet] = met_dict[met_name]
                        
    sites_that_did_not_calculate = []
    for i in did_not_calculate:
        if i[0] not in sites_that_did_not_calculate:
            sites_that_did_not_calculate.append(i[0])
    print('Some/all flow regime metrics did not calculate at these sites:', sites_that_did_not_calculate)

    return flow_mat, met_mat
