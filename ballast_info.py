from erddapy import ERDDAP
from pathlib import Path
import numpy as np
import pandas as pd
import voto_erddap_utils as utils
import matplotlib.pyplot as plt


def get_glider_dataset_ids():
    e = ERDDAP(server="https://erddap.observations.voiceoftheocean.org/erddap",
               protocol="tabledap")

    e.dataset_id = "allDatasets"
    df_datasets = e.to_pandas()['datasetID']
    df_glider_datasets = df_datasets[df_datasets.str.contains("_S")]
    return df_glider_datasets


def select_datasets(platform_serial=None, mission_num=None, data_type='nrt'):
    '''
    inputs:
    glider_serial= xx
    data_type= 'nrt' or 'delayed'
    '''

    df_datasets = get_glider_dataset_ids()
    if platform_serial:
        df_datasets = df_datasets[df_datasets.str.contains(f"{platform_serial}")]
    if mission_num:
        df_datasets = df_datasets[df_datasets.str.contains(f"M{mission_num}")]

    if data_type == 'nrt':
        df_datasets = df_datasets[df_datasets.str.contains(f"nrt")]
    elif data_type == 'delayed':
        df_datasets = df_datasets[df_datasets.str.contains(f"delayed")]
    return df_datasets.values


def ballast_info(glider_datasets, threshold=420, noise_threshold=5):
    '''
    threshold= xxx value in ml. Number of times glider pumps crosses positively
    noise_threshold= xx ml, minimum difference between two consequetive points in ballast position to be accounted for in calculating total active pumping during mission. To not account for noise in ballast pumping calculations.
    '''
    
    ds_dict = utils.download_glider_dataset(glider_datasets, nrt_only=False, variables=(['ballast_pos', 'time', 'dive_num', 'ballast_cmd', 'nav_state', 'security_level', 'voltage', 'longitude', 'latitude', 'depth']))
    #Max, min, and total pumping + max depth values for the full mission
    max_ballast=[]
    min_ballast=[]
    total_dives = []
    total_pump= []
    max_depth=[]
    
    #Average variables based on each dive in every mission. Range is average range per mission. Upper limit is based off nav state 117
    avg_pump_max =[]
    avg_pump_min =[]
    avg_pump_range=[]
    std_pump_max=[]
    std_pump_min=[]
    
    #For the amount of points above threshold which is specified below, not sure if this information is actually usable
    high_volume=[]
    percent_high_volume=[]
    
    #For mission name and basin name
    ds_name=[]
    basin=[]
    mission_no=[]
    platform_serial=[]
    
    #Which extreme value to check high pumping volumes and create array to how many times crossed over for each mission
    cross_over_threshold=[]
    threshold_value=[]

    starts = []
    ends = []
    durations = []
    start_lon = []
    start_lat = []
    end_lon = []
    end_lat = []
    start_voltage = []
    end_voltage = []


    for name, ds in ds_dict.items():
        starts.append(ds.time.values[0])
        ends.append(ds.time.values[-1])
        durations.append((ds.time.values[-1] - ds.time.values[0]).astype('timedelta64[D]').astype(int))
        start_lon.append(ds.longitude.values[0])
        start_lat.append(ds.latitude.values[0])
        end_lon.append(ds.longitude.values[-1])
        end_lat.append(ds.latitude.values[-1])
        start_voltage.append(np.nanmedian(ds.voltage[:10]))
        end_voltage.append(np.nanmedian(ds.voltage[-10:]))

        #Max and min ballast values per mission and total dives + total volume pumped for mission
        max_ballast.append(np.nanmax(ds.ballast_pos))
        min_ballast.append(np.nanmin(ds.ballast_pos))
        total_dives.append(ds['dive_num'].values.max())

        max_depth.append(int(np.nanmax(np.abs(ds.depth))))
        
        if np.diff(ds.time).mean()/ np.timedelta64(1, 's')<0.8:
            test=ds.thin({"time": 70}) #.isel(obs=ds.dive_num==528)
        else:
            test=ds.thin({"time": 15}) #.isel(obs=ds.dive_num==528)
        ballast = test.isel(time=np.isfinite(test.ballast_pos.values))
        ballast= ballast.isel(time=np.isfinite(ballast.ballast_cmd.values))
        
        pos=ballast.ballast_pos.values
        cmd = ballast.ballast_cmd.values
        
        pos_pre = pos.copy()[:-1]
        pos_post = pos.copy()[1:]
        pos_diff = pos_post - pos_pre
        
        cmd_pre = cmd.copy()[:-1]
        cmd_post = cmd.copy()[1:]
        cmd_diff = cmd_post - cmd_pre

        pos_pump_vol=np.where((pos_diff)>noise_threshold, pos_diff, 0)
        
        total_pump.append(int(np.sum(pos_pump_vol)))

        #Average max and min pumping
        ballast_top_range=[]
        ballast_low_range=[]

        # crossover
        ballast = ds.ballast_pos.values
        ballast = ballast[~np.isnan(ballast)]
        ballast_pre = ballast.copy()[:-1]
        ballast_post = ballast.copy()[1:]
        ballast_pre[ballast_pre > threshold] = np.nan
        ballast_post[ballast_post < threshold] = np.nan
        ballast_diff = ballast_post - ballast_pre
        cross_over = sum(ballast_diff > 0)

        #Create array of divenumbers in order for nrt datasets as all dives are not available. Remove duplicates and sort them
        dive_num=np.sort(np.array(list(set(ds.dive_num.values))))
        for i in dive_num:
            ds_dive=ds.sel(time=ds.dive_num==i)

            ds_nav_state=ds_dive.sel(time=ds_dive.nav_state==117) #Select data for navstate 117 only (Glider going up) to avoid ballast surfacing values
        
            if len(ds_dive.sel(time=ds_dive.security_level>0).time) >0: #If there are alarms at this dive, do not include this in avg pumping max or min range
                ballast_top_range.append(np.nan)
                ballast_low_range.append(np.nan)
                
            elif len(ds_nav_state.time)>0:
                ballast_top_range.append(int(ds_nav_state.ballast_pos.values.max()))
                ballast_low_range.append(int(ds_dive.ballast_pos.values.min()))
            else:
                ballast_top_range.append(np.nan) #As there are so few dives with no navstate 117, add nans for these TO BE REVISED
                ballast_low_range.append(int(ds_dive.ballast_pos.values.min()))

        #Calculate average pumping range
        pump_range= np.array(ballast_top_range) - np.array(ballast_low_range)
        avg_pump_max.append(int(np.nanmean(ballast_top_range)))
        avg_pump_min.append(int(np.nanmean(ballast_low_range)))
        avg_pump_range.append(int(np.nanmean(pump_range)))
    
        std_pump_max.append(int(np.nanstd(ballast_top_range)))
        std_pump_min.append(int(np.nanstd(ballast_low_range)))
    
        #How often is the volume over threshold
        cross_over_threshold.append(int(cross_over)) #How many times over the whole mission it crossed over threshold value
        threshold_value.append(threshold)
        
        # Add string categories: Basin and Mission name & number
        try:
            basin.append(ds.basin)
        except:
            basin.append("")
        ds_name.append(name)
        mission_no.append(ds.deployment_id)
        if "platform_serial" in ds.keys():
            platform_serial.append(ds.platform_serial)
        else:
            platform_serial.append(ds.glider_serial)


    #Make all values integers
    total_dives=np.array(total_dives).astype(int)
    max_ballast=np.array(max_ballast).astype(int)
    min_ballast=np.array(min_ballast).astype(int)
    total_dives=np.array(total_dives).astype(int)
    avg_pump_max=np.array(avg_pump_max).astype(int)
    avg_pump_min=np.array(avg_pump_min).astype(int)
    high_volume=np.array(high_volume).astype(int)
    starts = np.array(starts)
    ends = np.array(ends)
    durations = np.array(durations).astype(int)
    start_lon = np.array(start_lon).astype(float)
    start_lat = np.array(start_lat).astype(float)
    end_lon = np.array(end_lon).astype(float)
    end_lat = np.array(end_lat).astype(float)
    start_voltage = np.array(start_voltage).astype(float)
    end_voltage = np.array(end_voltage).astype(float)

    df_pumps = pd.DataFrame({'datasetID': ds_name, 'deployment_id': mission_no, 'platform_serial': platform_serial,
                             'total dives': total_dives, 'max depth (m)': max_depth, 'max ballast (ml)': max_ballast, 'min ballast (ml)': min_ballast, 'avg max pumping value (ml)': avg_pump_max,
                             'std_max': std_pump_max, 'std_min': std_pump_min , 'avg min pumping value (ml)': avg_pump_min, 'avg pumping range (ml)': avg_pump_range, 'total active pumping (ml)': total_pump, 
                             'times crossing over '+str(threshold)+' ml': cross_over_threshold, 'basin': basin, 'threshold': threshold_value,
                             'Deployment time': starts, 'Recovery time': ends, 'Mission duration (days)': durations, 'Deployment longitude': start_lon,
                             'Deployment latitude': start_lat, 'Recovery longitude': end_lon, 'Recovery latitude': end_lat,
                             'Deployment voltage (V)': start_voltage, 'Recovery voltage (v)': end_voltage})
    #'datapoints over '+str(threshold)+' ml': high_volume, 'Ballast positions over '+str(threshold)+' ml (%)' :percent_high_volume,
    return df_pumps


def ballast_plots(df_pumps):
    '''
    input: table generated by ballast_info
    
    Generates figure of avg. max value of pumping range baset on nav-state 177 and avg min values. Both with standarddeviations
    Generates plot of avg. pumping range and twinaxis with number of dives in mission and times ballast volume crossed from below threshold to above
    
    '''

    threshold=df_pumps['threshold'][0]
    
    figs=2
    var_1='avg max pumping value (ml)'
    var_2= 'avg min pumping value (ml)'
    
    fig, ax= plt.subplots(figs,1, figsize=[10,8], sharex=True, constrained_layout=True)
    
    #Max and min pumping with std
    
    ax[0].plot(df_pumps['mission no'], df_pumps[var_1], label='Avg. max volume', color='b')
    ax[0].plot(df_pumps['mission no'], df_pumps[var_2], label='Avg. min volume', color='orange')
    
    ax[0].fill_between(df_pumps['mission no'],df_pumps[var_1], df_pumps[var_1]+df_pumps['std_max'], alpha=.3, color='b')
    ax[0].fill_between(df_pumps['mission no'],df_pumps[var_1], df_pumps[var_1]-df_pumps['std_max'], alpha=.3, color='b')
    
    ax[0].fill_between(df_pumps['mission no'],df_pumps[var_2], df_pumps[var_2]+df_pumps['std_min'], alpha=.3, color='orange')
    ax[0].fill_between(df_pumps['mission no'],df_pumps[var_2], df_pumps[var_2]-df_pumps['std_min'], alpha=.3, color='orange')
    
    ax[0].set_ylabel('Ballast volume (ml)')
    ax[0].set_title('SEA0'+df_pumps['glider serial'][0]+' Pumping patterns per mission')
    
    #Pumping range + times crossing over threshold value
    var_1='avg pumping range (ml)'
    var_2='times crossing over '+str(threshold)+' ml'
    var_3='total dives'
    
    ax[1].plot(df_pumps['mission no'], df_pumps[var_1], label='Avg. pumping range', color='k')
    ax[1].set_ylabel('Ballast volume (ml)')
    #ax[1].plot(df_pumps['mission no'], df_pumps['max ballast (ml)'], label='Max ballast volume')
    
    ax2=ax[1].twinx()
    ax2.plot(df_pumps['mission no'], df_pumps[var_2], label='Times crossing over threshold volume')
    ax2.plot(df_pumps['mission no'], df_pumps[var_3], label='Total dives in mission')
    
    ax2.set_ylabel('no. of dives')
    
    for i in np.arange(figs):
        ax[i].grid(alpha=0.5)
        ax[i].legend(loc=2)
    
    ax2.legend()
    
    ax[1].set_xlabel('Mission number')


if __name__ == '__main__':
    df = ballast_info(["nrt_SEA069_M37"])
    from metadata_tables import write_csv
    outfile = Path("output/ballast.csv")
    all_nrt = select_datasets(mission_num=None, platform_serial=None, data_type='delayed')
    for ds_id in all_nrt:
        to_download = [ds_id]
        if outfile.exists():
            df = pd.read_csv(outfile, sep=';')
            to_download = set(to_download).difference(df['datasetID'].values)
        else:
            df = pd.DataFrame()
        if len(to_download) == 0:
            print("No datasets found matching supplied arguments")
        else:
            df_add = ballast_info(to_download)
            df = pd.concat((df, df_add))
            df = df.groupby('datasetID').first()
            write_csv(df, 'ballast')
            print(f"added dataset {ds_id}")
