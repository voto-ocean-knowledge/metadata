import numpy as np
import pathlib
import xarray as xr
import pandas as pd
import requests
from erddapy import ERDDAP
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict

cache_dir = pathlib.Path('voto_erddap_data_cache')


def init_erddap(protocol="tabledap"):
    # Setup initial ERDDAP connection
    e = ERDDAP(
        server="https://erddap.observations.voiceoftheocean.org/erddap",
        protocol=protocol,
    )
    return e


def _clean_dims(ds):
    if "timeseries" in ds.sizes.keys() and "obs" in ds.sizes.keys():
        ds = ds.drop_dims("timeseries")
    if "obs" in ds.sizes.keys():
        ds = ds.swap_dims({"obs": "time"})
    return ds


def _get_meta_griddap(dataset_id):
    e = init_erddap(protocol="griddap")
    e.dataset_id = dataset_id
    e.griddap_initialize()
    time = pd.read_csv(f"https://erddap.observations.voiceoftheocean.org/erddap/griddap/{dataset_id}.csvp?time")[
        "time (UTC)"].values
    e.constraints['time>='] = str(time[-20])
    ds = e.to_xarray()
    attrs = ds.attrs
    # Clean up formatting of variables list
    if "variables" in attrs.keys():
        if "\n" in attrs["variables"]:
            attrs["variables"] = attrs["variables"].split("\n")
    # evaluate dictionaries
    for key, val in attrs.items():
        if type(val) == str:
            if "{" in val:
                attrs[key] = eval(val)
    if "basin" not in attrs.keys():
        attrs["basin"] = ""
    return attrs


def _etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(_etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def get_meta(dataset_id, protocol="tabledap"):
    if "adcp" in dataset_id or protocol=="griddap":
        # Cannot use to_ncCF with griddap
        return _get_meta_griddap(dataset_id)
    e = init_erddap(protocol=protocol)
    e.dataset_id = dataset_id
    meta = e.to_ncCF(timeout=300)
    attrs = {}
    for key_name in dir(meta):
        if key_name[0] != "_":
            attrs[key_name] = meta.__getattribute__(key_name)
    # Clean up formatting of variables list
    if "variables" in attrs.keys():
        if type(attrs["variables"]) is dict:
            attrs["variables"] = list(attrs["variables"].keys())
    # evaluate dictionaries
    for key, val in attrs.items():
        if type(val) == str:
            if "{" in val:
                attrs[key] = eval(val)
    if "basin" not in attrs.keys():
        attrs["basin"] = ""
    return attrs


def date_from_iso(dataset_id):
    req = requests.get(f'https://erddap.observations.voiceoftheocean.org/erddap/tabledap/{dataset_id}.iso19115')
    with open('iso.xml', 'w') as file:
        file.write(req.text)
    tree = ET.parse('iso.xml')
    root = tree.getroot()
    ddict = _etree_to_dict(root)
    id_info = ddict[list(ddict.keys())[0]]['{http://www.isotc211.org/2005/gmd}identificationInfo'][0]
    citation_info = id_info['{http://www.isotc211.org/2005/gmd}MD_DataIdentification']['{http://www.isotc211.org/2005/gmd}citation']['{http://www.isotc211.org/2005/gmd}CI_Citation']
    date_info = citation_info['{http://www.isotc211.org/2005/gmd}date'][0]['{http://www.isotc211.org/2005/gmd}CI_Date']
    datestamp = date_info['{http://www.isotc211.org/2005/gmd}date']['{http://www.isotc211.org/2005/gco}Date']

    return datestamp


def find_glider_datasets(nrt_only=True):
    """
    Find the dataset IDs of all glider datasets on the VOTO ERDDAP server
    nrt_only: if True, only returns nrt datasets
    """
    e = init_erddap()

    # Fetch dataset list
    e.response = "csv"
    e.dataset_id = "allDatasets"
    df_datasets = e.to_pandas()

    datasets = df_datasets.datasetID
    # Select only nrt datasets
    if nrt_only:
        datasets = datasets[datasets.str[:3] == "nrt"]
    return datasets.values


def add_profile_time(ds):
    profile_num = ds.pressure.copy()
    profile_num.attrs = {}
    profile_num.name = "profile_num"
    profile_num[:] = 0
    start = 0
    for i, prof_index in enumerate(ds.profile_index):
        rowsize = ds.rowSize.values[i]
        profile_num[start:start + rowsize] = prof_index
        start = start + rowsize
    ds["profile_num"] = profile_num
    profile_time = ds.time.values.copy()
    profile_index = ds.profile_num
    for profile in np.unique(profile_index.values):
        mean_time = ds.time[profile_index == profile].mean().values
        new_times = np.empty((len(ds.time[profile_index == profile])), dtype='datetime64[ns]')
        new_times[:] = mean_time
        profile_time[profile_index == profile] = new_times
    profile_time_var = ds.time.copy()
    profile_time_var.values = profile_time
    profile_time_var.name = "profile_mean_time"
    ds["profile_mean_time"] = profile_time_var
    ds = _clean_dims(ds)
    return ds


def _cached_dataset_exists(ds_id, request):
    """
    Returns True if all the following conditions are met:
    1. A dataset corresponding to ds_id exists in the cache
    2. The cached dataset was downloaded with the same request
    3. The dataset has not been updated on the VOTO ERDDAP since it was last downloaded
    Otherwise, returns False
    """
    if not cache_dir.exists():
        print(f"Creating directory to cache datasets at {cache_dir.absolute()}")
        pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return False
    dataset_nc = cache_dir / f"{ds_id}.nc"
    if not dataset_nc.exists():
        print(f"Dataset {ds_id} not found in cache")
        return False
    try:
        df = pd.read_csv(cache_dir / "cache_info.csv", index_col=0)
    except:
        print(f"no cache records file found")
        return False

    if ds_id in df.index:
        stats = df.loc[ds_id]
    else:
        print(f"no cache record found for {ds_id}")
        return False
    if not stats["request"] == request:
        print(f"request has changed for {ds_id}")
        return False

    nc_time = pd.to_datetime(stats["date_created"][:-6])
    try:
        created_date = date_from_iso(ds_id)
        erddap_time = pd.to_datetime(created_date)
    except:
        print(f"Dataset {ds_id} date updated check failed. Will re-download")
        return False
    if nc_time < erddap_time:
        print(f"Dataset {ds_id} has been updated on ERDDAP")
        return False

    return True


def _update_stats(ds_id, request):
    """
    Update the stats for a specified dataset
    """
    dataset_nc = cache_dir / f"{ds_id}.nc"
    ds = xr.open_dataset(dataset_nc)
    try:
        df = pd.read_csv(cache_dir / "cache_info.csv", index_col=0)
    except:
        df = pd.DataFrame()

    nc_time = ds.attrs["date_created"]
    new_stats = {"request": request, "date_created": pd.to_datetime(nc_time)}
    if ds_id in df.index:
        df.loc[ds_id] = new_stats
    else:
        new_row = pd.DataFrame(new_stats, index=[ds_id])
        df = pd.concat((df, new_row))
    df = df.sort_index()
    df.to_csv(cache_dir / "cache_info.csv")
    ds.close()


def add_adcp_data(ds):
    dataset_id = ds.attrs["dataset_id"]
    parts = dataset_id.split("_")
    adcp_id = f"adcp_{parts[1]}_{parts[2]}"
    cached_ds = _cached_dataset_exists(adcp_id, "adcp")
    dataset_nc = cache_dir / f"{adcp_id}.nc"
    if cached_ds:
        print(f"Found {dataset_nc}. Loading from disk")
        adcp = xr.open_dataset(dataset_nc)
    else:
        dataset_ids = find_glider_datasets(nrt_only=False)
        if adcp_id not in dataset_ids:
            print(f"Requested ADCP dataset {adcp_id} does not exist on server! Returning standard dataset")
            return ds
        print(f"Downloading {adcp_id}")
        e = ERDDAP(server="https://erddap.observations.voiceoftheocean.org/erddap/", protocol="griddap", )
        e.dataset_id = adcp_id
        e.griddap_initialize()
        time = pd.read_csv(f"https://erddap.observations.voiceoftheocean.org/erddap/griddap/{adcp_id}.csvp?time")[
            "time (UTC)"].values
        e.constraints['time>='] = str(time[0])
        adcp = e.to_xarray()
        adcp = adcp.sortby("time")
        adcp.to_netcdf(dataset_nc)
        _update_stats(adcp_id, "adcp")
    ds = _clean_dims(ds)

    if parts[0] == "nrt":
        print("WARNING: matching adcp data to nearest nrt timestamp. Potential missmatch of ~ 15 seconds. "
              "Use delayed mode data for closer timestamp match")
        adcp = adcp.reindex(time=ds.time, method="nearest")
    for var_name in list(adcp):
        ds[{var_name}] = adcp[var_name]
    adcp_attrs_dict = {i: j for i, j in adcp.attrs.items() if i not in ds.attrs}
    ds.attrs["adcp_attributes"] = str(adcp_attrs_dict)
    return ds


def download_glider_dataset(dataset_ids, variables=(), constraints={}, nrt_only=False, delayed_only=False,
                            cache_datasets=True, adcp=False):
    """
    Download datasets from the VOTO server using a supplied list of dataset IDs.
    dataset_ids: list of datasetIDs present on the VOTO ERDDAP
    variables: data variables to download. If left empty, will download all variables
    """
    if nrt_only and delayed_only:
        raise ValueError("Cannot set both nrt_only and delayed_only")
    if nrt_only:
        ids_to_download = []
        for name in dataset_ids:
            if "nrt" in name:
                ids_to_download.append(name)
            else:
                print(f"{name} is not nrt. Ignoring")
    elif delayed_only:
        ids_to_download = []
        for name in dataset_ids:
            if "delayed" in name:
                ids_to_download.append(name)
            else:
                print(f"{name} is not delayed. Ignoring")
    else:
        ids_to_download = dataset_ids

    e = init_erddap()
    # Specify variables of interest if supplied
    if variables:
        e.variables = variables
    if constraints:
        e.constraints = constraints

    # Download each dataset as xarray
    glider_datasets = {}
    for ds_name in tqdm(ids_to_download):
        if cache_datasets and "delayed" in ds_name:
            e.dataset_id = ds_name
            request = e.get_download_url()
            cached_dataset = _cached_dataset_exists(ds_name, request)
            dataset_nc = cache_dir / f"{ds_name}.nc"
            if cached_dataset:
                print(f"Found {ds_name} in {cache_dir}. Loading from disk")
                ds = xr.open_dataset(dataset_nc)
                if adcp:
                    ds = add_adcp_data(ds)
                glider_datasets[ds_name] = ds
            else:
                print(f"Downloading {ds_name}")
                try:
                    ds = e.to_xarray(requests_kwargs={"timeout": 300})
                except BaseException as ex:
                    print(ex)
                    continue
                ds = _clean_dims(ds)
                print(f"Writing {dataset_nc}")
                ds = ds.sortby("time")
                ds.to_netcdf(dataset_nc)
                if adcp:
                    ds = add_adcp_data(ds)
                glider_datasets[ds_name] = ds
                _update_stats(ds_name, request)
        else:
            print(f"Downloading {ds_name}")
            e.dataset_id = ds_name
            try:
                ds = e.to_xarray()
            except BaseException as ex:
                print(ex)
                continue
            ds = _clean_dims(ds)
            if adcp:
                ds = add_adcp_data(ds)
            ds = ds.sortby("time")
            glider_datasets[ds_name] = ds
    return glider_datasets


