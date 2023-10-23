import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from influxdb_client import InfluxDBClient
import copy


token ='bm_-UvKDmNeKQKiaTERDKuv8hCx1tKyg4Yc91_PWrpBi62_kkp6xxOlM05vT2DBdEOMbLwEvDpoMDiyzoqYB7A=='
org = "aiot"
url = "http://123.123.117.126:8086"


server_names = [
    'fusion-101',
    'fusion-102',
    'fusion-103',
    'fusion-104',
    'fusion-105',
    'fusion-106',
    'fusion-107',
    'fusion-108',
    'fusion-109',
    'fusion-110',
    'fusion-111',
    'intel-102',
    'intel-103',
    'intel-104',
    'intel-105',
    'intel-106',
    'intel-107',
    'intel-108',
    'intel-109',
    'intel-110',
    'intel-111',
]

rack_names = {
    'rack-1': [
        'fusion-101',
        'fusion-102',
        'fusion-103',
        'fusion-104',
        'fusion-105',
    ],
    'rack-2': [
        'fusion-106',
        'fusion-107',
        'fusion-108',
        'fusion-109',
        'fusion-110',
        'fusion-111',
    ],
    'rack-3': [
        'intel-102',
        'intel-103',
        'intel-104',
        'intel-105',
        'intel-106',
        'intel-107',
        'intel-108',
        'intel-109',
        'intel-110',
        'intel-111',
    ]
}

MAX_CPU_ID_INTEL = 88
MAX_CPU_ID_FUSION = 112



def local_to_utc(time_str):
    from datetime import datetime, timedelta
    utc_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
    time_difference = timedelta(hours=-8)
    desired_time = utc_time + time_difference
    desired_time_str = desired_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    return desired_time_str


def utc_to_local(time_str):
    from datetime import datetime, timedelta
    utc_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
    time_difference = timedelta(hours=8)
    desired_time = utc_time + time_difference
    desired_time_str = desired_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    return desired_time_str


def get_relative_hours(df):
    """
    :param df: pandas data frame with time and values
    :return:
    """
    series = copy.deepcopy(df)
    series['time'] = pd.to_datetime(series['time'])
    series.set_index('time', inplace=True)
    relative_time = (series.index - series.index[0]).total_seconds()
    return relative_time/3600


def get_energy(df):
    series = copy.deepcopy(df)
    series['time'] = pd.to_datetime(series['time'])
    series.set_index('time', inplace=True)
    relative_time = (series.index - series.index[0]).total_seconds()
    energy = scipy.integrate.simps(series['values'], relative_time)
    return energy/3600 # return energy in kwh



def init_query_api(url, token):
    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()
    return query_api


def get_package_cpu_util(time_start, time_end, package_id, host):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)

    query_content = f"""
    from(bucket: "server")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "cpu")
      |> filter(fn: (r) => r["_field"] == "usage_active")
      |> filter(fn: (r) => r["physical_id"] == "{package_id}")
      |> filter(fn: (r) => r["host"] == "{host}")
      |> group(columns: ["_time"])
      |> mean(column: "_value")
    """
    return query_content


def get_core_cpu_freq(time_start, time_end, core_id, host, freq_field='cpu_busy_frequency_mhz'):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)

    query_content = f"""
    from(bucket: "server")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "powerstat_core")
      |> filter(fn: (r) => r["host"] == {host})
      |> filter(fn: (r) => r["_field"] == {freq_field})
      |> filter(fn: (r) => r["cpu_id"] == {core_id})
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """
    return query_content


def get_core_cpu_util(time_start, time_end, core_id, host):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    query_content = f"""
    from(bucket: "server")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "cpu")
      |> filter(fn: (r) => r["host"] == "{host}")
      |> filter(fn: (r) => r["_field"] == "usage_active")
      |> filter(fn: (r) => r["cpu"] == "cpu{core_id}")
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """

    return query_content


def get_core_cpu_temp(time_start, time_end, core_id, host):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    query_content = f"""
    from(bucket: "server")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "powerstat_core")
      |> filter(fn: (r) => r["_field"] == "cpu_temperature_celsius")
      |> filter(fn: (r) => r["cpu_id"] == {core_id})
      |> filter(fn: (r) => r["host"] == {host})
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """
    return query_content


def get_server_cpu_util(time_start, time_end, host):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    query_content = f"""
    from(bucket: "server")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "cpu")
      |> filter(fn: (r) => r["_field"] == "usage_active")
      |> filter(fn: (r) => r["cpu"] == "cpu-total")
      |> filter(fn: (r) => r["host"] == {host})
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """

    return query_content


def get_host_power(time_start, time_end, host):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    query_content = f"""
    from(bucket: "server")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "power")
      |> filter(fn: (r) => r["_field"] == "value")
      |> filter(fn: (r) => r["host"] == "{host}")
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """
    return query_content


def get_ac_set_temp(time_start, time_end, model_type="mpc_with_gcn"):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    query_content = f"""
    from(bucket: "AI-Model")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "model-action")
      |> filter(fn: (r) => r["_field"] == "Model-回风温度设定点")
      |> filter(fn: (r) => r["model_type"] == "{model_type}")
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """
    return query_content


def get_cluster_cpu_util(time_start, time_end, server_type=None):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    if server_type is None:
        query_content = f"""
        from(bucket: "server")
          |> range(start: {time_start}, stop: {time_end})
          |> filter(fn: (r) => r["_measurement"] == "cpu")
          |> filter(fn: (r) => r["_field"] == "usage_active")
          |> filter(fn: (r) => r["cpu"] == "cpu-total")
          |> group(columns: ["_time"])
          |> mean(column: "_value")
        """
    else:
        query_content = f"""
        from(bucket: "server")
          |> range(start: {time_start}, stop: {time_end})
          |> filter(fn: (r) => r["_measurement"] == "cpu")
          |> filter(fn: (r) => r["_field"] == "usage_active")
          |> filter(fn: (r) => r["cpu"] == "cpu-total")
          |> filter(fn: (r) => r["environment"] == "{server_type}")
          |> group(columns: ["_time"])
          |> mean(column: "_value")
        """
    return query_content


def get_cluster_avg_power(time_start, time_end, server_type=None):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    if server_type is None:
        query_content = f"""
        from(bucket: "server")
          |> range(start: {time_start}, stop: {time_end})
          |> filter(fn: (r) => r["_measurement"] == "power")
          |> filter(fn: (r) => r["_field"] == "value")
          |> group(columns: ["_time"])
          |> mean(column: "_value")
        """
    else:
        assert (server_type == 'intel' or server_type == 'fusion')
        query_content = f"""
        from(bucket: "server")
          |> range(start: {time_start}, stop: {time_end})
          |> filter(fn: (r) => r["_measurement"] == "power")
          |> filter(fn: (r) => r["_field"] == "value")
          |> filter(fn: (r) => r["environment"] == "{server_type}")
          |> group(columns: ["_time"])
          |> mean(column: "_value")
        """
    return query_content


def execute_query(query_api, query_content):
    timestamps = []
    values = []

    tables = query_api.query(query_content, org=org)

    for table in tables:
        for record in table.records:
            timestamps.append(record['_time'])
            values.append(record['_value'])

    df = pd.DataFrame(data={'time': timestamps, 'values': values})
    return df


def parallel_read_server_power(time_start, time_end, server_type=None):
    def _wrapper(url, token, time_start, time_end, host):
        try:
            query_api = init_query_api(url, token)
            query_content = get_host_power(time_start, time_end, host)
            df = execute_query(query_api, query_content)
            return df
        except Exception as e:
            print(time_start, time_end, host)
            print('error', e)
            return None

    if server_type is None:
        parameter_tuples = [
            (url, token, time_start, time_end, name) for name in server_names
        ]
    else:
        assert server_type == 'intel' or server_type == 'fusion'

        parameter_tuples = [
            (url, token, time_start, time_end, name) for name in server_names if server_type in name
        ]

    num_threads = os.cpu_count()-1
    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_wrapper, *ps) for ps in parameter_tuples]
        results = [future.result() for future in futures]
    end = time.time()
    return {name: result for name, result in zip(server_names, results)}


def get_ac_power(time_start, time_end):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    query_content = f"""
    from(bucket: "AC-power")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "modbus")
      |> filter(fn: (r) => r["_field"] == "0xA71A-有功总功率（乘变比）")
      |> filter(fn: (r) => r["host"] == "intel-112")
      |> filter(fn: (r) => r["location"] == "GDS")
      |> filter(fn: (r) => r["machine"] == "8001-列间空调")
      |> filter(fn: (r) => r["name"] == "AC-power")
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """
    return query_content


def get_return_air_temp(time_start, time_end, sensor_idx=1):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    query_content = f"""
    from(bucket: "AC-8001")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "modbus")
      |> filter(fn: (r) => r["_field"] == "0x10{sensor_idx-1}-回风温度{sensor_idx}")
      |> filter(fn: (r) => r["host"] == "intel-112")
      |> filter(fn: (r) => r["name"] == "AC")
      |> filter(fn: (r) => r["slave_id"] == "1")
      |> filter(fn: (r) => r["type"] == "holding_register")
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """

    return query_content


def get_avg_return_air_temp(time_start, time_end):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    query_content = f"""
    from(bucket: "AC-8001")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "modbus")
      |> filter(fn: (r) => r["_field"] == "0x100-回风温度1" or r["_field"] == "0x101-回风温度2")
      |> filter(fn: (r) => r["host"] == "intel-112")
      |> filter(fn: (r) => r["name"] == "AC")
      |> filter(fn: (r) => r["slave_id"] == "1")
      |> group(columns: ["_time"])
      |> mean(column: "_value")
    """
    return query_content


def get_manual_set_return_air_temp(time_start, time_end):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    query_content = f"""
    from(bucket: "AC-8001")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "modbus")
      |> filter(fn: (r) => r["_field"] == "0x316-回风温度设定点")
      |> filter(fn: (r) => r["host"] == "intel-112")
      |> filter(fn: (r) => r["name"] == "AC")
      |> filter(fn: (r) => r["slave_id"] == "1")
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """
    return query_content

    
def get_cold_asile_temp(time_start, time_end, host='fusion-101'):
    time_start = local_to_utc(time_start)
    time_end = local_to_utc(time_end)
    
    host_dns = {
        'fusion-101': '192.168.2.101',
        'fusion-102': '192.168.2.102',
        'fusion-103': '192.168.2.103',
        'fusion-104': '192.168.2.104',
        'fusion-105': '192.168.2.105',
        'fusion-106': '192.168.2.106',
        'fusion-107': '192.168.2.107',
        'fusion-108': '192.168.2.108',
        'fusion-109': '192.168.2.109',
        'fusion-110': '192.168.2.110',
        'fusion-111': '192.168.2.111',
    }

    url = host_dns[host]

    query_content = f"""
    from(bucket: "server-ibmc")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "http")
      |> filter(fn: (r) => r["_field"] == "InletTemperature")
      |> filter(fn: (r) => r["environment"] == "fusion")
      |> filter(fn: (r) => r["host"] == "intel-112")
      |> filter(fn: (r) => r["interval"] == "10s")
      |> filter(fn: (r) => r["url"] == "https://{url}/redfish/v1/SystemOverview")
      |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """

    return query_content



if __name__ == "__main__":
    ts = [
        # ('2023-09-25T22:50:00Z', '2023-09-26T10:50:00Z', 'server_baseline+cooling_baseline'),
        # ('2023-09-02T15:00:00Z', '2023-09-03T03:00:00Z', 'server_baseline+cooling_baseline'),
        # ('2023-09-02T01:25:00Z', '2023-09-02T13:25:00Z', 'server_opt+cooling_opt'),


        ('2023-09-25T02:30:00Z', '2023-09-25T14:30:00Z', 'server_baseline+cooling_opt'),
        ('2023-09-26T03:00:00Z', '2023-09-26T15:00:00Z', 'server_opt+cooling_opt'),

    ]

    for time_start, time_end, label in ts:
        print(time_start, time_end, label)
        query_api = init_query_api(url, token)

        qc = get_cluster_cpu_util(time_start, time_end)
        avg_cluster_util = execute_query(query_api, qc)

        qc = get_cluster_cpu_util(time_start, time_end, server_type='intel')
        avg_cluster_util_intel = execute_query(query_api, qc)

        qc = get_cluster_cpu_util(time_start, time_end, server_type='fusion')
        avg_cluster_util_fusion = execute_query(query_api, qc)

        print('util')
        print('total', avg_cluster_util['values'].mean())
        print('intel', avg_cluster_util_intel['values'].mean())
        print('fusion', avg_cluster_util_fusion['values'].mean())

        results = parallel_read_server_power(time_start, time_end)

        rack_1_power = {name: power_df['values'].values.mean() for name, power_df in results.items() if
                        name in rack_names['rack-1']}
        rack_2_power = {name: power_df['values'].values.mean() for name, power_df in results.items() if
                        name in rack_names['rack-2']}
        rack_3_power = {name: power_df['values'].values.mean() for name, power_df in results.items() if
                        name in rack_names['rack-3']}

        print('rack_1 pow avg:', sum(list(rack_1_power.values()))/len(rack_1_power))
        print('rack_2 pow avg:', sum(list(rack_2_power.values())) / len(rack_2_power))
        print('fusion pow avg:', (5*sum(list(rack_1_power.values()))/len(rack_1_power)+6*sum(list(rack_2_power.values())) / len(rack_2_power))/11)
        print('rack_3 (intel) pow avg:', sum(list(rack_3_power.values())) / len(rack_3_power))
        print('total:', sum([df['values'].values.mean() for df in results.values()]) / len(results))













