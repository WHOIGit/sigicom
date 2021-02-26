#!/usr/bin/env python
import os
import argparse
import json, csv, time
from functools import reduce
import datetime as dt
import requests
import pandas as pd
from numpy import isnan, invert

path = os.path.join

API_URL='https://{SUBDOM}.infralogin.com/api/v1'
SEARCH_RESULTS_FILE = '{ROOT}/query_results.{ID}.json'


def open_authfile(fname):
    with open(fname) as f:
        content = f.read().split()
    subdomain,user_id,api_key = content
    global API_URL
    API_URL = API_URL.format(SUBDOM=subdomain)
    return ('user:'+user_id, api_key)


def make_argparser():
    parser = argparse.ArgumentParser(description='This is a tool for searching for, downloading, and processing transient event data from Sigicom\'s v1 API. All dates and times are in UTC.')

    subparsers = parser.add_subparsers(dest='subcommand', required=True, help='These subcommands are mutually exclusive and intended to be used in successive order. Use "--help" to learn more about each subcommand.')
    exp_parser = subparsers.add_parser('AVAIL', help='Displays available Projects and Nodes based on your authentication.')
    query_parser = subparsers.add_parser('QUERY', help='Query Sigicom for data according to some criteria. Outputs a search-results file to output ROOT, a Search ID to be used by "GET", and some result metrics')
    fetch_parser = subparsers.add_parser('GET', help='Download transients from Sigicom and process events. Processed files are always GPS-time adjusted.')

    # Authentication
    parser.add_argument('authfile', help='Required Authentication File containing USER and api TOKEN key')

    # EXPLORE
    # nothing more need be added here.

    # aside: output default params
    default_root = path('output', '{ID}')
    outdir_kwargs = dict(dest='output_root', metavar='ROOT', default=default_root,
                         help='Output root directory. Default is "{}"'.format(default_root))
    default_transient = path('{ROOT}', 'raw_transient_events')
    default_results = path('{ROOT}', 'results')

    # QUERY
    project_or_nodes = query_parser.add_mutually_exclusive_group(required=True)
    project_or_nodes.add_argument('--project', metavar='ID', help='Limit results to a certain Project by ID (not Name). Node "ID"\'s will differ from --nodes "SN"\'s. Mutually exclusive with --nodes.')
    project_or_nodes.add_argument('--nodes', metavar='SN', nargs='+', type=int, help='Node Serial Numbers to query for. Currently only "C22" type devices are valid. Mutually exclusive with --project.')
    query_parser.add_argument('--date', metavar='START', required=True, type=dt.date.fromisoformat, help='Query Start Date (YYYY-MM-DD) utc. This field is required.')
    query_parser.add_argument('--enddate', metavar='END', type=dt.date.fromisoformat, help='Query End Date, inclusive (YYYY-MM-DD) utc. If not specified, the current date is assumed.')
    query_parser.add_argument('--output', **outdir_kwargs)
    #TODO --date and --enddate to ALSO accept epoch timestamps

    # GET: Fetch and Process #
    fetch_parser.add_argument('--id', required=True, help='The Search ID from a previously issues search query. A search results file will be downloaded/cached to the results/output ROOT directory. This value is non-optional')
    fetch_parser.add_argument('--limit-to', metavar='NUM', default='1+', help='Limit download/processing to transients with NUM number of unique overlapping instruments. Eg "4" will only download transients that are concurent-in-time with 3 other instruments\' transients. "3+" will download transients are concurent-in-time with 2 OR MORE other instruments\' transients. The default is "1+", ie no limit.')
    fetch_parser.add_argument('--overlap', metavar='SECONDS', default=40, type=int, help='Transient Events are aggregated according to whether they overlap in time. This argument adjusts the permitted overlap window. If "-1", processed transient events are not grouped. Default is "40" seconds')
    fetch_parser.add_argument('--dry-run', action='store_true', help='If invoked, a summary of files to-be-downloaded/created is shown. No downloading or processing is excecuted.')
    # TODO limit processing to start and end dates or epochs
    # TODO clobber-proc, otherwise skip already created files
    # TODO what to do when you run out of memory for a large overlap of files
    # TODO no-cache option.

    # Output Group
    outdir_group = fetch_parser.add_argument_group('Output Options')
    outdir_group.add_argument('--output', **outdir_kwargs)
    outdir_group.add_argument('--cache', metavar='DIR', default='transient_cache', help='Directory where raw transient data are downloaded and cached to. Default is "transient_cache"')
    outdir_group.add_argument('--raw', metavar='DIR', default=default_transient, help='Directory where individual, non-time-adjusted transient-events are saved to as csv\'s. Default is "{}"'.format(default_transient))
    outdir_group.add_argument('--proc', metavar='DIR', default=default_results, help='Directory where processed results are saved to. Defaults is "{}"'.format(default_results))
    outdir_group.add_argument('--clobber-raw', action='store_true', help='If invoked, previously downloaded and compiled files in CACHE and RAW will re-downloaded and re-compiled.')

    return parser


def do_AVAIL(args):
    AUTH = open_authfile(args.authfile)

    def get_sensors(AUTH):
        r = requests.get(API_URL+'/sensor', auth=AUTH)
        sensor_data = r.json()
        # Keys: serial, type, version, from_plm, disabled, *_url
        return sensor_data

    def get_projects(AUTH):
        r = requests.get(API_URL+'/project', auth=AUTH)
        project_data = r.json()
        # Keys: id, name, datetime_from, datetime_to, active,
        #       project_id, timestamp_from, timestamp_to, timezone, company_id,
        #       location (wgs84, gen, description)
        #       *_url, customer_*
        return project_data

    sensor_data = get_sensors(AUTH)
    print('\nAVAIL NODES')
    for sensor in sensor_data:
        print('  SN: {serial} (type {type})'.format(**sensor))

    project_data = get_projects(AUTH)
    print('\nAVAIL PROJECTS')
    for proj in project_data:
        date_from,date_to = proj['datetime_from'][:10],proj['datetime_to'][:10]
        print('  {id}: {name:<20} ({date_from} to {date_to})'.format(**proj,date_from=date_from,date_to=date_to))


def get_search(search_id, AUTH, interval=10):
    print('CHECKING SEARCH STATUS...')

    def _get_search(location, AUTH, mode=None):
        assert mode in [None, 'stats', 'data']
        url = '/'.join([API_URL, 'search', location])
        if mode:
            url += '/'+mode
            print('  Results URL:', mode, url)
        header = {'accept': 'application/json'}
        r = requests.get(url, auth=AUTH, headers=header)
        try:
            output = r.json()
            return output
        except Exception as e:
            print(type(e), e, r)

    data_ready = None
    while data_ready is None:
        chk = _get_search(search_id, AUTH)
        if chk['state'] == 'finished':
            print('  STATE:', chk['state'])
            data_ready = True
        elif chk['state'] == 'abort':
            print('  STATE:', chk['state'], '({})'.format(chk['abort_reason']))
            data_ready = False
        else:
            print('  STATE:', chk['state'],'...')
            time.sleep(interval)

    if data_ready:
        stats = _get_search(search_id, AUTH, 'stats')
        data = _get_search(search_id, AUTH, 'data')
        return data,stats
    else:
        return None,None


def print_query_result_summary(query_result):
    nodes_trancount = dict()
    for t in query_result['transients']:
        nodes = [sn for sn in t.keys() if sn not in ['timestamp','datetime']]
        for node in nodes:
            try: nodes_trancount[node] += 1
            except KeyError: nodes_trancount[node] = 1
    print('    Nodes : Transient Events')
    trancount = 0
    for sn,count in nodes_trancount.items():
        print('   {:>6} : {:>5}'.format(sn,count))
        trancount += count
    print('    TOTAL :', trancount)
    print('  First Transient:', query_result['transients'][0]['datetime'])
    print('  Last Transient: ', query_result['transients'][-1]['datetime'])


def do_QUERY(args):
    AUTH = open_authfile(args.authfile)

    # FORMING QUERY
    query = {'datetime_from':args.date.isoformat(),
             'timezone':'UTC',
             'data_types':{'transient':True}}
    if args.enddate:
        query['datetime_to'] = args.enddate.isoformat()+' 23:59:59'
    if args.nodes:
        query['devices'] = [{"type": "C22", "serial": sn} for sn in args.nodes]
    query = json.dumps(query)

    # FORMING QUERY URL
    if 'project' in args and args.project:
        query_url = '/'.join([API_URL, 'project', str(args.project), 'search'])
    else:
        query_url = '/'.join([API_URL, 'search'])

    # ISSUING QUERY to API
    print('QUERY:', query_url, query)
    header = {'Content-Type': 'application/json', 'accept': 'application/json'}
    r = requests.post(query_url, auth=AUTH, data=query, headers=header)
    output = r.json()
    # Keys: id state timezone created_at datetime_from company_id
    #       data_types (transient,blast,interval,monon)
    #       url self_url transient_url stats_url data_url analysis_url

    # Initial Validation
    if 'id' not in output:
        print('ERROR: "{}"'.format(output))
        return
    search_id = output['id']
    print('SEARCH_ID =', search_id)

    # Fetching search data and metrics
    # This may take a few moments.
    search_data,stats = get_search(search_id,AUTH)

    # Validating that we have some number of transients
    print('\nSummary...')
    assert 'transients' in search_data and len(search_data['transients'])>0, 'ERROR: Transients not found in query data results.'
    print_query_result_summary(search_data)

    # Writing Query Results File
    output_root = args.output_root.format(ID=search_id)
    os.makedirs(output_root, exist_ok=True)
    search_data__file = SEARCH_RESULTS_FILE.format(ROOT=output_root,ID=search_id)
    print('\nSaving Query Results File:')
    print(' ',search_data__file)
    with open(search_data__file,'w') as f:
        json.dump(search_data, f, indent=2)


def get_transient(key, AUTH, cache_dir=None, clobber=False):
    key = key.replace('/','%2F')
    transkey_fname = key+'.data'
    if cache_dir and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    localfile = os.path.join(cache_dir,transkey_fname) if cache_dir else None

    # Download data if a cache_dir not specified, or of the file doesn't exist, or if you're gonna overwrite it
    if cache_dir is None or not os.path.isfile(localfile) or clobber:
        url = '/'.join([API_URL,'search','transient_key',key])
        header = {'accept':'application/json'}
        r = requests.get(url, auth=AUTH, headers=header)
        try:
            meta = r.json()
            if cache_dir:
                with open(localfile,'w') as f:
                    json.dump(meta,f)
            data = meta.pop('data')
            meta['LOCALFILE'] =  localfile
            return data, meta
            # meta: self_url config_id y_label x_label unit trig_type max_values
            #       timestamp_start timestamp_end timestamp time_diff
            #       standard_text2 standard_text1 standard_code avail_standards
            #       sensor_type node_type node_serial quantity latest_calibration
            #       infra_timestamp_start infra_timestamp_end infra_timestamp
            #       data_resolution data_digits
            #       channel channel_values channel_map
            #       channel_min_level_to_calc_frequency  channel_db_ref_level
            # data:
        except json.decoder.JSONDecodeError:
            return r,r.status_code
    else: #if os.path.isfile(localfile):
        with open(localfile) as f:
            meta = json.load(f)
        data = meta.pop('data')
        meta['LOCALFILE'] = localfile
        return data, meta


def do_GET(args):
    AUTH = open_authfile(args.authfile)

    # 1 Resolve output template strings
    args.output_root = args.output_root.format(ID=args.id)
    dir_kwargs = dict(ROOT=args.output_root,ID=args.id)
    args.cache = args.cache.format(**dir_kwargs)
    args.raw = args.raw.format(**dir_kwargs)
    args.proc = args.proc.format(**dir_kwargs)
    search_data__file = SEARCH_RESULTS_FILE.format(**dir_kwargs)

    # 2 get search data
    if os.path.isfile(search_data__file):
        with open(search_data__file) as f:
            search_data = json.load(f)
    else:
        search_data, stats = get_search(args.id, AUTH)
        assert 'transients' in search_data and len(search_data['transients']) > 0, 'ERROR: Transients not found in query data results.'
        os.makedirs(args.output_root,exist_ok=True)
        with open(search_data__file, 'w') as f:
            json.dump(search_data, f, indent=2)

    # 3 group all transient keys by event
    events = []
    for sigi_evt in search_data['transients']:
        nodes = [dev for dev in sigi_evt if dev not in ('timestamp','datetime')]
        transients_per_node = [sigi_evt[node]['transients'] for node in nodes]
        for node,transients in zip(nodes,transients_per_node):

            # Example Transient
            # {  "value":          "2.89",
            #    "url":            "dHJzLQIAAAAAAAAAKJMBAAAAAAAAAAAAAAAAADCrRwLfEgAA",
            #    "unit":           "in/s",
            #    "transient_url":  "/api/v1/search/transient_key/dHJzLQIAAAAAAAAAKJMBAAAAAAAAAAAAAAAAADCrRwLfEgAA",
            #    "timestamp":      1579894799,
            #    "overload":       true,
            #    "meta_id":        "51B__300__OFF",
            #    "label":          "V",
            #    "frequency":      "2.95",
            #    "datetime":       "2020-01-24 19:39:59"
            # }

            assert all(t['timestamp'] == transients[0]['timestamp'] for t in transients), 'Error: transients have different timestamps'
            evt = dict(node=node,
                       ts=transients[0]['timestamp'],
                       keys={t['label']:t['url'] for t in transients},
                       #value,freq,overload...
                       )
            events.append(evt)
    events.sort(key=lambda evt: (evt['ts'],evt['node']))

    # 4 group all transient events according to args.overlap
    groups = [[events[0]]]
    for evt in events[1:]:
        prev_evt = groups[-1][-1]
        ts_diff = evt['ts']-prev_evt['ts']
        if ts_diff <= args.overlap:
            groups[-1].append(evt)
        else:
            groups.append([evt])

    # 5 filter groupings according to args.limit_to
    if args.limit_to.endswith('+'):
        limit_to, or_more = int(args.limit_to[:-1]), True
    else:
        limit_to, or_more = int(args.limit_to), False
    indices_to_remove = []
    for i,group in enumerate(groups):
        unique_nodes = set([evt['node'] for evt in group])
        if or_more and len(unique_nodes)>=limit_to:
            pass # keep
        elif len(unique_nodes)==limit_to:
            pass # keep
        else:
            indices_to_remove.append(i)
    for i in sorted(indices_to_remove,reverse=True):
        del groups[i]

    # 6 check args.dry_run, if true print summary and be done
    if args.dry_run:
        print('Transient Event Groups:')
        for group in groups:
            nodes = [evt['node'] for evt in group]
            node_count_dict = {dev:nodes.count(dev) for dev in set(nodes)}
            timestamp_epoch = group[0]['ts']
            timespan_seconds = group[-1]['ts'] - group[0]['ts'] + 40 #seconds
            timestamp_iso = dt.datetime.utcfromtimestamp(timestamp_epoch).isoformat().replace('T','_')
            template_str = '  {} ({})    Duration:{:>3}s    TotalTransients:{:>2}    NodeEvents: {}'
            print(template_str.format(timestamp_iso, timestamp_epoch, timespan_seconds,
                                      len(group), str(node_count_dict).replace("'","")))
        print('Total Events:', sum([len(group) for group in groups]))
        print('Total Groups:', len(groups))

        cache_files = sum([sum([len(evt['keys']) for evt in group]) for group in groups])
        raw_files = sum([len(group) for group in groups])
        group_files = len(groups)
        total_files = cache_files + raw_files + group_files
        cache_mem = cache_files * 6830000 #Bytes
        raw_mem = raw_files * 13850000 #Bytes
        group_mem = raw_mem # about right.
        total_mem = cache_mem+raw_mem+group_mem

        def siground(x, sig=1):
            from math import floor, log10
            rval = -int(floor(log10(abs(x))))+sig-1
            return round(x, rval)
        def humanize(fsize):
            for denom in ['Bytes', 'KB', 'MB', 'GB', 'TB','PT']:
                if fsize > 1024.0:
                    fsize /= 1024.0
                else:
                    return '{:g}{}'.format(siground(fsize, 2), denom)
        cache_mem,raw_mem,group_mem,total_mem = humanize(cache_mem),humanize(raw_mem),humanize(group_mem),humanize(total_mem)
        print('Download Estimates...')
        template_str = '  cache: {} files ({})\n  raw:   {} files, ({})\n  proc:  {} files, ({})\n  TOTAL: {} files, ({})'
        print(template_str.format(cache_files, cache_mem,
                                  raw_files,   raw_mem,
                                  group_files, group_mem,
                                  total_files, total_mem))
        return

    # 7 start downloading / processing files
    raw_fnames, proc_fnames = [],[]
    for group in groups:
        dfs_bynode = {}
        for evt in group:

        # A download event transients
            evt_transients = dict()
            time_diff = []
            for channel,transkey in evt['keys'].items():
                print('  Fetching Transient:', transkey, end=' ', flush=True)
                data, meta = get_transient(transkey, AUTH, args.cache,args.clobber_raw)
                if isinstance(data, requests.models.Response):
                    print('ERROR: {}'.format(data))
                    continue
                else:
                    print('SUCCESS!')
                evt_transients[channel] = pd.Series(data)
                time_diff.append(meta['time_diff'])
            if not evt_transients:
                print('ERROR: No data. Skipping "{ts}_{node}_GPS±x.xxx.csv"'.format(ts=evt['ts'],node=evt['node']))
                continue
            assert all(td == time_diff[0] for td in time_diff) or all(isnan(td) for td in time_diff)
            time_diff = time_diff[0] or float('nan')

        # B compiling raw transient data to save to file
            # filenaming
            os.makedirs(args.raw,exist_ok=True)
            raw_fname_template = '{ts}_{node}_GPS{td:+.4f}.csv'
            raw_fname_template = os.path.join(args.raw, raw_fname_template)
            raw_fname = raw_fname_template.format(ts=evt['ts'], node=evt['node'], td=time_diff)

            # very rarely, there there can be multiple transients for the same node for the same timestamp. This ensures both get recorded
            postfixes = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            while raw_fname in raw_fnames:
                raw_fname = raw_fname_template.format(ts=evt['ts'], node=evt['node'], td=str(time_diff)+'_'+next(postfixes))
            raw_fnames.append(raw_fname)

            # the write operation
            df_evt = pd.DataFrame(evt_transients)
            df_evt.index.rename('offset',inplace=True)
            df_evt.to_csv(raw_fname)

        # C time adjusting raw data
            def frac_round(num, denom, places=None):
              x = num * denom
              x = round(x)
              x = x / denom
              if places: x = round(x, places)
              return x
            ts = dt.datetime.fromtimestamp(evt['ts'])
            if not isnan(time_diff):
                td = frac_round(time_diff, 4096)
                ts = ts+dt.timedelta(seconds=td)
            df_evt.reset_index(inplace=True)
            df_evt['ts'] = df_evt.offset.apply(lambda val: ts+dt.timedelta(seconds=frac_round(float(val), 4096)))
            df_evt = df_evt.set_index('ts').drop(columns='offset')
            df_evt.rename(columns=lambda c: '{node}__{channel}'.format(ts=evt['ts'],node=evt['node'], channel=c), inplace=True)
            if evt['node'] in dfs_bynode:
                dfs_bynode[evt['node']].append(df_evt)
            else:
                dfs_bynode[evt['node']] = [df_evt]
            print('Raw File: {} ({})'.format(raw_fname, df_evt.first_valid_index()))

    # D compiling final file
        # proc filenaming
        os.makedirs(args.proc, exist_ok=True)
        proc_fname_template = '{ts_iso}_D{tspan}s_N{node_num}_T{tran_num}.csv'
        proc_fname_template = os.path.join(args.proc, proc_fname_template)
        ts_iso = dt.datetime.utcfromtimestamp(group[0]['ts']).isoformat()
        tspan = group[-1]['ts']-group[0]['ts']+40 #seconds
        unique_nodes = set([evt['node'] for evt in group])
        proc_fname = proc_fname_template.format(ts_iso=ts_iso.replace('T','_'), tspan=tspan, node_num=len(unique_nodes),tran_num=len(group))

        # in case of duplicate names
        postfixes = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        while proc_fname in raw_fnames:
            proc_fname = proc_fname_template.format(next(postfixes))
        proc_fnames.append(proc_fname)

        if not dfs_bynode:
            print('ERROR: no data. Skipping', proc_fname)
            continue
        print('Proc File:', proc_fname)

        def correct_1us_error(df_list):
            df_list_sparse = [pd.DataFrame(index=df.index) for df in df_list]
            sparse_df = reduce(lambda x, y: pd.merge(x, y, how='outer', left_index=True, right_index=True),df_list_sparse)
            idx_mask = sparse_df.index.to_series().diff() == dt.timedelta(microseconds=1)
            idx2fix = sparse_df[idx_mask].index
            from_to = {}
            for ts in idx2fix:
                ts2fix = ts-dt.timedelta(microseconds=1)
                from_to[ts2fix] = ts
            for df in df_list:
                df.rename(index=from_to, inplace=True)
            return df_list

        # Concatinating same-node data vertically, while handling overlapping time
        for node in dfs_bynode:
            dfs_bynode[node] = correct_1us_error(dfs_bynode[node])
            df_node = pd.concat(dfs_bynode[node])
            dupes = df_node.index.duplicated(keep='first')
            if any(dupes):
                df_node_dupes = df_node[dupes]
                df_node_dupes = df_node_dupes.rename(columns=lambda c: '{}_overlap'.format(c))
                df_node = df_node[invert(dupes)]
                df_node = pd.concat([df_node,df_node_dupes],axis=1)
            dfs_bynode[node] = df_node

        dfs_bynode = correct_1us_error(dfs_bynode.values())
        df_group = pd.concat(dfs_bynode,axis=1)

        # writing proc file
        df_group.to_csv(proc_fname)

    print('DONE!')


if __name__ == '__main__':
    parser = make_argparser()
    args = parser.parse_args()

    if args.subcommand == 'AVAIL':
        do_AVAIL(args)

    elif args.subcommand == 'QUERY':
        do_QUERY(args)

    elif args.subcommand == 'GET':
        do_GET(args)

    else:
        print(parser.print_help())

