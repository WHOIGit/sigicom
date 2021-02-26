# sigicom
A CLI tool for downloading and collating C22 Transient Event data from Sigicom's Infra API.

Created for the `SIDEx 2020` project (Kevin Manganini, Erin Fischell, feat. Kat Fung)

## Installation
The only requirements are python with pandas and requests.

`conda env create --file env.yml`


## USAGE
0. `conda activate sigicom`
1. Create an authentication file containing the three following space or newline deliminated values
    1. your Sigicom INFRA **subdomain**
    2. your INFRA User **id**
    3. your INFRA api **token**
* auth.txt example: "whoius xxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

2. Use `python sigicom.py AVAIL auth.txt` to get a sense of what devices and projects you have access to
3. Use `python sigicom.py QUERY --nodes xxxxxx xxxxxx xxxxxx --date YYYY-MM-DD --enddate YYYY-MM-DD auth.txt` or `python sigicom.py QUERY --project xxxxx --date YYYY-MM-DD auth.txt` to initiate a search against Sigicom's INFRA API. A search-ID and search results-file are returned.
4. Use `python sigicom.py GET --id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --overlap xx --limit-to x --dry-run auth.txt`, where --id referse to the aformentioned search-ID, to get a summary of the data you will be downloading. Here --overlap and --limit-to can quickly be adjusted to meet your processing parameters.
    * `--overlap xx` adjusts how many seconds (xx) from the start of one transient event the next transient must be to be compiled with the first. The default is 40 seconds, the typical duration of a transient event, ensuring that some transients will always overlap in time. 0, 1, or 2 seconds is also appropriate. If set to -1, transient events are still processed (adjusted with GPS time offset) but are not combined.
    * `--limit-to x` Limits the downloading and processing to transients with x number of unique overlapping nodes. Eg: "4" will only download transients that are concurent-in-time with 3 other nodes' transients. "3+" will download transients are concurent-in-time with 2 OR MORE other nodes' transients. The default is "1+" ie no limit.
5. Use `python sigicom.py GET --id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --overlap xx --limit-to x auth.txt` (the above command *without* --dry-run) to actually download and process the data. To download ALL the data, omit the `--limit-to` flag. 

**NOTE:** the `auth.txt` authentication file is always at the end.

Processed files are typically output to `output/<search-id>/results/YYYY-MM-DD_HH:MM:SS_Dxxs_Ny_Tz.csv`, where the timetamp is the start of the file, `xx` denotes the **D**uration of the file, `y` is the number of unique **N**odes in the file, and `z` is the total number of **T**ransient events of the file. 

To explore all the options, you may use the `--help` flag:
  * `python sigicom.py --help`
  * `python sigicom.py QUERY --help`
  * `python sigicom.py GET --help`

