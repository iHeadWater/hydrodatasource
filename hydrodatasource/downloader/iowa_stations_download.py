# download data from usgs and iowa networks
import io
import logging
from datetime import datetime

import pandas as pd
import urllib3 as ur3
from lxml.etree import XPathEvalError
from pandas.errors import ParserError
from urllib3.exceptions import MaxRetryError


def gen_iowa_link(network, sta_id, start_time, end_time, selectors=None):
    request_link = 'https://mesonet.agron.iastate.edu/cgi-bin/request/'
    # network should be asos_awos_metar/cocorahs/dcp_hads_shef/nwscoop/rwis
    if 'ASOS' in network:
        request_link = request_link + 'asos.py?'
        request_link = request_link + f'station={sta_id}&'
        request_link = request_link + 'data=all&'
        start_time = datetime.fromisoformat(start_time)
        end_time = datetime.fromisoformat(end_time)
        start_str = f'year1={start_time.year}&month1={start_time.month}&day1={start_time.day}&'
        end_str = f'year2={end_time.year}&month2={end_time.month}&day2={end_time.day}&'
        request_link = (request_link + start_str + end_str +
                        'tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=M&trace=T'
                        '&direct=no&report_type=1&report_type=3&report_type=4')
    elif 'DCP' in network:
        # 这里的network应该是某个dcp而不是asos这样的数据分类
        request_link = request_link + f'hads.py?network={network}&'
        request_link = request_link + f'stations={sta_id}&'
        start_time = datetime.fromisoformat(start_time)
        end_time = datetime.fromisoformat(end_time)
        start_str = f'year1={start_time.year}&month1={start_time.month}&day1={start_time.day}&hour1=0&minute1=0&'
        end_str = f'year2={end_time.year}&month2={end_time.month}&day2={end_time.day}&hour2=0&minute2=0&'
        request_link = request_link + start_str + end_str + 'delim=comma&threshold=&threshold-var=RG'
    elif 'COOP' in network:
        request_link = 'https://mesonet.agron.iastate.edu/request/'
        request_link = request_link + f'coop/obs-dl.php?network={network}&'
        request_link = request_link + f'station%5B%5D={sta_id}&'
        start_time = datetime.fromisoformat(start_time)
        end_time = datetime.fromisoformat(end_time)
        start_str = f'year1={start_time.year}&month1={start_time.month}&day1={start_time.day}&'
        end_str = f'year2={end_time.year}&month2={end_time.month}&day2={end_time.day}&'
        request_link = request_link + start_str + end_str + 'what=view&delim=comma'
    elif 'RWIS' in network:
        # https://mesonet.agron.iastate.edu/cgi-bin/request/rwis.py?minute1=0&minute2=0&network=IA_RWIS&stations=RAKI4&stations=RALI4&year1=2013&month1=1&day1=1&hour1=0&year2=2024&month2=4&day2=1&hour2=0&vars=tmpf&vars=dwpf&vars=sknt&vars=drct&vars=gust&vars=tfs0&vars=tfs0_text&vars=tfs1&vars=tfs1_text&vars=tfs2&vars=tfs2_text&vars=tfs3&vars=tfs3_text&vars=subf&what=html&tz=Etc%2FUTC&delim=comma&gis=yes
        request_link = request_link + f'rwis.py?minute1=0&minute2=0&network={network}&'
        request_link = request_link + f'stations={sta_id}&'
        start_time = datetime.fromisoformat(start_time)
        end_time = datetime.fromisoformat(end_time)
        start_str = f'year1={start_time.year}&month1={start_time.month}&day1={start_time.day}&hour1={start_time.hour}'
        end_str = f'year2={end_time.year}&month2={end_time.month}&day2={end_time.day}&hour2={end_time.hour}'
        request_link = request_link + start_str + '&' + end_str + '&' + 'vars=tmpf&vars=dwpf&vars=sknt&vars=drct&vars=gust&vars=tfs0&vars=tfs0_text&vars=tfs1&vars=tfs1_text&vars=tfs2&vars=tfs2_text&vars=tfs3&vars=tfs3_text&vars=subf&tz=Etc%2FUTC&delim=comma&gis=yes'
    elif 'ISUSM' in network:
        request_link = request_link + f'isusm.py?mode=hourly&timeres=hourly&station={sta_id}'
        start_time = datetime.fromisoformat(start_time)
        end_time = datetime.fromisoformat(end_time)
        start_str = f'year1={start_time.year}&month1={start_time.month}&day1={start_time.day}&'
        end_str = f'year2={end_time.year}&month2={end_time.month}&day2={end_time.day}&'
        request_link = request_link + start_str + end_str + (
            'vars=tmpf&vars=relh&vars=solar&vars=precip&vars=speed&vars'
            '=drct&vars=et&vars=soil04t&vars=soil12t&vars=soil24t&vars'
            '=soil50t&vars=soil12vwc&vars=soil24vwc&vars=soil50vwc&vars'
            '=bp_mb&vars=sv&format=comma&missing=M&tz=UTC')
    else:
        request_link = None
    return request_link


def download_from_link(request_link, selectors=None):
    try:
        pd_df_str = ur3.request('GET', request_link, timeout=600).data.decode('utf-8')
    except MaxRetryError:
        pd_df_str = ''
    # 区分返回值是否为html格式
    if 'dataframe' in pd_df_str:
        try:
            pd_df = pd.read_html(io.StringIO(pd_df_str))[0]
        except XPathEvalError:
            pd_df = pd.DataFrame()
    else:
        if pd_df_str == '':
            pd_df = pd.DataFrame()
        else:
            try:
                pd_df = pd.read_csv(io.StringIO(pd_df_str))
            except ParserError:
                pd_df = pd.DataFrame()
    if selectors is not None:
        pd_df_selector = pd_df[selectors]
    else:
        pd_df_selector = pd_df
    return pd_df_selector
