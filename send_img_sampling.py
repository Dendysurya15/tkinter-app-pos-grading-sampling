from importlib.metadata import files
import os
import aiohttp
import asyncio
import datetime
from datetime import datetime, timedelta
import pytz
import requests
import json
from pathlib import Path
from requests.structures import CaseInsensitiveDict
import time
tzInfo = pytz.timezone('Asia/Bangkok')
url_img = 'https://srs-ssms.com/post-img-sampling-py.php'
# time.sleep(60)
id_mill = '3'  # sesuaikan dengan id_mill 
headers_img = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36' }
# headers = {"content-type": "application/x-www-form-urlencoded",
#           'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36'}
# timer = 10
sendMin = 10
# log_dir = Path(os.getcwd() + '/log/log.TXT')
dir_bad_ffb = Path(os.getcwd() + '/img_inference/_worst_.JPG')
dir_good_ffb = Path(os.getcwd() + '/img_inference/_best_.JPG')

def send_file(filedir,timestamp):
            try:
                code = asyncio.get_event_loop().run_until_complete(post_file(filedir,timestamp))
                print("Status : " + str(code))
            except Exception as e:
                print("Errornya "+ str(e))
                #file kosong

async def post_file(filedir,timestamp):

    try:
        files = {
        'file': open(filedir,'rb'),
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"), 
        'id_mill': id_mill,
        }
     
        print(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        async with aiohttp.ClientSession() as session:
            async with session.post(url_img,data=files,headers=headers_img) as resp:
                response = resp.status

    except Exception as e:
        print("Errornya "+ str(e))
        response = 99999
    return response

lastMin = datetime.now(tz=tzInfo)+timedelta(seconds=10, minutes=0, hours=0)


# print(lastMin)
while True:
    

    # print(lastMin)
    if datetime.now(tz=tzInfo) > lastMin:
        send_file(dir_good_ffb, datetime.now(tz=tzInfo))
        send_file(dir_bad_ffb, datetime.now(tz=tzInfo))
        lastMin = datetime.now(tz=tzInfo) + timedelta(seconds=10, minutes=0, hours=0)
    


        
