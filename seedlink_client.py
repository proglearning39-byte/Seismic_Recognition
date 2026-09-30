from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import numpy as np

def fetch_latest_waveform(net="IU", sta="NACB", loc="00", cha="BHZ", duration_sec=16):
    """
    透過 FDSN / SeedLink 抓取近 16 秒最新即時數據
    """
    try:
        client = Client("IRIS")
        end_time = UTCDateTime()
        start_time = end_time - duration_sec
        st = client.get_waveforms(net, sta, loc, cha, start_time, end_time)
        tr = st[0]
        data = tr.data[:1600].astype(np.float32)
        if len(data) < 1600:
            data = np.pad(data, (0, 1600 - len(data)))
        return data, True
    except Exception as e:
        # 連線異常時回傳模擬雜訊
        return np.random.randn(1600).astype(np.float32) * 0.1, False
