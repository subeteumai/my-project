# %%
!curl -O http://database.rish.kyoto-u.ac.jp/arch/ncep/data/ncep.reanalysis.derived/surface/slp.mon.mean.nc

# %% [markdown]
# ここでは正弦波を描画する
# ##　描画

# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


ds = xr.open_dataset("slp.mon.mean.nc")

# %%
ds

# %%
ds.slp.sel(lon=135, lat=35)[7::12].plot(figsize=[7,5])
plt.show()

# %%
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point


wdata, wlon = add_cyclic_point(ds.slp.sel(time="2020-12")[0], ds.lon)

fig = plt.figure(figsize=[9, 7])
ax = fig.add_subplot(111, projection=ccrs.Orthographic(135,35))

p = ax.contourf(wlon, ds.lat, wdata, transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_global()
fig.colorbar(p)
plt.show()

# %%
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': 'volumetric_soil_water_layer_1',
        'year': '2023',
        'month': ['01', '02'],
        'day': ['01'],
        'time': ['00:00'],
        'format': 'netcdf',
    },
    'soil_moisture_2023.nc'
)

# %%
import cdsapi

dataset = "reanalysis-era5-land"
request = {
    "variable": ["volumetric_soil_water_layer_1"],
    "year": "2023",
    "month": "01",
    "day": ["01"],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()


# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# --- 設定 ---
# ダウンロードしたNetCDFファイル名
filename = 'ca31a37e94c92c0152389e3f490aea95.nc'
# プロットしたい時刻のインデックス (0は最初の時刻 00:00)
time_index = 0

ds = xr.open_dataset(filename)


# データセットの情報を表示して変数名を確認
print("--- データセット情報 ---")
print(ds)
print("--------------------")


# データ変数を選択 (多くの場合 'swvl1' という名前です)
# ds.data_vars の中から適切な変数名を選んでください
data_variable = ds['swvl1']

# 指定した時刻のデータを抽出
data_slice = data_variable.isel(valid_time=time_index)


# --- 描画処理 ---
# 描画サイズを指定
plt.figure(figsize=(12, 6))

# 地図の投影法を指定 (PlateCarreeは最も基本的な緯度経度図)
ax = plt.axes(projection=ccrs.PlateCarree())

# データを地図上にプロット
# pcolormeshを使うと高速に描画できます
data_slice.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='viridis_r',  # カラースキーム（逆順のviridis）
    cbar_kwargs={'label': 'Volumetric soil water (m³/m³)'} # カラーバーのラベル
)

# 海岸線を描画
ax.coastlines()

# 緯度経度のグリッド線を描画
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')

# タイムスタンプからタイトルを生成
time_str = str(data_slice.valid_time.values)
plt.title(f'Volumetric Soil Water Layer 1 - {time_str}')


# プロットを表示
plt.show()

# %%
lat_point = 35.7
lon_point = 140.0
# データ変数を選択
data_variable = ds['swvl1']

# 指定した緯度経度に最も近い点のデータを抽出
timeseries = data_variable.sel(latitude=lat_point, longitude=lon_point, method='nearest')

# --- 描画処理 ---
plt.figure(figsize=(12, 6))

timeseries.plot.line(marker='o')

# グラフのタイトルとラベルを設定
plt.title(f'Soil Moisture Time Series at (Lat: {lat_point}, Lon: {lon_point})')
# ★修正点(推奨): 'Time (UTC)' -> 'Valid Time (UTC)'
plt.xlabel('Time (UTC)')
plt.ylabel('Volumetric soil water (m³/m³)')
plt.grid(True)

plt.show()

# %%
import cdsapi

dataset = "reanalysis-era5-land"
download_file = "20250525_soil_moisture.nc"
request = {
    "variable": ["volumetric_soil_water_layer_1"],
    "year": "2025",
    "month": "05",
    "day": ["25"],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download(download_file)


# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# --- 設定 ---
# ダウンロードしたNetCDFファイル名
filename = '20250525_soil_moisture.nc'
# プロットしたい時刻のインデックス (0は最初の時刻 00:00)
time_index = 0

ds = xr.open_dataset(filename)




# データ変数を選択 (多くの場合 'swvl1' という名前です)
# ds.data_vars の中から適切な変数名を選んでください
data_variable = ds['swvl1']

# 指定した時刻のデータを抽出
data_slice = data_variable.isel(valid_time=time_index)


# --- 描画処理 ---
# 描画サイズを指定
plt.figure(figsize=(12, 6))

# 地図の投影法を指定 (PlateCarreeは最も基本的な緯度経度図)
ax = plt.axes(projection=ccrs.PlateCarree())

# データを地図上にプロット
# pcolormeshを使うと高速に描画できます
data_slice.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='viridis_r',  # カラースキーム（逆順のviridis）
    cbar_kwargs={'label': 'Volumetric soil water (m³/m³)'} # カラーバーのラベル
)

# 海岸線を描画
ax.coastlines()

# 緯度経度のグリッド線を描画
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')

# タイムスタンプからタイトルを生成
time_str = str(data_slice.valid_time.values)
plt.title(f'Volumetric Soil Water Layer 1 - {time_str}')


# プロットを表示
plt.show()

# %%
lat_point = 35.7
lon_point = 140.0
# データ変数を選択
data_variable = ds['swvl1']

# 指定した緯度経度に最も近い点のデータを抽出
timeseries = data_variable.sel(latitude=lat_point, longitude=lon_point, method='nearest')

# --- 描画処理 ---
plt.figure(figsize=(12, 6))

timeseries.plot.line(marker='o')

# グラフのタイトルとラベルを設定
plt.title(f'Soil Moisture Time Series at (Lat: {lat_point}, Lon: {lon_point})')
# ★修正点(推奨): 'Time (UTC)' -> 'Valid Time (UTC)'
plt.xlabel('Time (UTC)')
plt.ylabel('Volumetric soil water (m³/m³)')
plt.grid(True)

plt.show()

# %%
import cdsapi

client = cdsapi.Client()

# ダウンロードするファイル名
download_file = "solar_radiation_japan_202405.nc"

client.retrieve(
    'reanalysis-era5-single-levels', # ★データセット名が異なります
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'surface_solar_radiation_downwards',
        # 'variable': [ # 複数選択も可能
        #     'surface_solar_radiation_downwards',
        #     'total_sky_direct_solar_radiation_at_surface', # 直達日射量
        # ],
        'year': '2024',
        'month': '05',
        'day': '01',
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
        ],
        # 日本周辺の領域を指定 [North, West, South, East]
        'area': [
            46, 128, 30, 148,
        ],
    },
    download_file)

print(f"データが {download_file} にダウンロードされました。")

# %%
filename = 'solar_radiation_japan_202405.nc'

# NetCDFファイルを開く
ds = xr.open_dataset(filename)

# 変数名は 'ssrd' という短い名前になっていることが多い
# print(ds) で確認してください
data_variable = ds['ssrd']

# 日本時間の13時(04:00 UTC)のデータを計算
# 04:00の値(00-04時の積算)から03:00の値(00-03時の積算)を引く
rad_04_utc_J = data_variable.sel(valid_time='2024-05-01T04:00') - data_variable.sel(valid_time='2024-05-01T03:00')

# 単位を J/m² から W/m² (1時間平均) に変換 (1時間 = 3600秒)
rad_04_utc_W_m2 = rad_04_utc_J / 3600

# --- 描画処理 ---
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# W/m^2 に変換したデータをプロット
rad_04_utc_W_m2.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='hot_r', # 日射量に適したカラースキーム
    cbar_kwargs={'label': 'Solar Radiation (W/m²)'}
)

ax.coastlines(resolution='10m') # 高解像度の海岸線
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')

plt.title('Surface Solar Radiation at 13:00 JST (04:00 UTC), 2024-05-01')
plt.show()

# %%
import pandas as pd

# --- 設定 ---
filename = 'solar_radiation_japan_202405.nc'
lat_point = 35.68  # 東京
lon_point = 139.76

# NetCDFファイルを開く
ds = xr.open_dataset(filename)
data_variable = ds['ssrd']

# 東京に最も近い点のデータを抽出
timeseries_J = data_variable.sel(latitude=lat_point, longitude=lon_point, method='nearest')

# 【重要】積算値から1時間ごとの値に変換
hourly_J = timeseries_J.diff('time')

# 単位を J/m² から W/m² (1時間平均) に変換
hourly_W_m2 = hourly_J / 3600

# 描画のために pandas Series に変換
# xarrayの .diff() を使うと最初の時刻(00:00)が消えるので注意
s = hourly_W_m2.to_series()

# タイムゾーンをUTCからJSTに変換してX軸ラベルを分かりやすくする
s.index = s.index.tz_localize('UTC').tz_convert('Asia/Tokyo')


# --- 描画処理 ---
plt.figure(figsize=(12, 6))
s.plot(kind='line', marker='o')

plt.title(f'Hourly Solar Radiation in Tokyo (Lat: {lat_point}, Lon: {lon_point})')
plt.xlabel('Time (JST)')
plt.ylabel('Average Solar Radiation (W/m²)')
plt.grid(True)
plt.show()

# %%
import cdsapi

client = cdsapi.Client()

download_file = "2025wind_data_japan.nc"

client.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind',
        ],
        'year': '2025',
        'month': '05',
        'day': '20',
        # 日本時間の正午12時は03:00 UTC
        'time': '03:00',
        # 日本周辺の領域を指定 [North, West, South, East]
        'area': [
            46, 128, 30, 148,
        ],
    },
    download_file)

print(f"データが {download_file} にダウンロードされました。")

# %%
filename = '2025wind_data_japan.nc'

# NetCDFファイルを開く
ds = xr.open_dataset(filename)
# データセットの情報を確認して変数名を見る (u10, v10など)
print(ds)

# データ変数を選択 (多くの場合、短い名前 'u10', 'v10' になっています)
u_wind = ds['u10'].squeeze() # squeeze()で不要な次元を削除
v_wind = ds['v10'].squeeze()

# 風速を計算 (ピタゴラスの定理)
wind_speed = np.sqrt(u_wind**2 + v_wind**2)

# --- 描画処理 ---
plt.figure(figsize=(12, 12))
ax = plt.axes(projection=ccrs.PlateCarree())

# 1. 風速を色付きのコンター図でプロット
wind_speed.plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=15, # 色分けの段階数
    cmap='viridis', # カラースキーム
    cbar_kwargs={'label': 'Wind Speed (m/s)'}
)

# 2. 風向を矢羽(quiver)でプロット
# 全ての点に矢印を描くと真っ黒になるため、データを間引く (ここでは10点ごと)
thin = 10
quiver = ax.quiver(
    u_wind.longitude.values[::thin],
    u_wind.latitude.values[::thin],
    u_wind.values[::thin, ::thin],
    v_wind.values[::thin, ::thin],
    transform=ccrs.PlateCarree(),
    color='white',
    width=0.003
)

# 矢印のスケール（凡例）を追加
ax.quiverkey(quiver, X=0.85, Y=1.05, U=10,
             label='10 m/s', labelpos='E')


# 地図の装飾
ax.coastlines(resolution='10m', color='black')
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')

time_str = pd.to_datetime(ds.valid_time.values[0]).strftime('%Y-%m-%d %H:%M UTC')
plt.title(f'Wind Speed and Direction at 10m ({time_str})')
plt.show()


# %%
import cdsapi

client = cdsapi.Client()

download_file = "soil_moisture_japan_20250525.nc"

client.retrieve(
    'reanalysis-era5-land',
    {
        'variable': 'volumetric_soil_water_layer_1',
        'year': '2025',
        'month': '05',
        'day': '25',
        'time': '00:00/to/23:00/by/1', # 時間の指定をより簡潔に
        'area': [46, 128, 30, 148],
        'format': 'netcdf',
        # ★★★ ベストプラクティスとして常に追加 ★★★
        'download_format': 'unarchived',
    },
    download_file)

print(f"データが {download_file} にダウンロードされました。")

# %%
filename = 'soil_moisture_japan_20250525.nc'

# NetCDFファイルを開く
ds = xr.open_dataset(filename)
# 変数名(swvl1)と座標名(valid_time)を確認
print(ds)

# データ変数と時刻を選択
# 日本時間の正午12時は 03:00 UTC
data_slice = ds['swvl1'].sel(valid_time='2025-05-25T03:00:00')


# --- 描画処理 ---
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# 土壌水分量をプロット
data_slice.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='viridis_r',  # 水分量に適したカラースキーム（緑→黄）
    cbar_kwargs={'label': 'Volumetric Soil Water (m³/m³)'}
)

# 地図の装飾
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

time_str = pd.to_datetime(data_slice.valid_time.values).strftime('%Y-%m-%d %H:%M JST')
plt.title(f'Volumetric Soil Water Layer 1 at {time_str}')
plt.show()

# %%
client = cdsapi.Client()

download_file = "soil_moisture_japan_20250506.nc"

client.retrieve(
    'reanalysis-era5-land',
    {
        'variable': 'volumetric_soil_water_layer_1',
        'year': '2025',
        'month': '05',
        'day': '06',
        'time': '00:00/to/23:00/by/1', # 時間の指定をより簡潔に
        'area': [46, 128, 30, 148],
        'format': 'netcdf',
        # ★★★ ベストプラクティスとして常に追加 ★★★
        'download_format': 'unarchived',
    },
    download_file)

print(f"データが {download_file} にダウンロードされました。")

# %%
filename = 'soil_moisture_japan_20250506.nc'

# NetCDFファイルを開く
ds = xr.open_dataset(filename)
# 変数名(swvl1)と座標名(valid_time)を確認
print(ds)

# データ変数と時刻を選択
# 日本時間の正午12時は 03:00 UTC
data_slice = ds['swvl1'].sel(valid_time='2025-05-06T03:00:00')


# --- 描画処理 ---
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# 土壌水分量をプロット
data_slice.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='viridis_r',  # 水分量に適したカラースキーム（緑→黄）
    cbar_kwargs={'label': 'Volumetric Soil Water (m³/m³)'}
)

# 地図の装飾
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

time_str = pd.to_datetime(data_slice.valid_time.values).strftime('%Y-%m-%d %H:%M JST')
plt.title(f'Volumetric Soil Water Layer 1 at {time_str}')
plt.show()


