import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://www.tianqi24.com/guangzhou/'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def get_weather_data(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    weather_table = soup.find('ul', class_='col6')
    if weather_table is None:
        return pd.DataFrame()
    rows = weather_table.find_all('li')[1:]

    data = []
    for row in rows:
        cols = row.find_all('div')
        cols = [col.get_text(strip=True) for col in cols]
        data.append(cols)
    
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    result = pd.DataFrame(columns=['日期', '白天/晚上', '高温', '低温', 'AQI', '风向', '降水量'])
    for year in range(2020, 2026):
        for month in range(1, 13):
            month_str = f'{month:02d}'
            print(f'正在获取{year}年{month}月天气数据...')
            get_url = url + f'history{year}{month_str}.html'
            weather_df = get_weather_data(get_url)
            result = pd.concat([result, weather_df], ignore_index=False)
    result.to_csv('./weather_data.csv', index=False, encoding='utf-8-sig')
    print('天气数据已保存到 weather_data.csv')