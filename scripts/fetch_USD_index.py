import yfinance as yf
import pandas as pd
import time
from datetime import timedelta
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Parameters
symbol = "DX-Y.NYB"
start_date = pd.to_datetime("2022-01-01")
end_date = pd.Timestamp.today()

def fetch_historical_data(symbol, start, end, chunk_days=90, delay_sec=2):
    current_date = start
    df_list = []

    while current_date < end:
        next_date = min(current_date + timedelta(days=chunk_days), end)

        print(f"Downloading data from {current_date.date()} to {next_date.date()}...")

        try:
            data_chunk = yf.download(symbol, start=current_date, end=next_date)
            if not data_chunk.empty:
                df_list.append(data_chunk)
            else:
                print(f"No data fetched for period {current_date.date()} - {next_date.date()}")

        except Exception as e:
            print(f"Error fetching data: {e}")

        current_date = next_date + timedelta(days=1)

        # Delay to avoid rate limits
        time.sleep(delay_sec)

    if df_list:
        historical_data = pd.concat(df_list)
        return historical_data
    else:
        print("No data retrieved.")
        return pd.DataFrame()
    
def fetch_web():
    print("Launching browser for web scraping with Selenium...")

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get("https://www.investing.com/indices/us-dollar-index-historical-data")

        # Accept cookies if prompted
        try:
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))).click()
        except:
            pass

        # Click the date picker
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.history-date'))).click()

        # Fill the date fields
        start_input = driver.find_element(By.ID, 'startDate')
        end_input = driver.find_element(By.ID, 'endDate')

        start_input.clear()
        start_input.send_keys("01/01/2022")

        end_input.clear()
        end_input.send_keys(end_date.strftime("%m/%d/%Y"))

        # Click Apply
        driver.find_element(By.ID, 'applyBtn').click()

        # Wait for table to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.genTbl.closedTbl.historicalTbl'))
        )

        html = driver.page_source
        tables = pd.read_html(html)
        df = tables[0]

        df.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Change %']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df[df['Date'] >= '2022-01-01']

        print("Successfully scraped historical data via Selenium")
        return df

    except Exception as e:
        print(f"Selenium scraping failed: {e}")
        return pd.DataFrame()

    finally:
        driver.quit()

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a raw DXY DataFrame downloaded from Investing.com and formats it to:
    open_time, open, high, low, close, volume, change%
    
    Assumes input has columns: Date, Price, Open, High, Low, Vol., Change %
    """
    # Rename columns
    df = df.rename(columns={
        "Date": "open_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Price": "close",
        "Vol.": "volume",
        "Change %": "change%"
    })

    # Reorder columns
    df = df[["open_time", "open", "high", "low", "close", "volume", "change%"]]

    # Convert date to datetime and format to UTC
    df["open_time"] = pd.to_datetime(df["open_time"], format="%m/%d/%Y")
    df["open_time"] = df["open_time"].dt.strftime("%Y-%m-%d 00:00:00+00:00")


    # Sort in ascending time (optional, but good practice)
    df = df.sort_values("open_time").reset_index(drop=True)

    return df

def clean_ndx_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans NDX dataframe into: open_time, open, high, low, close, volume, change%
    Handles commas in numbers and converts volume from M/B suffixes to floats.
    """
    # Rename columns
    df = df.rename(columns={
        "Date": "open_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Price": "close",
        "Vol.": "volume",
        "Change %": "change%"
    })

    # Convert date format to UTC
    df["open_time"] = pd.to_datetime(df["open_time"], format="%m/%d/%Y")
    df["open_time"] = df["open_time"].dt.strftime("%Y-%m-%d 00:00:00+00:00")

    # Remove commas from all relevant columns
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)

    # Clean volume: e.g. "331.51M" → 331510000.0
    df["volume"] = (
        df["volume"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("M", "e6", regex=False)
        .str.replace("B", "e9", regex=False)
        .apply(pd.to_numeric, errors="coerce")
    )

    # Reorder columns
    df = df[["open_time", "open", "high", "low", "close", "volume", "change%"]]

    # Sort chronologically
    df = df.sort_values("open_time").reset_index(drop=True)

    return df

if __name__ == "__main__":
    # dxy_historical = fetch_historical_data(symbol, start_date, end_date)
    # dxy_historical = fetch_web()
    # if not dxy_historical.empty:
    #     dxy_historical.to_csv(os.path.join('data/historical/external', 'USD_data.csv'), index=True)
    #     print("Historical data saved to dxy_historical_2022_to_now.csv")
    dxy_historical = pd.read_csv(os.path.join('data/historical/external', 'NDX_2022-01-01_2025-04-04_1d.csv'))
    
    cleaned_df = clean_ndx_df(dxy_historical)
    cleaned_df.to_csv(os.path.join('data/historical/external', 'NDX_2022-01-01_2025-04-04_cleaned.csv'), index=False)