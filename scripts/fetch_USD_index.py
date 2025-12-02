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

# ------------------ GOLD functions ------------------
def fetch_gold_yf(symbol: str = "XAUUSD=X", start: pd.Timestamp = start_date, end: pd.Timestamp = end_date, chunk_days: int = 90, delay_sec: int = 2) -> pd.DataFrame:
    """
    Wrapper to fetch Gold (XAUUSD) historical data using yfinance in chunks.
    Default symbol uses Yahoo Finance spot gold ticker: "XAUUSD=X".
    Returns a DataFrame in Yahoo's default OHLCV format.
    """
    return fetch_historical_data(symbol, start, end, chunk_days=chunk_days, delay_sec=delay_sec)


def fetch_gold_web() -> pd.DataFrame:
    """
    Scrape Gold historical data from Investing.com via Selenium.
    URL: https://www.investing.com/commodities/gold-historical-data
    Returns a DataFrame with columns: Date, Price, Open, High, Low, Vol., Change %
    (as provided by the site), sorted ascending by Date and filtered from 2022-01-01.
    """
    print("Launching browser for web scraping GOLD with Selenium...")

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get("https://www.investing.com/commodities/gold-historical-data")

        # Accept cookies if prompted
        try:
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))).click()
        except Exception:
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
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.genTbl.closedTbl.historicalTbl'))
        )

        html = driver.page_source
        tables = pd.read_html(html)
        df = tables[0]

        # Normalize expected column names
        # Investing usually returns: Date, Price, Open, High, Low, Vol., Change %
        # In case of localization/format changes, keep only the first 6-7 columns.
        df = df.iloc[:, :7]
        df.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']

        # Parse and sort dates
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df[df['Date'] >= '2022-01-01']

        print("Successfully scraped GOLD historical data via Selenium")
        return df

    except Exception as e:
        print(f"Selenium scraping for GOLD failed: {e}")
        return pd.DataFrame()

    finally:
        driver.quit()


def clean_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans Gold dataframe (from Investing.com or Yahoo) into:
    columns = [open_time, open, high, low, close, volume, change%]
    - Handles commas and numeric strings
    - Converts volume suffixes K/M/B to float counts
    - Sets open_time to midnight UTC ISO string
    """
    # Standardize columns if coming from Investing.com style
    rename_map = {
        "Date": "open_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Price": "close",
        "Vol.": "volume",
        "Volume": "volume",       # in case of Yahoo format
        "Change %": "change%",
        "Adj Close": "close"      # Yahoo alt
    }
    df = df.rename(columns=rename_map)

    # Ensure required columns exist; if Yahoo format, create missing ones where possible
    required_cols = ["open_time", "open", "high", "low", "close"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"clean_gold_df: required column '{c}' not found after renaming")

    # If volume not present, create as NaN
    if "volume" not in df.columns:
        df["volume"] = pd.NA

    # If change% not present, create as NaN
    if "change%" not in df.columns:
        df["change%"] = pd.NA

    # Parse date: accept both datetime and string; try multiple formats
    if pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        dt = df["open_time"]
    else:
        # Try common formats: "YYYY-MM-DD", "MM/DD/YYYY"
        try:
            dt = pd.to_datetime(df["open_time"], format="%m/%d/%Y", errors="coerce")
        except Exception:
            dt = pd.to_datetime(df["open_time"], errors="coerce")
    df["open_time"] = dt.dt.strftime("%Y-%m-%d 00:00:00+00:00")

    # Numeric cleanup (remove commas)
    for col in ["open", "high", "low", "close"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
            .str.replace("%", "")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Volume cleanup: support K, M, B suffixes
    def _parse_volume(x: str):
        if pd.isna(x):
            return pd.NA
        s = str(x).replace(",", "").strip()
        try:
            if s.endswith("K"):
                return float(s[:-1]) * 1e3
            if s.endswith("M"):
                return float(s[:-1]) * 1e6
            if s.endswith("B"):
                return float(s[:-1]) * 1e9
            return float(s)
        except Exception:
            return pd.NA

    df["volume"] = df["volume"].apply(_parse_volume)

    # Reorder and sort
    out_cols = ["open_time", "open", "high", "low", "close", "volume", "change%"]
    df = df[out_cols]
    df = df.sort_values("open_time").reset_index(drop=True)

    return df

if __name__ == "__main__":
    # dxy_historical = fetch_historical_data(symbol, start_date, end_date)
    # dxy_historical = fetch_web()
    # if not dxy_historical.empty:
    #     dxy_historical.to_csv(os.path.join('data/historical/external', 'USD_data.csv'), index=True)
    #     print("Historical data saved to dxy_historical_2022_to_now.csv")

    # --- GOLD examples ---
    # Using yfinance (spot gold):
    # gold_raw = fetch_gold_yf(symbol="XAUUSD=X", start=start_date, end=end_date)
    # if not gold_raw.empty:
    #     gold_clean = clean_gold_df(gold_raw.rename(columns={"Date": "open_time"}))  # unify column naming
    #     gold_clean.to_csv(os.path.join('data/historical/external', 'GOLD_XAUUSD_1d_cleaned.csv'), index=False)

    # Using Investing.com via Selenium (Gold Futures):
    # gold_web = fetch_gold_web()
    # if not gold_web.empty:
    #     gold_clean_web = clean_gold_df(gold_web)
    #     gold_clean_web.to_csv(os.path.join('data/historical/external', 'GOLD_investing_1d_cleaned.csv'), index=False)

    dxy_historical = pd.read_csv(os.path.join('data/historical/external', 'NDX_2022-01-01_2025-04-04_1d.csv'))
    
    cleaned_df = clean_ndx_df(dxy_historical)
    cleaned_df.to_csv(os.path.join('data/historical/external', 'NDX_2022-01-01_2025-04-04_cleaned.csv'), index=False)