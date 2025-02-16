import json
import os

import requests

rated_limit_triggered = False


def load_mf_mapping(path: str):
    with open(path, "r") as f:
        data_dict = json.load(f)

    # Extract fields and data
    fields = data_dict["fields"]
    data = data_dict["data"]

    # Convert data to list of dictionaries
    mappings = [dict(zip(fields, row)) for row in data]

    return mappings


def get_mapping_from_secapi(ticker):
    global rated_limit_triggered

    if rated_limit_triggered:
        return None

    if not ticker or len(ticker) <= 3:
        return None

    api_key = os.getenv("SEC_API_KEY")
    if not api_key:
        return None

    print(f"Getting mapping for {ticker} from SEC API")

    response = requests.get(
        f"https://api.sec-api.io/mapping/ticker/{ticker}",
        headers={"Authorization": api_key},
    )
    if response.status_code == 200:
        result = response.json()
        if len(result) > 0:
            if "cik" in result[0]:
                return result[0]["cik"]
            else:
                return None
    elif response.status_code == 429:
        print("Rate limit triggered")
        rated_limit_triggered = True
        return None
    else:
        print(f"Failed to get mapping for {ticker}. Got {response}")
        return None


def process_fund_list():
    mappings = load_mf_mapping("scratch/company_tickers_mf.json")
    mapping_dict = {row["symbol"].upper().strip(): row["cik"] for row in mappings}
    cik_mappings = {}

    with open("scratch/FundTicker.csv", "r") as f:
        lines = f.readlines()

    for line in lines:
        ticker = line.split(",")[0].upper().strip()
        if ticker in mapping_dict:
            cik_mappings[ticker] = mapping_dict[ticker]
        else:
            cik = get_mapping_from_secapi(ticker)
            if cik:
                cik_mappings[ticker] = cik

    return cik_mappings


def fill_cik_in_mapping():
    cik_mappings = {}
    n_count = 0
    with open("scratch/ticker_cik_mapping.csv", "r") as f:
        for line in f.readlines():
            parts = line.split(",")
            ticker = parts[0].strip()
            cik = parts[1]
            if cik:
                cik_mappings[ticker] = str(cik)
            else:
                new_cik = get_mapping_from_secapi(ticker)
                if new_cik:
                    cik_mappings[ticker] = str(new_cik)
                    n_count += 1

    print(f"Filled {n_count} CIK mappings")
    return cik_mappings


if __name__ == "__main__":
    # cik_mappings = process_fund_list()
    cik_mappings = fill_cik_in_mapping()
    with open("scratch/ticker_cik_mapping_2.csv", "w") as f:
        for ticker, cik in cik_mappings.items():
            f.write(f"{ticker},{cik}\n")
