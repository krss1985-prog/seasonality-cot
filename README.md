# COT & Seasonality Analysis

This project analyzes Commitment of Traders (COT) data and combines it with price seasonality for SPX (^GSPC), Gold (GC=F), and Oil (CL=F) using Python. It fetches price data from Yahoo Finance, calculates daily and weekly seasonality signals, and prepares for merging with COT data for macro bias analysis.

## Features
- Fetches price data from Yahoo Finance
- Calculates daily and weekly seasonality scores and signals
- Prepares for merging with COT data
- Ready for further macro bias and cycle analysis

## Usage
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```
   python yahoo_seasonality_pack.py
   ```
3. Output files:
   - prices_daily_raw.csv
   - seasonality_daily.csv
   - seasonality_weekly.csv

## Next Steps
- Merge COT data and calculate total bias
- Add cycle seasonality and macro scoring

## Tickers Used
- SPX: ^GSPC
- Gold: GC=F
- Oil: CL=F

## Requirements
- Python 3.8+
- Internet connection for Yahoo Finance data

## License
MIT
