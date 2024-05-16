import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os


#Returns are plotted, if you want these on a 
#-monthly level put 'M'
#-Yearly level put 'Y'
#-Daily level put 'D'
Span='M'

CSV=1 #Do you want your portfolio to be saved in a CSV on your PC. If you come back later, it will load the CSV, and add the to_add.
#If you put it to 0, it will just work with the data provided here:

# Example data (replace with your actual data)
#Enter here your portfolio, if you use CSV, and a portfolio.csv file exists already this will be ignored
portfolio_data = [ 
    {"ticker": "EMIM.AS", "purchase_date": "2024-03-01", "quantity": 90, "total_cost": 2650},
    {"ticker": "EMIM.AS", "purchase_date": "2024-05-01", "quantity": 55, "total_cost": 1750},
    {"ticker": "JPGL.DE", "purchase_date": "2019-09-01", "quantity": 50, "total_cost": 1100},
    {"ticker": "JPGL.DE", "purchase_date": "2019-12-01", "quantity": 50, "total_cost": 1200},
    {"ticker": "JPGL.DE", "purchase_date": "2020-04-01", "quantity": 60, "total_cost": 1100},
    {"ticker": "JPGL.DE", "purchase_date": "2020-09-01", "quantity": 50, "total_cost": 1180},
    {"ticker": "JPGL.DE", "purchase_date": "2020-12-01", "quantity": 40, "total_cost": 1000},
    {"ticker": "JPGL.DE", "purchase_date": "2021-09-01", "quantity": 50, "total_cost": 1500},
    {"ticker": "JPGL.DE", "purchase_date": "2023-03-01", "quantity": 50, "total_cost": 1600},
    {"ticker": "ZPRV.DE", "purchase_date": "2019-01-01", "quantity": 200, "total_cost": 5800},
    {"ticker": "ZPRX.DE", "purchase_date": "2019-01-01", "quantity": 200, "total_cost": 6100},


]


#Do you want to add stocks to your portfolio? It deletes duplicates. If you prefer adding the data directly in the CSV file, that is also possible
to_add = [
    {"ticker": "JPGL.DE", "purchase_date": "2024-03-01", "quantity": 50, "total_cost": 1600},

]


def add_to_portfolio(portfolio, to_add):
    # Convert portfolio to DataFrame
    df = pd.DataFrame(portfolio)
    
    # Convert to_add to DataFrame
    to_add_df = pd.DataFrame(to_add)
    
    # Check if the line already exists in the portfolio
    merged_df = pd.concat([df, to_add_df]).drop_duplicates()
    
    # Update portfolio
    
    merged_df = merged_df.reset_index(drop=True)
    return merged_df

if CSV==1: #Load your current profolio as a CSV if you selected that you want that
    if os.path.isfile('portfolio.csv'):
        # If portfolio.csv exists, read it into a DataFrame
        portfolio_data = pd.read_csv('portfolio.csv')
    else: #if the CSV does not exists yet, use the above descired protfolio_data
        portfolio_data=pd.DataFrame(portfolio_data)
portfolio_data = add_to_portfolio(portfolio_data, to_add)

if CSV ==1: #Save your current profolio as a CSV if you selected that you want that
    portfolio_data.to_csv('portfolio.csv', index=False)


# Fetch historical stock prices
first_date= portfolio_data['purchase_date'].min()
end_date= pd.Timestamp.now() #current data, changable
# Fetch historical stock prices
def get_stock_prices(ticker, start_date, end_date, freq):
    stock_data = yf.download(ticker, interval= freq,start=start_date, end=end_date)
    return stock_data["Close"]


unique_tickers= portfolio_data['ticker'].unique().tolist()

# Calculate portfolio value over time
first_date=pd.to_datetime(pd.DataFrame(portfolio_data)['purchase_date']).min()
end_date= pd.Timestamp.now()

stock_prices=yf.download(unique_tickers,start=first_date, end=end_date)["Close"]
stock_prices=stock_prices.interpolate(method='linear')

purchase_dates=stock_prices.index
#purchase_dates = pd.date_range(start=first_date, end=end_date, freq="D")
portfolio_value = pd.DataFrame(index= purchase_dates)
purchase_price = pd.DataFrame(index= purchase_dates)
stock_prices = stock_prices.reindex(portfolio_value.index)

for ticker in portfolio_data['ticker'].unique():
    portfolio_value[ticker] = 0
    purchase_price[ticker] = 0

# Update portfolio_value and purchase_price DataFrames
for index, row in portfolio_data.iterrows():
    ticker = row['ticker']
    start_date = row['purchase_date']
    portfolio_value.loc[portfolio_value.index >= start_date, ticker] += row['quantity'] * stock_prices[ticker]
    purchase_price.loc[purchase_price.index >= start_date, ticker] += row['total_cost']
# Calculate cumulative costs (purchase price) over time

tot_portfolio_value=portfolio_value.sum(axis=1)  # Return both portfolio value and purchase price
tot_portfolio_cost=purchase_price.sum(axis=1)  # Return both portfolio value and purchase price

# Plot portfolio value and costs
plt.figure(figsize=(10, 6))
plt.plot(tot_portfolio_value.index, tot_portfolio_value, label="Portfolio Value")
plt.plot(purchase_price.index, tot_portfolio_cost, label="Purchase Price", linestyle="--", color="red")
plt.xlabel("Date")
plt.ylabel("Value (€)")
plt.title("Portfolio Value and Purchase Price Over Time")
plt.xticks(rotation=30)

plt.legend()
plt.grid(True)
plt.show()

# Main function



plt.figure(figsize=(10, 6))
plt.plot(tot_portfolio_value.index, tot_portfolio_value/tot_portfolio_cost, label="Portfolio Profit")
plt.xlabel("Date")
plt.ylabel("Value/cost")
plt.xticks(rotation=30)

plt.title("Portfolio profit Over Time")
plt.legend()
plt.grid(True)
plt.show()

nr_etf=portfolio_value.shape[1]
if nr_etf <3:
    rows=nr_etf
    cols=1
elif nr_etf <9:
    rows=int(nr_etf/2+0.999)
    cols=2
elif nr_etf <16:
    rows=int(nr_etf/3+0.999)
    cols=3
elif nr_etf >15:
    rows=int(nr_etf/4+0.999)
    cols=4

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5*cols, 4*cols))

# Flatten the axes array to iterate over it
axes = axes.flatten()
earliest_purchase = portfolio_data.groupby('ticker').agg({'purchase_date': 'min'}).reset_index()


for ax, (index, stock) in zip(axes, earliest_purchase.iterrows()):
    ticker = stock["ticker"]
    start_date = pd.to_datetime(stock["purchase_date"])
    # Filter data for each ticker from its first purchase date onwards
    ticker_portfolio_value = portfolio_value.loc[purchase_dates >= start_date, ticker]
    ticker_purchase_price = purchase_price.loc[purchase_dates >= start_date, ticker]
    
    # Plot portfolio value and costs for each ticker
    ax.plot(ticker_portfolio_value.index, ticker_portfolio_value, label=f"{ticker} Value")
    ax.plot(ticker_purchase_price.index, ticker_purchase_price, label=f"{ticker} Purchase Price", linestyle="--", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (€)")
    ax.set_title(f"{ticker} Value and Price")
    ax.xaxis.set_tick_params(rotation=45)  
    ax.grid(True)

plt.tight_layout()
plt.show()
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5*cols, 4*cols))

# Flatten the axes array to iterate over it
axes = axes.flatten()
for ax, (index, stock) in zip(axes, earliest_purchase.iterrows()):
    ticker = stock["ticker"]
    start_date = pd.to_datetime(stock["purchase_date"])
    # Filter data for each ticker from its first purchase date onwards
    ticker_portfolio_value = portfolio_value.loc[purchase_dates >= start_date, ticker]
    ticker_purchase_price = purchase_price.loc[purchase_dates >= start_date, ticker]
    
    # Plot portfolio value and costs for each ticker
    ax.plot(ticker_portfolio_value.index, ticker_portfolio_value/ticker_purchase_price, label=f"{ticker} Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value/costs")
    ax.set_title(f"{ticker} Value / Price")
    ax.xaxis.set_tick_params(rotation=45)  
    ax.grid(True)

plt.tight_layout()
plt.show()



monthly_return = tot_portfolio_value.resample(Span).ffill().pct_change()*100

# Create a bar plot
fig, ax = plt.subplots()
bars = ax.bar(monthly_return.index, monthly_return, color='grey', width=1200/(monthly_return.shape[0]-1))  # Adjust the width here
for i, bar in enumerate(bars):
    if monthly_return.iloc[i] < 0:
        bar.set_color('red')
    else:
        bar.set_color('green')

# Add labels to the axes
ax.set_xlabel('Date')  # X-axis label
ax.set_ylabel('Return (%)')  # Y-axis label
plt.show()


running_max = tot_portfolio_value.cummax()

# Ensure the result has the same index
running_max.index = tot_portfolio_value.index

# Calculate the drawdown as the difference between the running max and the current price
drawdown = 100*( tot_portfolio_value-running_max )/running_max
plt.figure(figsize=(12,6))
plt.plot(drawdown.index, drawdown)
plt.title('Drawdown of Portfolio Over Time')
plt.xlabel('Time')
plt.ylabel('Drawdown (%)')
plt.grid(True)
plt.show()




print(f'Current portfolio value= {np.round(tot_portfolio_value[-1],2)} \n Current costs= {np.round(tot_portfolio_cost[-1],2)} \n Current returns= {np.round((tot_portfolio_value[-1]/tot_portfolio_cost[-1]-1)*100,4)}% \n Current profits {np.round(tot_portfolio_value[-1]-tot_portfolio_cost[-1],2)}')