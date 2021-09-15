# 財務分析(3) -- 抓取公開資訊觀測站財務指標  
  
> 因為當沖的tick資料並不好取得, 所以針對台股打算進行日線-月線級別的投資, 風險與成本都相對較小  
> 所以這篇文章想要先從透過財務指標選取績優股來做起  
> 在[財務分析(1) -- 四大財報基本觀念](https://kspace.coderbridge.io/2021/09/12/kfinancial-1/), 以及[財務分析(2) -- 財務指標分析](https://kspace.coderbridge.io/2021/09/14/kfinancial-2/)將財務報表以及指標的概念都做了介紹, 這邊就要針對想要擷取的指標寫成程式  
  
- Target : 程式化抓取公開資訊觀測站財務指標  
- 工具 : python, pandas, requests, BeautifulSoup  
  
# 彙整總表財務分析資料  
原本還想要一個一個指標寫出來, 但後來發現[公開資訊觀測站-彙整總表-財務分析](https://mops.twse.com.tw/mops/web/t51sb02_q1)就已經幫我們把所有資料彙整好算好了, 爽翻!  
![公開資訊觀測站-彙整總表-財務分析](https://i.imgur.com/KVYRKNR.png)  
  
  
- 流動比率, 速動比率, 現金比率, 應收帳款週轉率, 存貨周轉率, 應付帳款周轉率  
- 財務槓桿指數
- 長期資金佔不動產, 廠房及設備比率
- 利息保障倍數
- 每股盈餘(Earning Per Share, EPS)
- 營業比率 (經營比率)
- ROA(總資產報酬率)
- ROE(權益報酬率)
- 利潤率
- 總資產周轉率
- 股利
- 股價
  
除了股利跟股價需要另外抓, 其他這裡面都有了  
那麼要怎麼爬這些資料呢?  
  
# Google 檢查很好用  
利用chrome瀏覽器的右鍵->檢查, 會教出如下的頁面, 這時候把要查詢的動作按下去  
network就會帶出剛剛動作的行為  
![檢查](https://i.imgur.com/6k8rBaC.png)  
這邊我是選擇了時間最長的那段, 裡面顯示了是用POST方法  
以及帶入了下圖的data structure讓後台可以帶出數據出來  
![formdata](https://i.imgur.com/UpnyPz8.png)  
  
觀察完了以後我們就嘗試去coding帶出這些資料  

# 程式碼  
下面特定將get data這段再拆一層get_data_from_mops_twse出來, 因為公開資訊觀測站的網頁, 基本都是用同樣的方式可以去獲取, 只有url, form_data有所不同而已, 以及後處理的column_name會有不同  

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_FinancialAnalysisIndex_year(year, stock_or_otc):
    url = 'https://mops.twse.com.tw/mops/web/t51sb02'
    form_data = {
            'encodeURIComponent':1,
            'run':'Y',
            'step':1,
            'TYPEK':'sii', #otc
            'year': 108,
            'firstin':1,
            'off':1,
            'ifrs':'Y',
        }

    if stock_or_otc == OTC:
        form_data['TYPEK'] = 'otc'

    data = get_data_from_mops_twse(url, form_data)
    df = pd.DataFrame(np.array(data),
                      columns=['Number', 
                               'Name', 
                               'DebtAssetRatio', 
                               'LongTermFundsRealEstateRatio', 
                               'CurrentRatio', 
                               'QuickCurrentRatio', 
                               'InterestProtectionMutiple',
                               'AccountsReceivableTurnover',
                               'AverageDaysOfCashCollection',
                               'InventoryTurnoverRate',
                               'AverageSalesDays',
                               'AverageSalesRealEstateTurover',
                               'TotalAssetsWeekTurnover',
                               'ROA',
                               'ROE',
                               'NetProfitBeforeTaxAccountForPaid-inCapitalTheRatio',
                               'NetProfitRatio',
                               'EPS',
                               'CashFlowRatio',
                               'CashFlowAllowableRatio',
                               'CashReinvestmentCapitalRatio']
                      )
    #print(df)
    return df
	
def get_data_from_mops_twse(url, form_data):
    response = requests.post(url,form_data)
    response.encoding = 'utf8'
    text = response.text
    soup = BeautifulSoup(text, 'html.parser')    
    tables = soup.find_all("table", {"class" : "hasBorder"})
    data = []
    for tb in table:
        for row in tb.find_all("tr"):
            #print(row.text)
            tempdata = []
            for col in row.find_all('td'):
                col.attrs = {}
                #print(col.text.strip().replace('\u3000', ''))
                tempdata.append(col.text.strip().replace('\u3000', ''))
            if len(tempdata) <= 1:
                pass 
            else:
                data.append(tempdata)
                #print(tempdata)
    return data
```
同理如果我們想要營業利益分析表的數據, 只需要把url跟form_data換一下就可以  
```python
def get_ProfitAnalysis_season(year, season, stock_or_otc):
    
    url = 'https://mops.twse.com.tw/mops/web/t163sb06'
    form_data = {

            'encodeURIComponent':1,
            'isQuery':'Y',
            'step':1,
            'TYPEK':'sii', #otc
            'year': 108,
            'firstin':1,
            'off':1,
            'season':season,
        }

    if stock_or_otc == OTC:
        form_data['TYPEK'] = 'otc'

    data = get_data_from_mops_twse(url, form_data)
    
    df = pd.DataFrame(np.array(data),
                      columns=['Number', 
                               'Name', 
                               'OperatingIncome', 
                               'GrossMargin', 
                               'OperatingProfitRatio', 
                               'NetProfitRateBeforeTax', 
                               'NetProfitAfterTax',
                               ]
                      )
    #print(df)
    return df
```

# 結尾  
在公開資訊觀測站的網頁大多可以透過此方式去獲得資料, 如果不想抓取彙整過的指標, 想要直接分析四大報表也都可以透過此方式獲得, 不過這樣的資料應該夠我們做初步的篩選了, 所以下一步應該會先根據這些指標以及先前教過的判斷方式對股票做篩選  
  
---  
目前針對財務分析這塊使用python 爬公開資訊觀測站, 只有爬了財務分析跟營益分析  
後續還想要將每月營收, 搭配yfinance的股價, 以及股利等數據下載下來, 做一個後臺  
開發算法  
以及風險控制  
做一個自動化投資建議系統, 如果有興趣一起加入的夥伴, 歡迎聯繫  
  
**如果喜歡文章, 不妨按下喜歡訂閱支持**  
  
如果真的想支持我進行創作與實踐計畫, 也可以進行打賞  
BTC (BTC) : 1LFRBqvWR9GGizBzoFddkb5xRpAzKRVoDC  
BTC (BSC) : 0xe1cda3eb778d1751af17feac796a4bbe4dbca815  
BTC (ERC20) : 0xe1cda3eb778d1751af17feac796a4bbe4dbca815  
USDT (TRC20) : TT7wgKuYoHwfRy3vCr38Qy3gnS3JJ1aKvn  
  
如果想使用幣安, 可以使用我的推薦連結可以**[節省手續費10%](https://accounts.binance.com/zh-TW/register?ref=A5YRMD2T)  
或使用推薦碼 : A5YRMD2T**  
