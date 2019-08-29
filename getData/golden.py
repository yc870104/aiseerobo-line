from bs4 import BeautifulSoup
import requests


def getGolden():
    res = requests.get('https://www.goldlegend.com/')
    soup = BeautifulSoup(res.text, "html.parser")
    buy = soup.find_all(class_="goldprice_tw_buy")[0].text.strip()
    sell=soup.find_all(class_="goldprice_tw_sell")[0].text.strip()
    price=soup.find_all(class_="d-inline-block goldprice_bid")[0].text.strip()
    return str('國際黃金價格:'+price+'美元/盎司\n銀樓買進的黃金價格:' + buy +'元/錢\n銀樓賣出的黃金價格:' + sell+'元/錢')


#print(getGolden())