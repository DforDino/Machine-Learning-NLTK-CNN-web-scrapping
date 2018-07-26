#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 06:04:09 2018

@author: dino
"""

import bs4

from urllib.request import urlopen as uReq

from bs4 import BeautifulSoup as soup

my_url = 'https://www.newegg.com/Video-Cards-Video-Devices/Category/ID-38?Tpk=graphics%20card'

#opening up connection, grabbing the page
uClient = uReq(my_url)
#Offloading html data and storing it after reading
page_html = uClient.read()
#closing the connection
uClient.close()
#html parsing
page_soup = soup(page_html,"html.parser")
#grab each product
containers = page_soup.find_all("div",{"class":"item-container"})

filename = "products.csv"

f = open(filename, "w")

headers = "product_brand, product_title, product_price, product_shipping \n"

f.write(headers)

for container in containers:

    product_brand = container.div.div.a.img["title"]
    
    title_container = container.find_all("a",{"class":"item-title"})
    
    product_title = title_container[0].text
    #if all prices are listed,then following command will work fine
    
    product_price = '$'+container.find_all("li",{"class":"price-current"})[0].strong.text+container.find_all("li",{"class":"price-current"})[0].sup.text
    
    product_shipping = container.find_all("li",{"class":"price-ship"})[0].text.strip()

    print("product_brand:" + product_brand)
    print("product_title:" + product_title)
    print("product_price:" + product_price)
    print("product_shipping:" + product_shipping)
    #since , presents in title that will hamper the columns in csv format hence we replace all , to | in the title
    f.write(product_brand+","+product_title.replace(",","|")+","+product_price+","+product_shipping+ "\n")
    
f.close()
    
    
    
    