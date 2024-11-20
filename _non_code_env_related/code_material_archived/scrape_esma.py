# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By

from capitalfour.quant import database as c4_db


import time
import os
import shutil

user_name = os.path.expanduser("~")[-3:]  # get initials for user
path_to_chromedriver = "C:\\Users\\" + user_name + "\\Documents\\chromedriver_win32\\chromedriver.exe" # change path as needed

download_dir = "C:\\Users\\" + user_name + "\\Downloads"
#Specify where files should end up
new_dir = "C:\\Users\\" + user_name + "\\OneDrive - Capital Four Management Fondsmæglerselskab A S\\Documents\\NLP_PROJECT\\HP00_pre2020"
chrome_options = webdriver.ChromeOptions()
preferences = {"download.default_directory": download_dir ,
                      "directory_upgrade": True,
                      "safebrowsing.enabled": True }
chrome_options.add_experimental_option("prefs", preferences)
driver = webdriver.Chrome(chrome_options=chrome_options)

capabilities = {'chromeOptions': {'useAutomationExtension': False}}


browser = webdriver.Chrome(executable_path = path_to_chromedriver, desired_capabilities=capabilities)


def get_prospectus(browser, isin):
    url = 'https://registers.esma.europa.eu/publication/searchRegister?core=esma_registers_priii_securities'
    browser.get(url)
    
    time.sleep(1)
    #Find ISIN field
    input_isin = browser.find_element(By.XPATH,'//*[@id="searchFields"]/div[1]/div/input')
    
    #Send ISIN
    input_isin.clear()
    input_isin.send_keys(isin)
    
    time.sleep(2)
    #Search
    search_button = browser.find_element(By.XPATH,'//*[@id="searchSolrButton"]')
    search_button.click()
    
    #Sleep to accommodate loading
    time.sleep(5)
    
    #Press
    prosp_button= browser.find_element(By.XPATH,'//*[@id="T01"]/tbody/tr/td[11]/a')
    prosp_button.click()
    
    #Sleep to accommodate loading
    time.sleep(5)
    
    #Expand
    for val in range(7):
        try:
            time.sleep(1)
            exp_button= browser.find_element(By.XPATH,'//*[@id="ui-id-' + str(val) + '"]')
            exp_button.click()
        except:
            print(val)
    
    time.sleep(5)
    #Click document
    try:
        doc_button = browser.find_element(By.XPATH, '//*[@id="prospectuses"]/tbody/tr/td[2]/a')
        new_url = doc_button.get_attribute('href')
    except Exception as e:
        print('Trying other approach')
        doc_button = browser.find_element(By.XPATH, '//*[@id="related_docs"]/tbody/tr/td[2]/a')
        new_url = doc_button.get_attribute('href')
    
    print(new_url)
    browser.get(new_url)
    time.sleep(5)
    for val in range(10):
        try:
            time.sleep(1)
            browser.find_element(By.XPATH, '//*[@id="detailsParent"]/tbody/tr[' + str(val) + ']/td[2]/a').click()
            #exp_button = WebDriverWait(browser, 10).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="ui-id-5"]')))
        except:
            print(val)
    
    #wait for download to finish
    time.sleep(20)
    filename = max([download_dir +"\\"+ f for f in os.listdir(download_dir)], key=os.path.getctime)
    shutil.move(filename, os.path.join(new_dir,isin + '.pdf'))
    time.sleep(1)



db = c4_db.Database()

# UR00 - Sterling Corporate Index
# ER00 - Euro Corp IG
# COCO - Contingent Capital Index
# HP00 - European Currency High Yield Index
# H0A0 - US High Yield Index
# EMHB
# C0A0 - US Corp IG
# HE0F 

#Data only available from 2020 and forward
#PH: dynamic indeholder løbende ændrende data på bonds mange rows per bond (97 cols)
#PH: static indeholder statisk data 1 row per bond(10 cols)
sql_read = """ SELECT distinct stat.ISIN_number
                FROM [CfQuant].[Bond].[Constituents_Dynamic] as dyn
                JOIN [CfQuant].[Bond].[Constituents_Static] as stat
                on dyn.FK_Bond_id = stat.Bond_id
                WHERE dyn.As_of_Date >= '2020-01-01'
                AND dyn.Original_index = 'HP00'
                  """

names = db.read_sql(sql_read)

for name in names.values:
    print(name)
    try:
        get_prospectus(browser, name[0])
    except Exception as e:
        print(e)
        continue
    
browser.close()

