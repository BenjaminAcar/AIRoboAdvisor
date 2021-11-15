from django.shortcuts import render, redirect
import json
from flask import Flask, request, render_template
from django.http import HttpResponseRedirect, HttpResponse
from django.http import JsonResponse
from chartanalysis import stock
import plotly
from chartanalysis import methods
from chartanalysis import ai_stock
from chartanalysis import pdf
from chartanalysis import email_sending
"""
views.py is the script that handles the requests to the server.
Every request gets mapped via "urls.py" and leads to triggering one of the following functions.
The return is always an HTTP response, including the current plot as a HTML object (div code)
"""

#by default the AAPL stock is chosen (to avoid an empty GUI)
#npm install -- save sweetalert2
targetStock = stock.Stock("AAPL")

def products(request):
    print("DRIN")
    return redirect('/products/')
    
def index(request):
    headerStocks = methods.getHeaderStocks()
    stockList = methods.getProposals()
    context = {
        'currentGraph': targetStock.currentGraph,
        'stockList': stockList,
        'stock_1': headerStocks[0],
        'stock_2': headerStocks[1],
        'stock_3': headerStocks[2],
        'stock_4': headerStocks[3],
        'stock_5': headerStocks[4],
        'stock_6': headerStocks[5],
        'stock_7': headerStocks[6],
        'stock_8': headerStocks[7],
        'stock_9': headerStocks[8],
        'stock_10':headerStocks[9],
        'stock_11':headerStocks[10],
        'stock_12':headerStocks[11],
        'stock_13':headerStocks[12],
        'stock_14':headerStocks[13],
        'stock_15':headerStocks[14],
    }
    return render(request, 'index.html', context = context)

def addResistanceButton(request):
    if request.method == 'GET':
        try:
            targetStock.calcLines("Resistance")
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)
        except:
            print("Adding Resistance Lines failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getMonteCarlo(request):
        if request.method == 'GET':
            try:
                targetStock.createMonteChart()
                return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
            except:
                print("Plotting Monte Carlo Graph failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
                return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getLinearRegression(request):           
        if request.method == 'GET':
            try:
                targetStock.createLinearRegression() 
                return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
            except:
                print("Plotting Linear Regression Graph failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
                return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getRSI(request):
        if request.method == 'GET':
            try:
                targetStock.createRSI()
                return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
            except:
                print("Plotting RSI Graph failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
                return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)



def getAnalysis(request):
    ticket = request.GET["ticket"]
    mail = request.GET["mail"]
    ai_stock.AI(ticket)
    pdf.PDF()
    email_sending.Email(mail)
    return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)

def getCandleChart(request):
    if request.method == 'GET':
        try:
            targetStock.createCandleChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting Candle Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getADI(request):
    if request.method == 'GET':
        try:
            targetStock.createADIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting Bollinger Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getVWAP(request):
    if request.method == 'GET':
        try:
            targetStock.createVWAPChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VWAP Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getCMF(request):
    if request.method == 'GET':
        try:
            targetStock.createCMFChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting CMF Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getEMV(request):
    if request.method == 'GET':
        try:
            targetStock.createEMVChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting EMV Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getFI(request):
    if request.method == 'GET':
        try:
            targetStock.createFIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting FI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getMFI(request):
    if request.method == 'GET':
        try:
            targetStock.createMFIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting MFI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)


def getAO(request):
    if request.method == 'GET':
        try:
            targetStock.createAOChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getSO(request):
    targetStock.createSOChart()
    if request.method == 'GET':
        try:
            targetStock.createSOChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getKAMA(request):
    if request.method == 'GET':
        try:
            targetStock.createKAMAChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getPPO(request):
    targetStock.createPPOChart()
    if request.method == 'GET':
        try:
            targetStock.createPPOChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getPVO(request):
    if request.method == 'GET':
        try:
            targetStock.createPVOChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getROC(request):
    if request.method == 'GET':
        try:
            targetStock.createROCChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getSRSI(request):
    if request.method == 'GET':
        try:
            targetStock.createSRSIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getUO(request):
    if request.method == 'GET':
        try:
            targetStock.createUOChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getWRI(request):
    if request.method == 'GET':
        try:
            targetStock.createWRIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)



def getNVI(request):
    if request.method == 'GET':
        try:
            targetStock.createNVIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting NVI Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getOBV(request):
    if request.method == 'GET':
        try:
            targetStock.createOBVChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting OBV Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)


def getVPT(request):
    if request.method == 'GET':
        try:
            targetStock.createVPTChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)


def getART(request):
    if request.method == 'GET':
        try:
            targetStock.createATRChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)
def getDC(request):
    if request.method == 'GET':
        try:
            targetStock.createDCChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)
def getKC(request):
    if request.method == 'GET':
        try:
            targetStock.createKCChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)
def getUI(request):
    if request.method == 'GET':
        try:
            targetStock.createUIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)


#new
def getADX(request):
    if request.method == 'GET':
        try:
            targetStock.createADXChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getADX(request):
    if request.method == 'GET':
        try:
            targetStock.createADXChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getADX(request):
    if request.method == 'GET':
        try:
            targetStock.createADXChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getAI(request):
    if request.method == 'GET':
        try:
            targetStock.createAIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getCCI(request):
    if request.method == 'GET':
        try:
            targetStock.createCCIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getDPO(request):
    if request.method == 'GET':
        try:
            targetStock.createDPOChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getEMA(request):
    if request.method == 'GET':
        try:
            targetStock.createEMAChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getII(request):
    if request.method == 'GET':
        try:
            targetStock.createIIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getKST(request):
    if request.method == 'GET':
        try:
            targetStock.createKSTChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getMACD(request):
    if request.method == 'GET':
        try:
            targetStock.createMACDChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getMI(request):
    if request.method == 'GET':
        try:
            targetStock.createMIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getPSAR(request):
    if request.method == 'GET':
        try:
            targetStock.createPSARChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getSTC(request):
    if request.method == 'GET':
        try:
            targetStock.createSTCChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getTRIX(request):
    if request.method == 'GET':
        try:
            targetStock.createTRIXChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getVI(request):
    if request.method == 'GET':
        try:
            targetStock.createVIChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)


def getWMA(request):
    if request.method == 'GET':
        try:
            targetStock.createWMAChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting VPT Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getBollingerBands(request):
    if request.method == 'GET':
        try:
            targetStock.createBollingerBands()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting Bollinger Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def getBasicChart(request):
    if request.method == 'GET':
        try:
            targetStock.createBasicChart()
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Plotting Basic Chart failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def addSupportButton(request):
    if request.method == 'GET':
        try:
            targetStock.calcLines("Support")
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)
        except:
            print("Adding Support Lines failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def deleteLinesButton(request):
    if request.method == 'GET':
        try:
            targetStock.fig['layout']['shapes'] = ()
            targetStock.currentGraph = plotly.offline.plot(targetStock.fig, include_plotlyjs=False, output_type='div')
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)
        except:
            print("Deleting Fibonacci Lines failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def updateTimespan(request):
    if request.method == 'GET':
        try:
            if len(request.GET.keys()) >= 1:
                timespan = (''.join(list(request.GET.keys()))).replace(" ", "")
                targetStock.getStartByInput(timespan)
                targetStock.getStock()
                if targetStock.graphStamp == "Basic":
                    targetStock.createBasicChart()
                if targetStock.graphStamp =="BollingerBands":
                    targetStock.createBollingerBands()
                if targetStock.graphStamp =="Candle":
                    targetStock.createCandleChart()
                if targetStock.graphStamp =="Monte":
                    targetStock.createMonteChart()    
                if targetStock.graphStamp =="LinearRegression":
                    targetStock.createLinearRegression()
                if targetStock.graphStamp =="NonlinearRegression":
                    targetStock.createNonLinearRegression()
                if targetStock.graphStamp =="RSI":
                    targetStock.createRSI()
                if targetStock.graphStamp =="ADI":
                    targetStock.createADIChart()
                if targetStock.graphStamp =="VWAP":
                    targetStock.createVWAPChart()
                if targetStock.graphStamp =="CMF":
                    targetStock.createCMFChart()
                if targetStock.graphStamp =="EMV":
                    targetStock.createEMVChart()
                if targetStock.graphStamp =="FI":
                    targetStock.createFIChart()
                if targetStock.graphStamp =="MFI":
                    targetStock.createMFIChart()
                if targetStock.graphStamp =="NVI":
                    targetStock.createNVIChart()
                if targetStock.graphStamp =="OBV":
                    targetStock.createOBVChart()
                if targetStock.graphStamp =="VPT":
                    targetStock.createVPTChart()
                if targetStock.graphStamp =="AO":
                    targetStock.createAOChart()
                if targetStock.graphStamp =="KAMA":
                    targetStock.createKAMAChart()
                if targetStock.graphStamp =="PPO":
                    targetStock.createPPOChart()
                if targetStock.graphStamp =="PVO":
                    targetStock.createPVOChart()
                if targetStock.graphStamp =="ROC":
                    targetStock.createROCChart()
                if targetStock.graphStamp =="SRSI":
                    targetStock.createSRSIChart()
                if targetStock.graphStamp =="SO":
                    targetStock.createSOChart()
                if targetStock.graphStamp =="UO":
                    targetStock.createUOChart()
                if targetStock.graphStamp =="WRI":
                    targetStock.createWRIChart()
                if targetStock.graphStamp =="ATR":
                    targetStock.createATRChart()
                if targetStock.graphStamp =="DC":
                    targetStock.createDCChart()
                if targetStock.graphStamp =="KC":
                    targetStock.createKCChart()
                if targetStock.graphStamp =="UI":
                    targetStock.createUIChart()


                if targetStock.graphStamp =="AI":
                    targetStock.createAIChart()
                if targetStock.graphStamp =="CCI":
                    targetStock.createCCIChart()
                if targetStock.graphStamp =="DPO":
                    targetStock.createDPOChart()
                if targetStock.graphStamp =="II":
                    targetStock.createIIChart()
                if targetStock.graphStamp =="KST":
                    targetStock.createKSTChart()
                if targetStock.graphStamp =="MACD":
                    targetStock.createMACDChart()
                if targetStock.graphStamp =="MI":
                    targetStock.createMIChart()
                if targetStock.graphStamp =="PSAR":
                    targetStock.createPSARChart()
                if targetStock.graphStamp =="STC":
                    targetStock.createSTCChart()
                if targetStock.graphStamp =="TRIX":
                    targetStock.createTRIXChart()
                if targetStock.graphStamp =="VI":
                    targetStock.createVIChart()
                if targetStock.graphStamp =="WMA":
                    targetStock.createWMAChart()
                
                targetStock.updateParam()
                return JsonResponse({'currentGraph': targetStock.currentGraph, "daysBetween": targetStock.daysBetween()}, safe = False)
            else:
                return JsonResponse({'currentGraph': targetStock.currentGraph, "daysBetween": targetStock.daysBetween()}, safe = False)
        except:
            print("Updating Timespan failed: Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def closeFullScreen(request):
    if request.method == 'GET':
        try:
            targetStock.closeFullScreen() 
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Resizing Graph failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def fullScreen(request):
    if request.method == 'GET':
        try:
            targetStock.fullScreen() 
            return JsonResponse({'currentGraph': targetStock.currentGraph}, safe = False)
        except:
            print("Resizing Graph failed. Ticket Symbol:" + str(targetStock.symbolOfStock))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)

def findStock(request):
    if request.method == 'GET':
        try:
            searchForStock = targetStock.stockSearch(request)
            if searchForStock[0] == True:
                return JsonResponse({'Name': searchForStock[1], 'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)
            else:
                return JsonResponse({'Name': "No match!"}, safe = False)
        except:
            print("Finding Ticket Symbol failed. Request:" + str(request))
            return JsonResponse({'currentGraph': targetStock.currentGraph, 'daysBetween': targetStock.daysBetween()}, safe = False)