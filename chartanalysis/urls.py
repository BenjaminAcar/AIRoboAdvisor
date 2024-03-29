from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('/products/', views.products, name='products'),
    path('get_resistance', views.addResistanceButton, name='get_resistance'),
    path('get_support', views.addSupportButton, name='get_support'),
    path('delete_lines', views.deleteLinesButton, name='delete_lines'),
    path('get_stock', views.findStock, name='get_stock'),
    path('timespan_update', views.updateTimespan, name='timespan_update'),
    path('candle_chart', views.getCandleChart, name='candle_chart'),
    path('bollinger_bands', views.getBollingerBands, name='bollinger_bands'),
    path('basic_chart', views.getBasicChart, name='basic_chart'),
    path('monte_carlo', views.getMonteCarlo, name='monte_carlo'),
    path('linear_regression', views.getLinearRegression, name='linear_regression'),
    path('get_rsi', views.getRSI, name='get_rsi'),
    path('full_screen', views.fullScreen, name='full_screen'),
    path('close_full_screen', views.closeFullScreen, name='close_full_screen'),
    path('adi', views.getADI, name='adi'),
    path('vwap', views.getVWAP, name='vwap'),
    path('cmf', views.getCMF, name='cmf'),
    path('emv', views.getEMV, name='emv'),
    path('fi', views.getFI, name='fi'),
    path('mfi', views.getMFI, name='mfi'),
    path('nvi', views.getNVI, name='nvi'),
    path('obv', views.getOBV, name='obv'),
    path('vpt', views.getVPT, name='vpt'),
    path('ao', views.getAO, name='ao'),
    path('so', views.getSO, name='so'),
    path('kama', views.getKAMA, name='kama'),
    path('ppo', views.getPPO, name='ppo'),
    path('pvo', views.getPVO, name='pvo'),
    path('roc', views.getROC, name='roc'),
    path('srsi', views.getSRSI, name='srsi'),
    path('uo', views.getUO, name='uo'),
    path('wri', views.getWRI, name='wri'),
    path('art', views.getART, name='art'),
    path('dc', views.getDC, name='dc'),
    path('kc', views.getKC, name='kc'),
    path('ui', views.getUI, name='ui'),
    path('ai', views.getAI, name='ai'),
    path('cci', views.getCCI, name='cci'),
    path('dpo', views.getDPO, name='dpo'),
    path('ii', views.getII, name='ii'),
    path('kst', views.getKST, name='kst'),
    path('macd', views.getMACD, name='macd'),
    path('mi', views.getMI, name='mi'),
    path('psar', views.getPSAR, name='psar'),
    path('stc', views.getSTC, name='stc'),
    path('trix', views.getTRIX, name='trix'),
    path('vi', views.getVI, name='vi'),
    path('wma', views.getWMA, name='wma'),
    path('analysis', views.getAnalysis, name='analysis'),
]

