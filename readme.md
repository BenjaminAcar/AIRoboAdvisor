This project is a Python-Django based website for analysing stocks. For this usage a few benchmark tools are provided as well as the data 
of the stocks itself (using Yahoo Finance API). 

Note: If you do not want to run the script by yourself, the current output of the website is stored in the "screenshots" folder.

But if you want to run it by yourself, follow these steps:
1. Create a virtual environment and activate it (using Anaconda is the fastest way)
2. Install all requirements. Use the "requirements.txt" for this step.
3. Go via shell into the main folder (it includes the "manage.py" script)
4. Type "python manage.py runserver" into the shell and enter it.
5. Finish: Type into the browser URL line the local host IP (http://127.0.0.1:8000/#). Now you can see the Website

Additional Information:
1. \chartanalysis\stock.py contains the main class, that provides the stock analysis tools
2. \chartanalysis\ai_stock.py contains the main class, that provides the AI-based analysis
3. \chartanalysis\views.py handles all server requests
4. \chartanalysis\templates\index.html contains the website itself. The website interacts via AJAX calls with the server/backend.
5. \chartanalysis\pdf.py generates the PDF file.
6. \chartanalysis\email_send.py contains a class to send E-Mails.

All the other files are irrelevant, because most of them are per default created for every Django project.


To send emails, create a localhost server via:
"python -m smtpd -n -c DebuggingServer localhost:1025"


#Nasdaq list:
https://www.nasdaq.com/market-activity/stocks/screener?exchange=nasdaq&letter=0&render=download
