import pdfkit as pdf
import os 
import sys

class PDF():
    def __init__(self):
        self.options = {
                "enable-local-file-access": None,
                'dpi': 365,
                'page-size': 'A4',
                'margin-top': '0.0in',
                'margin-right': '0.0in',
                'margin-bottom': '0.0in',
                'margin-left': '0.0in',
                'encoding': "UTF-8",
                'custom-header' : [
                    ('Accept-Encoding', 'gzip')
                ],
                'no-outline': None,
                '--keep-relative-links': '',
            }
        self.create_pdf()
        self.delete_tmp()
            
    def create_pdf(self):
        dir_html = os.path.join("chartanalysis", "templates", "pdf_mail.html")
        dir_pdf = os.path.join("static", "chartanalysis", "img", "result.pdf")
        pdf.from_file(dir_html, dir_pdf, options=self.options)

    def delete_tmp(self):
        dir_img = os.path.join("static", "chartanalysis", "img")
        os.remove(os.path.join(dir_img, "Figure_1.png")) 
        os.remove(os.path.join(dir_img, "Figure_2.png")) 