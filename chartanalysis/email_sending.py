import yagmail
import os

class Email():
    def __init__(self, mail):
        self.receiver_email = mail
        self.sender_email = "**********"
        self.pw = "********"
        self.dir_img = os.path.join("static", "chartanalysis", "img")
        self.contents = [
            "Thank you for your purchase. Find as attachment the ordered stock analysis. Good Luck with your investments!",
            os.path.join(self.dir_img, "result.pdf")
        ]
        self.send_mail()
    
    def send_mail(self):
        yag = yagmail.SMTP(user = self.sender_email, password = self.pw)
        header ="Your Stock Analysis!"
        yag.send(to = self.receiver_email, subject = header, contents = self.contents)
        os.remove(os.path.join(self.dir_img, "result.pdf")) 


