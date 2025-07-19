import smtplib
from email.mime.text import MIMEText
import random


def send_email(message, recipient_email):
    smtp_server = "smtp.qq.com"
    smtp_user = "ricckker@qq.com"
    smtp_password = "rbhgjcoshpqqjadi"
    server = smtplib.SMTP(smtp_server, 587)
    server.starttls()
    server.login(smtp_user, smtp_password)
    email_subject = "注册验证码"
    email_body = message
    msg = MIMEText(email_body)
    msg["Subject"] = email_subject
    msg["From"] = smtp_user
    msg["To"] = recipient_email
    server.sendmail(smtp_user, recipient_email, msg.as_string())
    server.quit()


def generate_random_string(length):
    letters = "0123456789"
    random_string = ""
    for _ in range(length):
        random_string += random.choice(letters)
    return random_string
