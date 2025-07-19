import smtplib
from email.mime.text import MIMEText

# 连接到SMTP服务器
smtp_server = "smtp.qq.com"
smtp_user = "ricckker@qq.com"
smtp_password = "rbhgjcoshpqqjadi"

server = smtplib.SMTP(smtp_server, 587)
server.starttls()
server.login(smtp_user, smtp_password)

# 创建邮件消息
email_subject = "Test email"
email_body = "Hello, this is a test email!"
msg = MIMEText(email_body)
msg["Subject"] = email_subject
msg["From"] = smtp_user
msg["To"] = "3356203629@qq.com"

# 发送邮件
server.sendmail(smtp_user, "3356203629@qq.com", msg.as_string())

# 断开SMTP服务器连接
server.quit()
