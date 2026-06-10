import base64
from datetime import datetime
import os.path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from googleapiclient.discovery import build
from auth import get_credentials


def get_gmail_service():
    
    creds = get_credentials()

    service = build("gmail", "v1", credentials=creds)
    
    return service



def get_latest_emails(n):
    service = get_gmail_service()

    results = service.users().messages().list(userId="me", maxResults=n, q="in:inbox").execute()
    messages = results.get("messages", [])

    emails = []

    for msg in messages:
        txt = service.users().messages().get(userId="me", id=msg["id"]).execute()

        payload = txt.get("payload", {})
        headers = payload.get("headers", [])

        subject = ""
        to = ""
        sender  = ""
        date = ""

        for h in headers:
            if h["name"] == "Subject":
                subject = h["value"]
            elif h["name"] == "To":
                to = h["value"]
            elif h["name"] == "From":
                sender = h["value"]
            elif h["name"] == "Date":
                date = h["value"]
        
        snippet = txt.get("snippet","")
        
        body = ""
        try:
            data = payload["body"]["data"]
            decoded_data = base64.urlsafe_b64decode(data).decode("utf-8")
            body = decoded_data
        except:
            body = "No readable body"



        emails.append({
            "id": msg["id"],
            "from":sender,
            "to":to,
            "date":date,
            "subject": subject,
            "snippet": snippet,
            "body":body
        })

    return emails


def send_email(to, subject, body):
    service = get_gmail_service()

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    message = {"raw": raw}

    sent_message = service.users().messages().send(userId = "me", body = message).execute()

    return {

        "success": True,

        "message": "Email sent successfully",

        "recipient": to,

        "subject": subject,

        "gmail_message_id": sent_message["id"],

        "thread_id": sent_message["threadId"],

        "timestamp": datetime.now().isoformat()

    }




def send_email_with_attachment(to, subject, body, file_paths):
    service = get_gmail_service()

    message = MIMEMultipart()
    message["subject"] = subject
    message["to"] = to

    message.attach(MIMEText(body, "plain"))

    for file_path in file_paths :
        part = MIMEBase("application", "octet-stream")
        file_name = os.path.basename(file_path)

        with open(file_path, "rb") as attachement :

            part.set_payload(attachement.read())

        encoders.encode_base64(part)

        part.add_header("Content-Disposition",f"attachment; filename = {file_name}")
        message.attach(part)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {"raw": raw}

    sent_message = service.users().messages().send(userId = "me", body = body).execute()
    
    return {

        "success": True,

        "message": "Email sent successfully",

        "recipient": to,

        "subject": subject,

        "gmail_message_id": sent_message["id"],

        "thread_id": sent_message["threadId"],

        "timestamp": datetime.now().isoformat()

    }


