
from datetime import datetime
import os.path

from googleapiclient.discovery import build
from auth import get_credentials






def get_calendar_service():
    
    creds = get_credentials()

    service = build("calendar", "v3", credentials=creds)
    return service



def create_event(title, start_time, end_time,attendees=None):
    service = get_calendar_service()

    event = {
        "summary": title,
        "start": {
            "dateTime": start_time,
            "timeZone": "UTC"
        },
        "end": {
            "dateTime": end_time,
            "timeZone": "UTC"
        }
    }
    # Ajouter les participants si fournis

    if attendees:
            event["attendees"] = [{"email": email} for email in attendees]


    created_event = (
        service.events()
        .insert(
            calendarId="primary",
            body=event,
            sendUpdates="all" # envoie les invitations par email automatiquement
        )
        .execute()
    )

    return {
        "success": True,
        "event_id": created_event["id"],
        "title": title,
        "event_link": created_event["htmlLink"]
    }


def get_upcoming_events(n=10):

    service = get_calendar_service()

    now = datetime.utcnow().isoformat() + "Z"

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=n,
            singleEvents=True,
            orderBy="startTime"
        )
        .execute()
    )

    return events_result.get("items", [])


def delete_event(event_id):

    service = get_calendar_service()

    service.events().delete(
        calendarId="primary",
        eventId=event_id
    ).execute()

    return {
        "success": True,
        "event_id": event_id
    }