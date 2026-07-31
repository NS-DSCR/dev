import requests
import json
import logging
from langchain_core.tools import tool
from typing import Optional

logger = logging.getLogger(__name__)

def _send_to_n8n(webhook_url: str, action: str, data: dict) -> str:
    """Internal helper to send data to the n8n bridge"""
    if not webhook_url:
        return "Error: No Webhook URL configured for this agent."
    
    payload = {
        "action": action,
        "payload": data
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=15)
        if response.status_code in [200, 201]:
            return f"Success: {action} triggered. Response: {response.text[:100]}"
        return f"Error: n8n returned {response.status_code}"
    except Exception as e:
        return f"Connection Failed: {str(e)}"

@tool
def send_gmail_message(recipient: str, subject: str, body: str, webhook_url: str = "") -> str:
    """
    Sends an email via Gmail. 
    Requires recipient email, subject, and the message body.
    """
    return _send_to_n8n(webhook_url, "gmail_send", {
        "to": recipient,
        "subject": subject,
        "body": body
    })

@tool
def create_calendar_event(title: str, start_time: str, end_time: Optional[str] = None, description: str = "", webhook_url: str = "") -> str:
    """
    Schedules a meeting or event in Google Calendar.
    start_time and end_time should be in ISO format (e.g. 2024-05-20T10:00:00).
    """
    return _send_to_n8n(webhook_url, "calendar_create", {
        "title": title,
        "start": start_time,
        "end": end_time,
        "description": description
    })

@tool
def call_marketplace_connector(connector_id: str, action_details: str, payload_json: str, webhook_url: str = "") -> str:
    """
    Triggers a marketplace connector (e.g., Slack, Hubspot, Shopify).
    'connector_id' is the ID of the app (e.g. 'slack').
    'action_details' is a short description of what the AI is doing.
    'payload_json' is the structured data needed for the action.
    """
    return _send_to_n8n(webhook_url, f"marketplace_{connector_id}", {
        "action_description": action_details,
        "data": payload_json
    })

@tool
def send_discord_message(content: str, webhook_url: str = "") -> str:
    """
    Sends a message to a Discord channel via a Webhook.
    'content' is the text message to send.
    """
    return _send_to_n8n(webhook_url, "discord_send", {
        "content": content
    })

@tool
def trigger_external_workflow(payload_json: str, webhook_url: str = "") -> str:
    """
    Generic tool to trigger any external automation workflow.
    Use this for Slack, Microsoft Teams, or custom webhooks.
    """
    try:
        payload = json.loads(payload_json)
        return _send_to_n8n(webhook_url, "custom_action", payload)
    except Exception as e:
        return f"Invalid JSON: {str(e)}"
