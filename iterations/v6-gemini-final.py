import os
import streamlit as st
from streamlit_lottie import st_lottie
# Streamlit page configuration must be the first Streamlit command
st.set_page_config(
    page_title="AI Powered CRM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory
import supabase
from supabase.client import Client, ClientOptions
from pathlib import Path
import uuid
import sys
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type
import re
import openai
import locale
from PyPDF2 import PdfReader, PdfWriter
from thefuzz import fuzz
import telegram
import asyncio
import schedule
import threading
import time

def format_currency_with_commas(amount):
    """Format currency amounts with thousands separators"""
    try:
        # Set locale for thousands separator
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        return locale.format_string("%.2f", amount, grouping=True)
    except:
        # Fallback if locale is not available
        return f"{amount:,.2f}"
import datetime
import pandas as pd
import json
import requests
import urllib.parse
from serpapi import GoogleSearch  # Keep SerpAPI for combined results
from bs4 import BeautifulSoup
import numpy as np
from fpdf import FPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io

# --- Custom CSS for beautiful UI ---
st.markdown("""
<style>
/* Global Styles */
body {
    background-color: #f5f7fa; /* Light background */
    font-family: 'Segoe UI', sans-serif;
}

/* Adjust main content area padding for spacious layout */
.main .block-container {
    padding-top: 40px;
    padding-right: 60px;
    padding-left: 60px;
    padding-bottom: 40px;
}

/* Sidebar styling */
.css-1d391kg, .css-1v0mbdj { /* Adjust these class names if needed based on Streamlit version */
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Slightly stronger shadow */
    margin-bottom: 20px;
}

/* Input fields */
input {
    border: 1px solid #e0e0e0; /* Lighter border */
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
    width: 100%;
    margin-bottom: 15px; /* Increased space */
    background-color: #f9f9f9; /* Slightly different input background */
}

/* Hide 'Press Enter to apply' text */
.stTextInput > div > div > input + div {
    display: none;
}

/* Buttons */
button[kind="primary"] {
    background-color: #1e73c4; /* A shade of blue */
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: bold;
    margin-top: 15px; /* Increased space */
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s ease;
}

button[kind="primary"]:hover {
    background-color: #155b9e; /* Darker shade on hover */
}

/* Header styles */
h1, h2, h3, h4 {
    color: #333; /* Darker text for headers */
    font-weight: 700;
    margin-bottom: 15px;
    padding-top: 5px; /* Add some padding above headers */
}

/* Features layout (Cards) */
.feature-box {
    background-color: #ffffff;
    padding: 25px; /* Increased padding inside cards */
    border-radius: 16px; /* Rounded corners */
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1); /* More prominent shadow */
    margin-top: 15px; /* Increased margin */
    margin-bottom: 15px; /* Increased margin */
    text-align: left;
    height: 100%;
    transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth hover effect */
}

.feature-box:hover {
    transform: translateY(-5px); /* Lift effect on hover */
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15); /* Enhanced shadow on hover */
}

.stMarkdown > div > p {
    margin-bottom: 1rem; /* Standard space below paragraphs */
    color: #555; /* Slightly lighter text for paragraph */
}

/* Adjust spacing around columns */
.st-emotion-l8z9g2 > div { /* This targets the div inside the columns, adjust class name if needed */
    margin-bottom: 30px; /* Space between rows of columns */
    padding: 0 10px; /* Add horizontal padding between columns */
}

/* Center the main content block */
.css-18e3gdp.e8zbici2 {
    max-width: 1200px; /* Set a max width for content */
    margin: auto; /* Center the block */
}

/* Class for bold text */
.bold-text {
    font-weight: bold;
}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #1e1e1e; /* Darker background */
    }
    .stApp {
        background-color: #1e1e1e;
    }
    .main .block-container {
        padding-top: 40px;
        padding-right: 60px;
        padding-left: 60px;
        padding-bottom: 40px;
    }
    .css-1d391kg, .css-1v0mbdj {
        background-color: #2c2c2c;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    input {
        border: 1px solid #444;
        background-color: #333;
        color: #f0f0f0;
    }
    button[kind="primary"] {
        background-color: #1a4d7d; /* Dark mode blue */
    }
    button[kind="primary"]:hover {
        background-color: #12395a; /* Darker shade on hover */
    }
    h1, h2, h3, h4 {
        color: #f0f0f0;
    }
    .feature-box {
        background-color: #2c2c2c;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.45);
    }
     .stMarkdown > div > p {
        color: #cccccc;
    }
}

</style>
""", unsafe_allow_html=True)

# Module-level notification scheduler state to avoid multiple threads/jobs
_notif_scheduler_started_global = False
_notif_scheduler_thread = None
_notif_scheduler_lock = threading.Lock()

class GeminiServiceUnavailableError(Exception):
    """Raised when Gemini returns 503 due to high demand."""
    pass

class GeminiRateLimitError(Exception):
    """Raised when Gemini returns 429 quota/rate-limit errors."""
    pass

# Load environment variables
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path, override=True)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")
supabase_client = supabase.create_client(supabase_url, supabase_key)

model = os.getenv('MODEL_CHOICE', 'gpt-3.5-turbo')

# --- Gemini integration (minimal, no new requirements) ---
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini').lower()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', None)
GEMINI_CHAT_MODEL = os.getenv('GEMINI_CHAT_MODEL', 'gemini-2.5-flash')
GEMINI_EMBED_MODEL = os.getenv('GEMINI_EMBED_MODEL', 'embedding-001')
GEMINI_CHAT_URL = f'https://generativelanguage.googleapis.com/v1/models/{GEMINI_CHAT_MODEL}:generateContent'
_gemini_embed_model_path = GEMINI_EMBED_MODEL if GEMINI_EMBED_MODEL.startswith('models/') else f"models/{GEMINI_EMBED_MODEL}"
GEMINI_EMBED_URL = f'https://generativelanguage.googleapis.com/v1/{_gemini_embed_model_path}:embedContent'

# --- Telegram Notification Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', None)
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', None)
TELEGRAM_CHAT_IDS = [c.strip() for c in (TELEGRAM_CHAT_ID or '').split(',') if c.strip()]
NOTIFICATION_ENABLED = os.getenv('NOTIFICATION_ENABLED', 'false').lower() == 'true'

# --- Notification Test UI Access Control ---
def can_show_notification_test_ui():
    try:
        # Global on/off flag (default true for dev convenience)
        flag = os.getenv('NOTIFICATION_TEST_UI', 'true').lower() == 'true'
        if not flag:
            return False

        # Optional allow-lists
        allowed_env = os.getenv('NOTIFICATION_TEST_ALLOWED_EMAILS', '')
        manager_emails_env = os.getenv('MANAGER_EMAILS', '')
        manager_domain = (os.getenv('MANAGER_DOMAIN') or '').strip().lower()

        allowed_set = {e.strip().lower() for e in allowed_env.split(',') if e.strip()}
        manager_set = {e.strip().lower() for e in manager_emails_env.split(',') if e.strip()}

        # If no constraints provided, show by default
        if not allowed_set and not manager_set and not manager_domain:
            return True

        # Check current user email if available
        user = st.session_state.get('user') if 'user' in st.session_state else None
        user_email = None
        try:
            if user:
                user_email = getattr(user, 'email', None) or (user.user_metadata.get('email') if getattr(user, 'user_metadata', None) else None)
        except Exception:
            user_email = None

        if not user_email:
            return False

        ue = user_email.strip().lower()
        if ue in allowed_set or ue in manager_set:
            return True
        if manager_domain and ue.endswith('@' + manager_domain):
            return True

        return False
    except Exception:
        # If anything goes wrong, default to safe: hide
        return False

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type((
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        GeminiServiceUnavailableError,
        GeminiRateLimitError
    ))
)
def gemini_chat(messages):
    """Call Gemini chat API with OpenAI-style messages."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment.")
    
    # Convert OpenAI-style messages to Gemini format
    # Gemini doesn't support system messages, so we combine system + user into one user message
    prompt_parts = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'system':
            # Prepend system message with a clear label
            prompt_parts.append(f"[System Instructions]\n{content}\n[/System Instructions]")
        elif role == 'user':
            prompt_parts.append(content)
        elif role == 'assistant':
            # For assistant messages in context, we can include them but Gemini expects a different format
            # For now, just append as user context
            prompt_parts.append(f"[Previous Response]\n{content}\n[/Previous Response]")
    
    # Combine all parts into a single prompt
    prompt = "\n\n".join(prompt_parts)
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    
    try:
        response = requests.post(
            GEMINI_CHAT_URL, 
            params=params, 
            headers=headers, 
            data=json.dumps(payload),
            timeout=60
        )
        
        # Check status code and handle errors gracefully
        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                error_message = error_json.get('error', {}).get('message', error_text)
                error_code = error_json.get('error', {}).get('code', response.status_code)
                if response.status_code == 429:
                    raise GeminiRateLimitError(
                        f"Gemini rate limit/quota exceeded: {error_message}"
                    )
                if response.status_code == 503:
                    raise GeminiServiceUnavailableError(
                        f"Gemini service unavailable (high demand): {error_message}"
                    )
                if response.status_code == 401:
                    if "Incorrect API key provided" in error_message or "openai" in error_message.lower():
                        raise PermissionError(
                            "Gemini call failed with an OpenAI-style authentication error. "
                            "Please verify `GEMINI_API_KEY` is set correctly (and not an OpenAI key), "
                            "and ensure your provider/model settings are aligned."
                        )
                    raise PermissionError(f"Gemini authentication failed (401): {error_message}")
                raise requests.exceptions.HTTPError(
                    f"Gemini API Error {error_code}: {error_message}\n"
                    f"Status Code: {response.status_code}\n"
                    f"Full Response: {error_text[:500]}"
                )
            except (json.JSONDecodeError, KeyError):
                raise requests.exceptions.HTTPError(
                    f"Gemini API Error: Status {response.status_code}\n"
                    f"Response: {error_text[:500]}"
                )
        
        response_data = response.json()
        candidates = response_data.get('candidates', [])
        
        if not candidates:
            # Check if there's a blocking reason
            if 'promptFeedback' in response_data:
                feedback = response_data['promptFeedback']
                if feedback.get('blockReason'):
                    reason = feedback.get('blockReason')
                    raise ValueError(f"Gemini blocked the prompt. Reason: {reason}")
            return "[No response from Gemini - no candidates returned]"
        
        # Extract text from first candidate
        candidate = candidates[0]
        if 'content' in candidate and 'parts' in candidate['content']:
            text_parts = [part.get('text', '') for part in candidate['content']['parts'] if 'text' in part]
            if text_parts:
                return ''.join(text_parts)
        
        return "[No response from Gemini - unexpected response format]"
        
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout("Gemini API request timed out after 60 seconds")
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Error communicating with Gemini API: {str(e)}")

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type((
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        GeminiServiceUnavailableError,
        GeminiRateLimitError
    ))
)
def gemini_embed(text):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment.")
    try:
        payload = {
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        response = requests.post(
            GEMINI_EMBED_URL, 
            params=params, 
            headers=headers, 
            data=json.dumps(payload),
            timeout=30
        )
        if response.status_code != 200:
            if response.status_code == 429:
                try:
                    err_msg = response.json().get('error', {}).get('message', response.text)
                except Exception:
                    err_msg = response.text
                raise GeminiRateLimitError(f"Gemini embedding rate limit/quota exceeded: {err_msg}")
            if response.status_code == 503:
                try:
                    err_msg = response.json().get('error', {}).get('message', response.text)
                except Exception:
                    err_msg = response.text
                raise GeminiServiceUnavailableError(f"Gemini embedding unavailable (high demand): {err_msg}")
            print(f"Gemini embedding API error {response.status_code}: {response.text}")
            response.raise_for_status()
        response_data = response.json()
        print("DEBUG: Gemini embed API raw response:", response_data)  # <--- Debug print
        if 'embedding' in response_data:
            embedding = response_data['embedding'].get('values', None)
        elif 'data' in response_data:
            embedding = response_data['data'][0].get('embedding', None)
        else:
            print(f"Unexpected Gemini embedding response format: {response_data}")
            raise ValueError("No embedding found in response")
        if not embedding:
            raise ValueError("No embedding returned from Gemini API")
        return embedding
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error to Gemini embedding API: {str(e)}")
        raise
    except requests.exceptions.Timeout as e:
        print(f"Timeout error to Gemini embedding API: {str(e)}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Request error to Gemini embedding API: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error in gemini_embed: {str(e)}")
        raise

# --- Telegram Notification Functions ---
async def send_telegram_message(message: str):
    """Send a message via Telegram bot"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        print("Warning: Telegram bot token or chat ID not configured; skipping notification.")
        return False
    
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        any_success = False
        for chat_id in TELEGRAM_CHAT_IDS:
            try:
                await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                any_success = True
            except Exception as e:
                print(f"Warning: Telegram send failed for chat_id {chat_id}: {str(e)}")
                continue
        return any_success
    except Exception as e:
        print(f"Warning: Telegram async send failed: {str(e)}")
        return False

def send_telegram_message_sync(message: str):
    """Send a Telegram message to all configured chat IDs via HTTP (multi-recipient)."""
    if not NOTIFICATION_ENABLED:
        print("Warning: Notifications are disabled; skipping Telegram send.")
        return False
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        print("Warning: Telegram bot token or chat IDs not configured; skipping Telegram send.")
        return False
    try:
        api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        any_success = False
        for chat_id in TELEGRAM_CHAT_IDS:
            try:
                payload = {
                    "chat_id": chat_id,
                    "text": message,
                    "disable_web_page_preview": True
                }
                resp = requests.post(api_url, json=payload, timeout=10)
                if resp.status_code != 200:
                    print(f"Warning: Telegram API returned {resp.status_code} for {chat_id}: {resp.text[:200]}")
                any_success = any_success or (resp.status_code == 200)
            except Exception as e:
                print(f"Warning: Telegram exception sending to {chat_id}: {str(e)}")
                continue
        return any_success
    except Exception as e:
        print(f"Warning: Telegram sync send failed: {str(e)}")
        return False

def generate_daily_deal_summary():
    """Generate a daily summary of all deals for Telegram notification"""
    try:
        # Get all customers and their interactions
        customers = get_all_customer_data()
        
        if not customers:
            # No customers -> don't produce a notification payload
            print("generate_daily_deal_summary: no customers found; skipping summary generation")
            return None
        
        # Collect deal information from all customers
        all_deals = []
        total_customers = len(customers)
        
        for customer in customers:
            customer_name = customer.get('customer_name', 'Unknown')
            interactions = customer.get('interaction_metadata', [])
            
            if interactions:
                # Get the latest interaction
                latest_interaction = interactions[-1]
                latest_output = latest_interaction.get('output', '')
                
                # Extract deal information from the output
                if 'CURRENT DEALS:' in latest_output:
                    deals_section = latest_output.split('CURRENT DEALS:')[1].split('CLOSED DEALS:')[0] if 'CLOSED DEALS:' in latest_output else latest_output.split('CURRENT DEALS:')[1]
                    
                    # Parse deals from the table
                    lines = deals_section.strip().split('\n')
                    for line in lines:
                        if '|' in line and 'Deal_ID' not in line and '---' not in line:
                            parts = [part.strip() for part in line.split('|')]
                            if len(parts) >= 6:
                                deal_id = parts[1]
                                product = parts[2]
                                qty = parts[3]
                                price = parts[4]
                                stage = parts[5]
                                progress = parts[6] if len(parts) > 6 else 'N/A'
                                
                                all_deals.append({
                                    'customer': customer_name,
                                    'deal_id': deal_id,
                                    'product': product,
                                    'qty': qty,
                                    'price': price,
                                    'stage': stage,
                                    'progress': progress
                                })
        
        # Generate the summary message
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        message = f"*📊 LeanChems CRM Daily Deal Summary*\n"
        message += f"*Date:* {today}\n\n"
        
        if all_deals:
            message += f"*Active Deals:* {len(all_deals)}\n"
            message += f"*Total Customers:* {total_customers}\n\n"
            
            # Group deals by stage
            open_deals = [deal for deal in all_deals if deal['stage'] in ['Open', 'InProcess']]
            won_deals = [deal for deal in all_deals if deal['stage'] == 'Won']
            lost_deals = [deal for deal in all_deals if deal['stage'] == 'Lost']
            
            if open_deals:
                message += "*🟢 Open Deals:*\n"
                for deal in open_deals[:5]:  # Show top 5 open deals
                    message += f"• {deal['customer']} - {deal['product']} ({deal['qty']}) - {deal['stage']}\n"
                if len(open_deals) > 5:
                    message += f"• ... and {len(open_deals) - 5} more\n"
                message += "\n"
            
            if won_deals:
                message += "*✅ Won Deals:*\n"
                for deal in won_deals[:3]:  # Show top 3 won deals
                    message += f"• {deal['customer']} - {deal['product']} ({deal['qty']})\n"
                message += "\n"
            
            if lost_deals:
                message += "*❌ Lost Deals:*\n"
                for deal in lost_deals[:3]:  # Show top 3 lost deals
                    message += f"• {deal['customer']} - {deal['product']} ({deal['qty']})\n"
                message += "\n"
            
            # Add action items
            message += "*🎯 Today's Action Items:*\n"
            if open_deals:
                message += "• Follow up on open deals\n"
                message += "• Send quotes for pending requests\n"
                message += "• Schedule customer meetings\n"
            else:
                message += "• Focus on lead generation\n"
                message += "• Review customer profiles\n"
                message += "• Plan marketing activities\n"
        else:
            message += "*No active deals found.*\n\n"
            message += "*🎯 Today's Action Items:*\n"
            message += "• Focus on lead generation\n"
            message += "• Review customer profiles\n"
            message += "• Plan marketing activities\n"
        
        message += "\n*💡 Tip:* Check the CRM dashboard for detailed insights and analytics."
        
        return message
        
    except Exception as e:
        return f"Error generating daily summary: {str(e)}"

def send_daily_notification():
    """Send daily morning notification with deal summary"""
    if not NOTIFICATION_ENABLED:
        print("Notifications are disabled")
        return
    
    try:
        summary = generate_daily_deal_summary()

        # Skip sending when there's no content to share
        if not summary:
            print("send_daily_notification: no summary to send; skipping Telegram notification")
            return

        success = send_telegram_message_sync(summary)

        if success:
            print(f"Daily notification sent successfully at {datetime.datetime.now()}")
        else:
            print(f"Failed to send daily notification at {datetime.datetime.now()}")
            
    except Exception as e:
        print(f"Error in daily notification: {str(e)}")

def get_actor_display(user_id: str) -> str:
    """Return a human-friendly actor label using session user metadata when available."""
    try:
        user = st.session_state.get('user')
        if user and getattr(user, 'id', None) == user_id:
            full_name = (getattr(user, 'user_metadata', {}) or {}).get('full_name')
            if full_name:
                return f"{full_name} ({user_id})"
    except Exception:
        pass
    return user_id or "unknown"

def send_interaction_notification(customer_name: str, customer_id: str, actor: str, input_text: str, output_text: str, timestamp: datetime.datetime = None):
    """Send a formatted interaction notification with input and AI response."""
    print(f"DEBUG: send_interaction_notification called with customer_name={customer_name}, NOTIFICATION_ENABLED={NOTIFICATION_ENABLED}")
    if not NOTIFICATION_ENABLED:
        print("DEBUG: Notifications disabled")
        return False
    ts = (timestamp or datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
    # Use only the actor's display name (without any ID in parentheses)
    actor_name_only = actor.split(' (')[0] if actor else 'Unknown'
    # Trim large bodies to keep Telegram happy
    def trim(text: str, limit: int = 2000) -> str:
        if not text:
            return ""
        text = text.strip()
        return text if len(text) <= limit else text[: limit - 3] + "..."
    input_block = trim(input_text, 800)
    output_block = trim(output_text, 1200)
    message = (
        f"💬 New Customer Interaction!\n\n"
        f"👤 Customer: {customer_name}\n"
        f"🆔 Customer ID: {customer_id}\n"
        f"📅 Timestamp: {ts}\n"
        f"👨‍💼 Updated by: {actor_name_only}\n\n"
        f"📝 Input:\n{input_block}\n\n"
        f"🤖 AI Response:\n{output_block}"
    )
    print(f"DEBUG: Sending message: {message[:200]}...")
    result = send_telegram_message_sync(message)
    print(f"DEBUG: send_telegram_message_sync returned: {result}")
    return result

def send_deal_update_notification(customer_name: str, deal_info: str):
    """Send immediate notification when a deal is updated"""
    if not NOTIFICATION_ENABLED:
        return False
    
    try:
        message = f"*🔄 Deal Update Alert*\n\n"
        message += f"*Customer:* {customer_name}\n"
        message += f"*Time:* {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        message += f"*Update Details:*\n{deal_info}\n\n"
        message += f"*💡 Action Required:* Check the CRM dashboard for full details."
        
        success = send_telegram_message_sync(message)
        if success:
            print(f"Deal update notification sent for {customer_name}")
        else:
            print(f"Failed to send deal update notification for {customer_name}")
        return success
    except Exception as e:
        print(f"Error sending deal update notification: {str(e)}")
        return False

def send_new_customer_notification(customer_name: str, customer_id: str, actor: str = None, created_at: datetime.datetime = None, profile_summary: str = None):
    """Send notification when a new customer is created"""
    if not NOTIFICATION_ENABLED:
        return False
    
    try:
        # Build message in requested format
        created_str = (created_at or datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
        summary_preview = (profile_summary or "").strip()
        if len(summary_preview) > 400:
            summary_preview = summary_preview[:397] + "..."
        # Use only the actor's display name (without any ID in parentheses)
        actor_name_only = actor.split(' (')[0] if actor else 'Unknown'
        message = (
            f"🚀 New Customer Created!\n\n"
            f"👤 Customer: {customer_name}\n"
            f"🆔 Display ID: {customer_id}\n"
            f"📅 Created: {created_str}\n"
            f"👨‍💼 Created by: {actor_name_only}\n\n"
        )
        if summary_preview:
            message += (
                f"📋 Profile Summary:\n"
                f"{summary_preview}"
            )
        
        success = send_telegram_message_sync(message)
        if success:
            print(f"New customer notification sent for {customer_name}")
        else:
            print(f"Failed to send new customer notification for {customer_name}")
        return success
    except Exception as e:
        print(f"Error sending new customer notification: {str(e)}")
        return False

def start_notification_scheduler():
    """Start the notification scheduler in a separate thread"""
    # Prevent spawning multiple scheduler threads on Streamlit reruns
    # Use both a module-level flag and session_state to reduce duplicates
    global _notif_scheduler_started_global, _notif_scheduler_thread

    if not NOTIFICATION_ENABLED:
        print("Notifications are disabled - scheduler not started")
        return

    # Acquire lock so only one caller can start the scheduler
    with _notif_scheduler_lock:
        if _notif_scheduler_started_global:
            print("Notification scheduler already started (global)")
            return

        # Clear previously scheduled daily jobs to avoid duplicates
        try:
            schedule.clear('daily_notification')
        except Exception:
            schedule.clear()

        def run_scheduler():
            # Schedule daily notification at 08:00 local time and tag it
            try:
                schedule.every().day.at("08:00").do(send_daily_notification).tag('daily_notification')
            except Exception as e:
                print(f"Failed to schedule daily notification: {e}")

            while True:
                try:
                    schedule.run_pending()
                except Exception as e:
                    print(f"Scheduler run_pending error: {e}")
                time.sleep(60)

        _notif_scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        _notif_scheduler_thread.start()
        _notif_scheduler_started_global = True

        try:
            st.session_state["_notif_scheduler_started"] = True
        except Exception:
            pass

        print("Notification scheduler started")

# Cache OpenAI client and Memory instance
@st.cache_resource
def get_openai_client():
    return OpenAI()

@st.cache_resource
def get_memory():
    class NoopMemory:
        def search(self, query: str = "", user_id: str = "", limit: int = 3):
            return {"results": []}

        def add(self, *args, **kwargs):
            return None

    # Validate connection string first
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        st.warning("DATABASE_URL not set. Memory features are disabled.")
        return NoopMemory()

    # mem0 provider support can vary by version; when running Gemini mode,
    # disable memory backend to avoid OpenAI-auth related background errors.
    if LLM_PROVIDER == 'gemini':
        return NoopMemory()

    llm_model = GEMINI_CHAT_MODEL if LLM_PROVIDER == 'gemini' else model

    config = {
        "llm": {
            "provider": LLM_PROVIDER,
            "config": {
                "model": llm_model
            }
        },
        "vector_store": {
            "provider": "supabase",
            "config": {
                "connection_string": db_url,
                "collection_name": "memories"
            }
        }
    }

    try:
        return Memory.from_config(config)
    except Exception as e:
        st.error(f"Memory backend init failed; running without vector memory. Error: {str(e)}")
        return NoopMemory()

# Get cached resources
openai_client = get_openai_client()
memory = get_memory()

def generate_customer_id():
    """Generate a unique customer ID"""
    # Generate a UUID for the database
    return str(uuid.uuid4())

def generate_display_id():
    """Generate a human-readable customer ID in format LC-YYYY-CUST-XXXX"""
    year = datetime.datetime.now().year
    
    # Get all customer IDs to find the highest number
    response = supabase_client.table('customers').select('display_id').execute()
    max_num = 0
    
    if response.data:
        for customer in response.data:
            display_id = customer.get('display_id', '')
            # Check if the ID matches our format (LC-YYYY-CUST-XXXX)
            if isinstance(display_id, str) and display_id.startswith(f'LC-{year}-CUST-'):
                try:
                    num = int(display_id.split('-')[-1])
                    max_num = max(max_num, num)
                except ValueError:
                    continue
    
    # Increment the highest number found
    new_num = max_num + 1
    return f"LC-{year}-CUST-{new_num:04d}"

def find_similar_customers(customer_name: str, threshold: int = 80):
    """Find similar customer names using fuzzy matching"""
    response = supabase_client.table('customers').select('customer_name,customer_id').execute()
    similar_customers = []
    
    for customer in response.data:
        # Calculate similarity score
        ratio = fuzz.ratio(customer_name.lower(), customer['customer_name'].lower())
        if ratio >= threshold:
            similar_customers.append({
                'name': customer['customer_name'],
                'id': customer['customer_id'],
                'similarity': ratio
            })
    
    return sorted(similar_customers, key=lambda x: x['similarity'], reverse=True)

def search_web_for_company(company_name: str):
    """Search the web for company information using both Google PSE, SerpAPI, and force-include Wikipedia and official site."""
    try:
        combined_results = []
        # 1. Google PSE Search
        pse_api_key = os.getenv("GOOGLE_PSE_API_KEY")
        pse_cx = os.getenv("GOOGLE_PSE_CX")
        if pse_api_key and pse_cx:
            query = f"{company_name} company information business profile"
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.googleapis.com/customsearch/v1?key={pse_api_key}&cx={pse_cx}&q={encoded_query}&num=5"
            response = requests.get(url)
            if response.status_code == 200:
                results = response.json()
                if "items" in results:
                    for item in results["items"]:
                        result = {
                            'title': item.get('title', ''),
                            'snippet': item.get('snippet', ''),
                            'link': item.get('link', ''),
                            'source': 'Google PSE'
                        }
                        if 'pagemap' in item and 'metatags' in item['pagemap']:
                            metatags = item['pagemap']['metatags'][0]
                            if 'og:description' in metatags:
                                result['description'] = metatags['og:description']
                        combined_results.append(result)
        # 2. SerpAPI Search
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if serpapi_key:
            params = {
                "engine": "google",
                "q": f"{company_name} company information business profile",
                "api_key": serpapi_key,
                "num": 5
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            if "organic_results" in results:
                for result in results["organic_results"]:
                    combined_results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'source': 'SerpAPI'
                    })
        # 3. Force-include Wikipedia page
        wiki_url = f"https://en.wikipedia.org/wiki/{company_name.replace(' ', '_')}"
        wiki_resp = requests.get(wiki_url)
        if wiki_resp.status_code == 200:
            soup = BeautifulSoup(wiki_resp.text, 'html.parser')
            p = soup.find('p')
            snippet = p.text.strip() if p else ''
            combined_results.append({
                'title': f"Wikipedia: {company_name}",
                'snippet': snippet,
                'link': wiki_url,
                'source': 'Wikipedia'
            })
        # 4. Force-include official site if pattern matches
        for domain in [f"https://{company_name.replace(' ', '').lower()}.com", f"https://{company_name.replace(' ', '').capitalize()}.com"]:
            try:
                resp = requests.get(domain, timeout=3)
                if resp.status_code == 200:
                    combined_results.append({
                        'title': f"Official Site: {company_name}",
                        'snippet': f"Official website for {company_name}.",
                        'link': domain,
                        'source': 'Official Site'
                    })
                    break
            except Exception:
                continue
        # Remove duplicates based on URL
        unique_results = []
        seen_urls = set()
        for result in combined_results:
            if result['link'] not in seen_urls:
                seen_urls.add(result['link'])
                unique_results.append(result)
        # Format results
        web_context = ""
        for result in unique_results:
            web_context += f"\nTitle: {result['title']}\n"
            web_context += f"Snippet: {result['snippet']}\n"
            web_context += f"Link: {result['link']}\n"
            web_context += f"Source: {result['source']}\n"
            if 'description' in result:
                web_context += f"Description: {result['description']}\n"
            web_context += "---\n"
        return web_context
    except Exception as e:
        print(f"Web search failed for '{company_name}': {str(e)}")
        return ""

def search_linkedin_profiles_ethiopia(company_name: str):
    """Search for LinkedIn profiles in Ethiopia using both Google PSE and SerpAPI"""
    try:
        all_profiles = []
        
        # Broader search query as a fallback
        search_queries = [
            f'site:linkedin.com/in/ "{company_name}" Ethiopia (CEO OR "Managing Director" OR "General Manager")',
            f'site:linkedin.com/in/ "{company_name}" Ethiopia (Operations OR "Plant Manager" OR Production)',
            f'site:linkedin.com/in/ "{company_name}" Ethiopia (Procurement OR "Supply Chain" OR Purchasing)',
            f'site:linkedin.com/in/ "{company_name}" Ethiopia (Technical OR R&D OR Quality)',
            f'site:linkedin.com/in/ "{company_name}" Ethiopia (Sales OR "Business Development")',
            f'site:linkedin.com/in/ "{company_name}" Ethiopia' # General search
        ]
        
        # Check for API keys and show warnings
        pse_api_key = os.getenv("GOOGLE_PSE_API_KEY")
        pse_cx = os.getenv("GOOGLE_PSE_CX")
        serpapi_key = os.getenv("SERPAPI_API_KEY")

        if not pse_api_key or not pse_cx:
            print("Google Custom Search API keys are not set. LinkedIn search may be incomplete.")
        
        if not serpapi_key:
            print("SerpAPI key is not set. LinkedIn search may be incomplete.")

        # 1. Google PSE Search
        if pse_api_key and pse_cx:
            for query in search_queries:
                encoded_query = urllib.parse.quote(query)
                url = f"https://www.googleapis.com/customsearch/v1?key={pse_api_key}&cx={pse_cx}&q={encoded_query}&num=5&gl=et&hl=en"
                
                response = requests.get(url)
                if response.status_code == 200:
                    results = response.json()
                    if "items" in results:
                        for item in results["items"]:
                            title = item.get('title', '')
                            snippet = item.get('snippet', '')
                            link = item.get('link', '')
                            
                            name = title.split('|')[0].strip()
                            position = snippet.split('·')[0].strip() if '·' in snippet else 'Not specified'
                            
                            all_profiles.append({
                                'name': name,
                                'position': position,
                                'link': link,
                                'snippet': snippet,
                                'source': 'Google PSE'
                            })
        
        # 2. SerpAPI Search
        if serpapi_key:
            for query in search_queries:
                params = {
                    "engine": "google",
                    "q": query,
                    "api_key": serpapi_key,
                    "num": 5,
                    "gl": "et",
                    "hl": "en",
                    "filter": 0
                }
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                if "organic_results" in results:
                    for result in results["organic_results"]:
                        title = result.get('title', '')
                        snippet = result.get('snippet', '')
                        link = result.get('link', '')
                        
                        name = title.split('|')[0].strip()
                        position = snippet.split('·')[0].strip() if '·' in snippet else 'Not specified'
                        
                        all_profiles.append({
                            'name': name,
                            'position': position,
                            'link': link,
                            'snippet': snippet,
                            'source': 'SerpAPI'
                        })
        
        # Remove duplicates based on LinkedIn URL and format results
        unique_profiles = []
        seen_links = set()
        for profile in all_profiles:
            if profile['link'] and profile['link'] not in seen_links:
                seen_links.add(profile['link'])
                unique_profiles.append(profile)
        
        linkedin_context = "\nLinkedIn Profiles in Ethiopia:\n"
        if unique_profiles:
            for profile in unique_profiles[:10]:  # Limit to top 10 profiles
                linkedin_context += f"\n- Name: {profile['name']}\n"
                linkedin_context += f"  Position: {profile['position']}\n"
                linkedin_context += f"  Profile: {profile['link']}\n"
                linkedin_context += f"  Source: {profile['source']}\n"
                if profile['snippet']:
                    context = profile['snippet'].split('·')[-1].strip()
                    if context:
                        linkedin_context += f"  Context: {context}\n"
                linkedin_context += "---\n"
        else:
            linkedin_context += "\nNo relevant LinkedIn profiles found. This could be due to missing API keys, search limitations, or no public profiles for this company.\n"
            
        return linkedin_context
    except Exception as e:
        print(f"LinkedIn search failed for '{company_name}': {str(e)}")
        return ""

def generate_customer_profile(customer_name: str, user_id: str):
    """Generate a customer profile using AI, existing conversations, and web search"""
    # Search relevant documents and memories
    relevant_docs = search_documents(customer_name, user_id)
    try:
        relevant_memories = get_cached_memories(customer_name, user_id)
    except Exception as e:
        print(f"Memory retrieval failed during profile generation: {str(e)}")
        relevant_memories = {"results": []}
    
    # Search web for company information
    web_context = search_web_for_company(customer_name)
    
    # Search for LinkedIn profiles in Ethiopia
    linkedin_context = search_linkedin_profiles_ethiopia(customer_name)
    
    # Combine all context
    context = ""
    if relevant_docs:
        context += "\nRelevant conversations:\n"
        for doc in relevant_docs:
            context += f"\n{doc.get('content', '')}\n"
    
    if relevant_memories["results"]:
        context += "\nRelevant memories:\n"
        for memory in relevant_memories["results"]:
            context += f"\n{memory['memory']}\n"
    
    if web_context:
        context += "\nWeb Search Results:\n"
        context += web_context
    
    if linkedin_context:
        context += "\nLinkedIn Information:\n"
        context += linkedin_context
    
    # Create the enhanced prompt for profile generation
    system_prompt = """You are an Industry-Intel Research Assistant and B2B Chemical-Supply Strategist for LeanChem. Your mission is to perform a deep-dive analysis of {Target Company} and all of its construction-relevant subsidiaries operating in Ethiopia, to:

Identify all business units manufacturing products for the construction sector: cement, dry-mix mortar, concrete admixtures, and paint/coatings.

Evaluate how LeanChem's chemical portfolio aligns with each unit's product and operational profile.

Recommend precise engagement strategies tailored by subsector and supply pain points.

Provide verified decision-maker contacts for B2B outreach.

If the company is a conglomerate, list all major business units and subsidiaries relevant to construction, chemicals, and manufacturing, even if not all are found in the immediate context.

🧾 Primary Deliverables
Company Overview & Recent News

≤500-character summary of {Target Company}'s core business, size, and activity in Ethiopia.

Highlight recent expansions, investments, or new product lines in cement, dry-mix, admixtures, or coatings, using GPT-4o/web-search insights or official sources.

Include citations [1], [2], … from reliable sources.

Construction-Sector Manufacturing Overview

A structured table listing all business units in Ethiopia manufacturing construction-related materials.

Table Columns:

Business Unit

Construction Products

Location (City, Country)

Scale Metric (e.g., capacity, project size)

Source

Strategic-Fit Matrix

For each subsidiary, assess alignment to LeanChem's offerings across 4 subsectors:

Cement

Dry-Mix

Admixtures

Paint & Coatings

Score each axis using:

0 = No Fit

1 = Low Fit

2 = Moderate Fit

3 = High Fit

Base scores on:

Volume opportunity vs LeanChem capacity

LeanChem's ability to solve supply or technical pain points (e.g., forex, lead time, performance)

Competitive pressure and likelihood of switching

Strategic Insights & Action Plan

Max 200-word narrative outlining 3–5 high-leverage opportunities and pain-point matches.

Segment by subsector (Cement, Dry-Mix, Admixtures, Paints) and recommend clear engagement actions such as:

Outreach channel (email, event, enabler)

Sample trial with product match

Proposal for supply contract, JIT, or SEZ warehousing

Technical advisory to improve performance or reduce cost

Key Contacts for Engagement

List up to 10 decision-makers or influencers in operations, procurement, or technical roles.

Columns:

Name

Position

LinkedIn Profile (full clickable URL)

Source

Extract only real individuals verified via LinkedIn or company websites.

🧩 Research Inputs
LeanChem Offerings

Dry-Mix/Plaster: RDP, HPMC, Starch Ether, Fiber, Zinc Stearate, Plasticizer, Defoamer, SBR, Acrylic Waterproofing, White Cement, Iron Oxide, Titanium Dioxide

Concrete Admixtures: PCE, SNF, Lignosulphonate, Sodium Gluconate, Penetrol-type waterproofing

Paint/Coatings: Styrene-Acrylic Binders, Pure Acrylics, VAE, HEC, White Cement, Iron Oxide, Titanium Dioxide

Cement Grinding: Cement grinding aids

🔍 Research Tools & Constraints
Source from:

{Target Company} official website and group/subsidiary pages

Annual reports and press releases

LinkedIn (for verified role-based contacts)

News outlets, trade journals, government registries

Use structured search queries like:

"{Target Company} cement plant Ethiopia"

"{Target Company} paint coatings manufacturer Ethiopia"

"{Target Company} dry mix mortar factory site"

"{Target Company} procurement manager LinkedIn"

Use numbered citations [1], [2], etc.

Provide honest results—if a construction vertical is not present, list as "N/A" or "0" in the fit matrix.
..."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate a profile for: {customer_name}\n\nContext:{context}"}
    ]
    
    # Get response and convert to string
    try:
        response = gemini_chat(messages)
    except RetryError as e:
        raise
    except PermissionError:
        raise
    except Exception as e:
        raise RuntimeError(f"Profile generation failed: {str(e)}") from e
    profile_text = response
    return profile_text

def ensure_vector(embedding):
    # If it's a string, try to parse as JSON
    if isinstance(embedding, str):
        try:
            embedding = json.loads(embedding)
        except Exception:
            raise ValueError("Embedding string could not be parsed as JSON array.")
    # If it's a list of numbers (possibly as strings), convert all to float
    if isinstance(embedding, list):
        # If it's a list of lists (shouldn't happen for a single embedding), flatten
        if len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = embedding[0]
        return [float(x) for x in embedding]
    # If it's a single float or int, raise error
    if isinstance(embedding, (float, int)):
        raise ValueError("Embedding is a single number, expected a list of floats.")
    raise ValueError(f"Unexpected embedding type: {type(embedding)}")

def create_new_customer(customer_name: str, user_id: str):
    """Handle the complete customer creation workflow"""
    # Ensure we have a valid state
    if 'customer_creation_state' not in st.session_state or st.session_state.customer_creation_state is None:
        st.session_state.customer_creation_state = {
            'step': 1,
            'customer_name': customer_name,
            'profile': None,
            'profile_generation_error': None,
            'profile_generated': False,
            'confirmed': False
        }

    state = st.session_state.customer_creation_state
    
    # Step 1: Check for similar customers
    if state['step'] == 1:
        similar_customers = find_similar_customers(customer_name)
        
        if similar_customers:
            st.warning(f"Similar customer found: {similar_customers[0]['name']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create New Customer Anyway"):
                    state['step'] = 2
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.customer_creation_state = None
                    st.rerun()
            return None
        else:
            state['step'] = 2
            st.rerun()
    
    # Step 2: Generate profile, preview, then choose save/discard
    if state['step'] == 2:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate AI Profile", key=f"generate_profile_{customer_name}", type="primary"):
                with st.spinner("API rate limit reached. Waiting for Google to allow the next request... this may take up to a minute."):
                    try:
                        state['profile'] = generate_customer_profile(customer_name, user_id)
                        state['profile_generation_error'] = None
                        state['profile_generated'] = True
                    except RetryError as e:
                        root_error = e.last_attempt.exception() if e.last_attempt else e
                        print(f"Profile generation exhausted retries: {str(root_error)}")
                        state['profile'] = None
                        state['profile_generation_error'] = None
                        state['profile_generated'] = False
                        st.warning("Please wait 1 minute and try again.")
                    except Exception as e:
                        state['profile'] = None
                        state['profile_generation_error'] = str(e)
                        state['profile_generated'] = False
        with col2:
            if st.button("Cancel", key=f"cancel_create_{customer_name}"):
                st.session_state.customer_creation_state = None
                st.rerun()

        if state.get('profile_generation_error'):
            st.error(f"Profile generation failed: {state['profile_generation_error']}")
            return None

        if state.get('profile_generated') and state.get('profile'):
            st.write("Generated Profile Preview:")
            st.write(state['profile'])
            save_col, discard_col = st.columns(2)
            with save_col:
                if st.button("Save Customer", key=f"save_customer_{customer_name}", type="primary"):
                    state['confirmed'] = True
                    state['step'] = 3
                    st.rerun()
            with discard_col:
                if st.button("Discard Profile", key=f"discard_profile_{customer_name}"):
                    st.session_state.customer_creation_state = None
                    st.rerun()
        return None

    # Step 3: Create database entry (silent background failures except main save failure)
    if state['step'] == 3 and state.get('confirmed'):
        with st.spinner("API rate limit reached. Waiting for Google to allow the next request... this may take up to a minute."):
            if not state.get('profile'):
                st.error("No generated profile found. Please generate the profile before saving.")
                state['step'] = 2
                state['confirmed'] = False
                return None

            customer_id = generate_customer_id()
            display_id = generate_display_id()
            profile_input = f"Create profile for {customer_name}"
            profile_output = str(state.get('profile') or f"Customer {customer_name} created without AI profile.")
            # --- Generate embedding for the profile input ---
            interaction_embeddings = []
            try:
                embedding = gemini_embed(profile_input)
                embedding = ensure_vector(embedding)
                interaction_embeddings = [embedding]  # Always a list of lists
            except Exception as e:
                print(f"Embedding generation failed during customer save: {str(e)}")
            interaction_json = {
                "input": profile_input,
                "output": profile_output,
                "timestamp": datetime.datetime.now().isoformat(),
                "user_id": user_id
            }
            data = {
                "customer_id": customer_id,
                "display_id": display_id,
                "customer_name": customer_name,
                "input_conversation": [profile_input],
                "output_conversation": [profile_output],
                "interaction_metadata": [interaction_json],  # list of dicts (JSON)
                "interaction_embeddings": interaction_embeddings  # list of lists of floats (vector per interaction)
            }
            try:
                response = supabase_client.table('customers').insert(data).execute()
                if response.data:
                    # Clear the creation state first
                    st.session_state.customer_creation_state = None

                    # Notification failure should not block save.
                    try:
                        actor = get_actor_display(user_id)
                        created_at = datetime.datetime.now()
                        profile_summary = profile_output
                        notification_ok = send_new_customer_notification(
                            customer_name, display_id, actor, created_at, profile_summary
                        )
                        if not notification_ok:
                            print(
                                f"Warning: New customer notification not delivered for "
                                f"{customer_name} ({display_id})."
                            )
                    except Exception as notify_err:
                        print(f"Customer notification failed: {str(notify_err)}")
                    return response.data[0]
                else:
                    st.error("Failed to create customer")
                    return None
            except Exception as e:
                st.error(f"Error creating customer: {str(e)}")
                return None

# Authentication functions
def sign_up(email, password, full_name):
    try:
        response = supabase_client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name
                }
            }
        })
        if response and response.user:
            st.success("Sign up successful! Please check your email and confirm your address before logging in.")
            # Do NOT log the user in automatically after signup
            # Do NOT set st.session_state.authenticated = True or st.session_state.user = response.user here
        return response
    except Exception as e:
        st.error(f"Error signing up: {str(e)}")
        return None

def sign_in(email, password):
    try:
        response = supabase_client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if response and response.user:
            # Store user info directly in session state
            st.session_state.authenticated = True
            st.session_state.user = response.user
            st.rerun()
        return response
    except Exception as e:
        st.error(f"Error signing in: {str(e)}")
        return None

def sign_out():
    try:
        supabase_client.auth.sign_out()
        # Clear only authentication-related session state
        st.session_state.authenticated = False
        st.session_state.user = None
        # Set a flag to trigger rerun on next render
        st.session_state.logout_requested = True
    except Exception as e:
        st.error(f"Error signing out: {str(e)}")

# Add caching for frequently accessed data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_memories(query, user_id):
    try:
        return memory.search(query=query, user_id=user_id, limit=2)
    except Exception as e:
        # Keep customer/profile flow usable even if memory backend is unavailable.
        print(f"Memory retrieval disabled or failed: {str(e)}")
        return {"results": []}

# Add retry decorator for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_llm_response(messages, model):
    try:
        if LLM_PROVIDER == 'openai':
            return openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )
        elif LLM_PROVIDER == 'gemini':
            # Gemini does not support streaming, so yield a single chunk
            class DummyChunk:
                def __init__(self, text):
                    self.choices = [type('Delta', (), {'delta': type('DeltaContent', (), {'content': text})()})()]
            text = gemini_chat(messages)
            yield DummyChunk(text)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        raise

def search_documents(query: str, user_id: str, limit: int = 3):
    try:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def get_embedding():
            return gemini_embed(query)
        query_embedding = get_embedding()
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def search_supabase():
            try:
                response = supabase_client.rpc(
                    'match_conversation',
                    {
                        'query_embedding': query_embedding,
                        'match_count': limit,
                        'match_threshold': 0.5,
                        'filter': {}
                    }
                ).execute()
                return response
            except Exception as rpc_error:
                print(f"Supabase RPC error details: {str(rpc_error)}")
                # Remove .message and .details accesses
                # if hasattr(rpc_error, 'message'):
                #     st.error(f"RPC Error message: {rpc_error.message}")
                # if hasattr(rpc_error, 'details'):
                #     st.error(f"RPC Error details: {rpc_error.details}")
                raise
        response = search_supabase()
        if response.data:
            return response.data
        else:
            return []
    except Exception as e:
        print(f"Document search failed: {str(e)}")
        return []

# --- Customer management functions ---
def store_customer_conversation(customer_name: str, user_input: str, ai_output: str):
    response = supabase_client.table('customers').insert({
        'customer_name': customer_name,
        'input_conversation': [user_input],
        'output_conversation': [ai_output]
    }).execute()
    return response.data

def fetch_customer(customer_name: str):
    response = supabase_client.table('customers').select("*").eq('customer_name', customer_name).execute()
    if response.data:
        return response.data[0]
    return None

def update_customer_memory(customer_id: str, new_input: str, new_output: str):
    customer = supabase_client.table('customers').select("*").eq('customer_id', customer_id).single().execute()
    if customer.data:
        updated_inputs = customer.data['input_conversation'] + [new_input]
        updated_outputs = customer.data['output_conversation'] + [new_output]
        response = supabase_client.table('customers').update({
            'input_conversation': updated_inputs,
            'output_conversation': updated_outputs
        }).eq('customer_id', customer_id).execute()
        return response.data
    return None

def handle_create_customer_flow(customer_name: str, user_input: str, ai_output: str):
    customer = fetch_customer(customer_name)
    if customer:
        update_customer_memory(customer['customer_id'], user_input, ai_output)
        st.success(f"Customer {customer_name} found. Conversation memory updated.")
    else:
        store_customer_conversation(customer_name, user_input, ai_output)
        st.success(f"Customer {customer_name} not found. Created new customer.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_all_customer_names():
    response = supabase_client.table('customers').select('customer_name,customer_id').execute()
    if response.data:
        return {c['customer_name']: c['customer_id'] for c in response.data}
    return {}

# Add a function to load Lottie animation JSON from a URL.
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def detect_latest_interaction_query(message):
    pattern = re.compile(r'(latest|last) interaction with ([\w\s]+)', re.IGNORECASE)
    match = pattern.search(message)
    if match:
        customer_name = match.group(2).strip()
        return customer_name
    return None

def get_latest_interaction_by_name(customer_name: str):
    customer = supabase_client.table('customers').select('interaction_metadata').eq('customer_name', customer_name).single().execute()
    if customer.data and customer.data.get('interaction_metadata'):
        interactions = customer.data['interaction_metadata']
        if interactions:
            return interactions[-1]  # Last (latest) interaction
    return None

def detect_summarize_query(message):
    pattern = re.compile(r'summarize (my )?interaction(s)? with ([\w\s]+)', re.IGNORECASE)
    match = pattern.search(message)
    if match:
        customer_name = match.group(3).strip()
        return customer_name
    return None

def summarize_interactions_with_customer(customer_name, user_id, n=5):
    customer = supabase_client.table('customers').select('interaction_metadata').eq('customer_name', customer_name).single().execute()
    if customer.data and customer.data.get('interaction_metadata'):
        interactions = customer.data['interaction_metadata'][-n:]  # Last n interactions
        if not interactions:
            return f"No interactions found for {customer_name}."
        context = ""
        for i, interaction in enumerate(interactions, 1):
            context += f"Interaction {i}:\nInput: {interaction['input']}\nOutput: {interaction['output']}\n"
        system_prompt = f"Summarize the following interactions with {customer_name}:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        return gemini_chat(messages)
    return f"No interactions found for {customer_name}."

def chat_with_memories(message, user_id):
    # Meta-query detection for latest interaction
    customer_name_meta = detect_latest_interaction_query(message)
    if customer_name_meta:
        latest = get_latest_interaction_by_name(customer_name_meta)
        if not latest:
            return f"No interactions found for {customer_name_meta}."
        formatted = (
            f"Latest interaction with {customer_name_meta}:\n"
            f"Input: {latest['input']}\n"
            f"Output: {latest['output']}\n"
            f"Timestamp: {latest['timestamp']}\n"
        )
        return formatted
    # Meta-query detection for summarize
    customer_name_summarize = detect_summarize_query(message)
    if customer_name_summarize:
        return summarize_interactions_with_customer(customer_name_summarize, user_id)
    try:
        # 1. Detect if a customer is mentioned in the message
        customer_dict = get_all_customer_names()
        mentioned_customer = None
        for name in customer_dict:
            if name.lower() in message.lower():
                mentioned_customer = name
                break

        # 2. Fetch customer conversations if mentioned (RAG retrieval)
        customer_context = ""
        if mentioned_customer:
            customer_id = customer_dict[mentioned_customer]
            relevant_interactions = retrieve_relevant_interactions(customer_id, message, top_k=3)
            if relevant_interactions:
                customer_context += f"\nCustomer: {mentioned_customer}\n"
                for interaction in relevant_interactions:
                    customer_context += f"User: {interaction['input']}\nAI: {interaction['output']}\n(Similarity: {interaction['similarity']:.2f})\n"

        # 3. Fetch relevant memories (filter out empty/irrelevant)
        relevant_memories = get_cached_memories(message, user_id)
        memories_str = "\n".join(
            f"- {entry['memory']}" for entry in relevant_memories["results"]
            if entry['memory'] and entry['memory'].strip() and entry['memory'].strip().lower() != "not specified"
        )
        # 4. Search relevant documents
        relevant_docs = search_documents(message, user_id)
        docs_str = ""
        if relevant_docs:
            docs_str = "\nRelevant Conversations from Database:\n"
            for i, doc in enumerate(relevant_docs, 1):
                docs_str += f"\nConversation {i}:\n{doc.get('content', '')}\n"
        # 5. Build the system prompt/context
        system_prompt = f"""
You are a helpful AI assistant specialized in chemical trading and CRM.
If the user asks about a specific customer, use the customer's most relevant past interactions below (retrieved by semantic similarity).
Also use the provided memories and relevant conversations from the database.
If you don't find relevant information, say so.

Customer context:
{customer_context}

        User Memories:
        {memories_str}
        
{docs_str}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        with st.spinner("Thinking..."):
            # For Gemini, just get the full response at once
            full_response = gemini_chat(messages)
            response_placeholder = st.empty()
            response_placeholder.markdown(full_response)
            # Show what conversations were used
            if mentioned_customer and relevant_interactions:
                with st.expander("Relevant Past Interactions Used for this Response"):
                    for i, interaction in enumerate(relevant_interactions, 1):
                        st.write(f"Interaction {i} (Similarity: {interaction['similarity']:.2f}):")
                        st.write(f"User: {interaction['input']}")
                        st.write(f"AI: {interaction['output']}")
        # Create new memories from the conversation
        messages.append({"role": "assistant", "content": full_response})
        memory.add(messages, user_id=user_id)
        # --- New: Automatically store conversation if customer is mentioned ---
        try:
            if mentioned_customer:
                update_customer_interaction(customer_dict[mentioned_customer], message, full_response, user_id)
                st.info(f"Conversation added to {mentioned_customer}'s record.")
        except Exception as e:
            st.warning(f"Tried to auto-store conversation for mentioned customer, but got error: {str(e)}")
        return full_response
    except Exception as e:
        st.error(f"An error occurred during chat: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

def search_customers(query: str, limit: int = 5):
    try:
        # Search for customers by name
        response = supabase_client.table('customers').select('*').ilike('customer_name', f'%{query}%').limit(limit).execute()
        
        if response.data:
            return response.data
        return []
    except Exception as e:
        st.error(f"Error searching customers: {str(e)}")
        return []

def get_customer_conversations(customer_name: str):
    try:
        # Get customer conversations
        response = supabase_client.table('customers').select('*').eq('customer_name', customer_name).single().execute()
        
        if response.data:
            return response.data
        return None
    except Exception as e:
        st.error(f"Error getting customer conversations: {str(e)}")
        return None

# Update the sidebar section to handle the customer creation state
def render_customer_creation_ui_tab(user_id):
    st.subheader("Create New Customer")

    # Initialize the input field state if it doesn't exist
    if 'tab_new_customer_name' not in st.session_state:
        st.session_state.tab_new_customer_name = ""

    # Text input for customer name
    new_customer_name = st.text_input("Enter Customer Name", key="tab_new_customer_name")

    # Add the Enter button always below the input field
    if st.button("Enter", key="create_customer_enter_button", type="primary"):
        if not new_customer_name:
            st.warning("Please enter a customer name.")
        else:
            # Initialize the creation state
            st.session_state.customer_creation_state = {
                'step': 1,
                'customer_name': new_customer_name,
                'profile': None,
                'profile_generation_error': None,
                'profile_generated': False,
                'confirmed': False
            }
            st.rerun()

    # --- Customer creation flow logic based on state ---
    if st.session_state.get('customer_creation_state') is not None:
        # Call create_new_customer to handle the current step (1, 2, or 3)
        created_customer_data = create_new_customer(
            st.session_state.customer_creation_state['customer_name'],
            user_id
        )

        # After create_new_customer finishes its steps, check if creation was successful (step 3 completed)
        if created_customer_data:
            # Customer creation finished successfully in create_new_customer (step 3)
            # Clear the creation state
            st.session_state.customer_creation_state = None
            # Show one clean success message for the whole flow
            st.success("Customer saved successfully!")

# Initialize session state
if not st.session_state.get("messages", None):
    st.session_state.messages = []

if not st.session_state.get("authenticated", None):
    st.session_state.authenticated = False

if not st.session_state.get("user", None):
    st.session_state.user = None

# Check for logout flag and clear it after processing
if st.session_state.get("logout_requested", False):
    st.session_state.logout_requested = False
    st.rerun()

# --- All function definitions (move these to the top, before main logic) ---
def get_customer_interactions(customer_id: str):
    """Fetch all interactions for a specific customer from the customers table"""
    try:
        response = supabase_client.table('customers').select('*').eq('customer_id', customer_id).single().execute()
        if response.data:
            # Combine input and output conversations into interactions
            interactions = []
            for i, (input_msg, output_msg) in enumerate(zip(
                response.data.get('input_conversation', []),
                response.data.get('output_conversation', [])
            )):
                interactions.append({
                    'interaction_input': input_msg,
                    'llm_output_summary': output_msg,
                    'created_at': response.data.get('created_at')  # Using the customer's creation time
                })
            return interactions
        return []
    except Exception as e:
        st.error(f"Error fetching interactions: {str(e)}")
        return []

def delete_customer_interaction(customer_id: str, interaction_index: int):
    """Delete a single interaction (and aligned embedding/metadata) by index for a customer."""
    try:
        customer = supabase_client.table('customers').select("*").eq('customer_id', customer_id).single().execute()
        if not customer.data:
            st.error("Customer not found.")
            return False

        input_conversation = customer.data.get('input_conversation') or []
        output_conversation = customer.data.get('output_conversation') or []
        interaction_embeddings = customer.data.get('interaction_embeddings') or []
        interaction_metadata = customer.data.get('interaction_metadata') or []

        # Validate index against the shortest list among the aligned arrays
        aligned_len = min(len(input_conversation), len(output_conversation), len(interaction_embeddings), len(interaction_metadata))
        if interaction_index < 0 or interaction_index >= aligned_len:
            st.error("Invalid interaction index.")
            return False

        def remove_at(lst):
            if isinstance(lst, list) and 0 <= interaction_index < len(lst):
                return lst[:interaction_index] + lst[interaction_index+1:]
            return lst

        updated_inputs = remove_at(input_conversation)
        updated_outputs = remove_at(output_conversation)
        updated_embs = remove_at(interaction_embeddings)
        updated_metas = remove_at(interaction_metadata)

        response = supabase_client.table('customers').update({
            'input_conversation': updated_inputs,
            'output_conversation': updated_outputs,
            'interaction_embeddings': updated_embs,
            'interaction_metadata': updated_metas,
            'updated_at': datetime.datetime.now().isoformat()
        }).eq('customer_id', customer_id).execute()

        return bool(response.data)
    except Exception as e:
        st.error(f"Error deleting interaction: {str(e)}")
        return False

def delete_customer(customer_id: str):
    """Delete a customer and all their data from the customers table."""
    try:
        response = supabase_client.table('customers').delete().eq('customer_id', customer_id).execute()
        return bool(response.data)
    except Exception as e:
        st.error(f"Error deleting customer: {str(e)}")
        return False

def analyze_deals_multi(
        new_interaction: str,
        past_context: str,
        last_deal_block: str = "",
        now_iso: str = None):
    """
    Multi-deal manager – no new tables required.
    Reads the LAST_DEAL_BLOCK text, decides update vs new,
    and returns updated tables + follow-up questions.
    """
    import datetime, json
    if not now_iso:
        now_iso = datetime.datetime.utcnow().isoformat()

    system_prompt = f"""
You are "LeanChem Deal-Management Analyst".

▼ DATA PROVIDED
LAST_DEAL_BLOCK (may be empty)
\"\"\"{last_deal_block}\"\"\"

PAST_INTERACTIONS (extra context)
\"\"\"{past_context}\"\"\"

NEW_INTERACTION
\"\"\"{new_interaction}\"\"\"

CURRENT_UTC = {now_iso}

===============  TASKS  ===============
1. Parse LAST_DEAL_BLOCK (if any) into two lists:
   • open_deals   (Stage ∈ Open, InProcess)  
   • closed_deals (Stage ∈ Won, Lost)

   Required keys in each deal-dict  
   {{Deal_ID, Customer, Product, Qty, Price, Currency, Incoterm,
     Stage, Last_Event, Progress, Last_Update_ISO}}

   *Progress* is a short string that shows how far the buyer has moved
   through the chemical-purchase steps:  
   "RFQ → TDS → Sample → Quote → PO → Payment → Delivered → Closed".

2. Detect every DEAL EVENT present in NEW_INTERACTION.  
   Codes: 1 RFQ  2 TDSReq  3 SampleReq  4 InfoSent/TDSSent  5 QuoteSent  
          6 OrderConfirmed  7 Payment  8 Transit  9 Delivered  
          10 Closed-Won  11 Closed-Lost  12 Other.

3. For each event decide
   ▸ Update an existing deal (match by Customer + Product + any PO/Quote ref), or  
   ▸ Create a new deal (generate sequential Deal_ID D-001, D-002…).

4. Update **Last_Event**, **Stage**, **Progress** (append the new step),
   and **Last_Update_ISO**.  
   Keep only Stage ∈ {{Open, InProcess}} in open_deals; move the rest to closed_deals.

5. Build a concise NARRATIVE (≤ 5 bullets) that tells
   the story of each open deal from day 1 to now, e.g.  
   "D-001: RFQ received → TDS sent → 25 kg sample delivered → waiting for PO".

6. Prepare FOLLOW-UP QUESTIONS  
   • Ask for any missing Qty / Price / Currency / Incoterm.  
   • Ask for an update if Last_Update_ISO is >72 h ago.

===============  OUTPUT  ===============
Return *markdown only* with **three sections** in this exact order:

CURRENT DEALS:
| Deal_ID | Product | Qty | Price | Stage | Progress | Last_Update |
|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … |

CLOSED DEALS:
| Deal_ID | Product | Outcome | Qty | Price | Progress |
|---|---|---|---|---|---|
| … | … | … | … | … | … |

DEAL NARRATIVE:
- D-001 – …  
- D-002 – …  

FOLLOW-UP QUESTIONS:
- …
- …

Return nothing else.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Update the deal tables and generate follow-up questions."}
    ]
    return gemini_chat(messages)

def sales_stage_tracker(new_interaction: str,
                        past_context: str,
                        last_stage_block: str = ""):
    """
    Narrative version of Brian-Tracy's 7-stage tracker.
    Returns two blocks:
      • SALES STAGE STATUS  – one-line narrative for each stage 1-7
      • ACTION PLAN         – ≤3 bullets to advance / expand the deal
    """

    system_prompt = f"""You are "LeanChem 7-Stage Narrative Tracker".
                
▼ PREVIOUS_STATUS (may be empty)
\"\"\"{last_stage_block}\"\"\"

▼ PAST_CONTEXT  (customer profile + earlier interactions)
\"\"\"{past_context}\"\"\"

▼ NEW_INTERACTION
\"\"\"{new_interaction}\"\"\"

=================  TASKS  =================
1.  If PREVIOUS_STATUS is empty → assume **Stage 1 Prospecting = CURRENT**.

2.  Otherwise read the highest completed stage (curr_stage_old).

3.  Analyse NEW_INTERACTION → decide stage_new  
    • If curr_stage_old ≥ 6 → lock stage_new = 7 unless this is a routine follow-up.

4.  For **each of the seven stages** write:

Stage N – <Name>: **STATUS**  
• Evidence 1 (from earlier interactions, if any)  
• Evidence 2 …  
• New Evidence – derived from NEW_INTERACTION (if relevant)

Allowed STATUS words (MUST be upper-case and in bold):  
**DONE**   = fully completed in the past  
**CURRENT** = active focus right now  
**PENDING** = not achieved yet

5.  After the seven stages add an "ACTION PLAN" block –  
    maximum 3 concise bullets describing what LeanChem should do next.

=================  OUTPUT  =================
Return **markdown only**, nothing else:

SALES STAGE STATUS:
- Stage 1 – Prospecting: **STATUS**  
    • Evidence 1 .. 
    • Evidence 2 ..
    • Evidence N ..
- Stage 2 – Rapport: **STATUS**  
    • Evidence 1 .. 
    • Evidence 2 ..
    • Evidence N ..
- Stage 7 – Follow-up & Cross-sell: **STATUS**  
    • Evidence 1 .. 
    • Evidence 2 ..
    • Evidence N ..

ACTION PLAN:
- …
- …
- …
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Return the SALES STAGE STATUS block and ACTION PLAN."}
    ]
    return gemini_chat(messages)

def suggest_next_action(new_interaction: str,
                        past_context: str,
                        deal_analysis: str,
                        sales_stage: str):
    """
    Return the next best action **plus a table of enablers**
    (each enabler scored for impact and willingness).
    """
    system_prompt = f"""
You are LeanChem's CRM copilot.

▼ DATA SOURCE PRIORITY
Pull the *Enablers (people)* only from:
1.  "Key Contacts for Engagement" section of the customer profile  
2.  Names mentioned in NEW_INTERACTION  
3.  Names mentioned in any historical interaction logs for the same customer

Ignore anyone else.

▼ CONTEXT
PAST_CONTEXT:
\"\"\"{past_context}\"\"\"

NEW_INTERACTION:
\"\"\"{new_interaction}\"\"\"

DEAL_ANALYSIS:
\"\"\"{deal_analysis}\"\"\"

SALES_STAGE_ANALYSIS:
\"\"\"{sales_stage}\"\"\"

▼ TASK
1. Draft one **Primary Action** plus up to four supporting tasks—sales-focused and concrete.  
2. Give an indicative timeline (e.g. "within 3 days").  
3. Build an **Enablers table** that lists the people who can influence the outcome.

   For every person found (max 8 rows):
   • Infer their **Position / Connection** to the customer organisation  
   • Score their **Impact** on the deal (High / Med / Low)  
   • Score their **Willingness** to help (High / Med / Low) using available clues  
   • Recommend a *Suggested Management Approach* in ≤25 words

▼ OUTPUT – **markdown only**
Suggested Next Action:
- Primary Action: …
- Supporting Tasks:
  • …
  • …
- Timeline: …

Enablers:
| Name | Position / Connection | Impact | Willingness | Suggested Management Approach |
|------|-----------------------|--------|-------------|--------------------------------|
| …    | …                     | …      | …           | … |
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Generate the action plan and enablers table."}
    ]

    # stream or single-chunk depending on provider
    response = get_llm_response(messages, model)
    return gemini_chat(messages)

def analyze_customer_update(update_text: str, customer_id: str, customer_name: str):
    """Analyze customer update with AI reasoning (Brian Tracy 7-stage journey style), using all past interactions and the new interaction."""
    # Fetch all past interactions for this customer
    interactions = get_customer_interactions(customer_id)
    # Format the interaction history for the prompt
    history_str = ""
    for i, interaction in enumerate(interactions, 1):
        history_str += f"Interaction {i}:\nInput: {interaction['interaction_input']}\nOutput: {interaction['llm_output_summary']}\n"
    # Compose the user prompt
    user_prompt = f"Customer: {customer_name}\n\nHistorical Interactions:\n{history_str}\nNEW_INTERACTION:\n{update_text}\n"
    system_prompt = '''You are "LeanChem AI CRM Analyst", an expert B2B sales assistant trained in Brian Tracy's methods.
Follow the 7-stage journey:
1 Prospecting\t2 Rapport\t3 Needs Analysis\t4 Present\t5 Objections\t6 Close\t7 Follow-up
For every customer you will:
* Retrieve the full record from the key "customerdatabase".
* Summarise existing information stage-by-stage (max 70 words per stage).
* If NEW_INTERACTION is present, append it to interactions[], overwriting conflicting facts; then re-evaluate all stage scores (1–10).
* Map every enabler with {impact, willingness}.
* Assign sequential Deal-IDs when stage 6 is triggered.
* Return PLAIN TEXT with two blocks: \n  A) 7-Stage Summary\tB) Insights & Action Plan (Brian Tracy style, ≤200 words)
'''
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return gemini_chat(messages)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def update_customer_interaction(customer_id: str, new_input: str, new_output: str, user_id: str):
    """Update customer interaction, storing a structured JSON object in the metadata."""
    # 1. Generate embedding for the new input
    embedding = gemini_embed(new_input)
    print("DEBUG: embedding from gemini_embed:", embedding, type(embedding))
    import json
    import numpy as np
    # --- Robust ensure_vector function ---
    def ensure_vector(x):
        # If it's already a list of floats, return as is
        if isinstance(x, list):
            # If it's a list of lists (shouldn't happen), flatten
            if len(x) > 0 and isinstance(x[0], list):
                return x[0]
            return x
        # If it's a string, try to parse as JSON list, or wrap as list
        if isinstance(x, str):
            try:
                val = json.loads(x)
                if isinstance(val, list):
                    return val
                else:
                    return [float(val)]
            except Exception:
                return [float(x)]
        # If it's a float or int, wrap as list
        if isinstance(x, float) or isinstance(x, int):
            return [float(x)]
        # If it's a numpy array
        if isinstance(x, np.ndarray):
            return x.tolist()
        # Fallback: wrap as list
        return [float(x)]

    embedding = ensure_vector(embedding)
    # 2. Create the new interaction object (as JSON)
    new_interaction_json = {
        "input": new_input,
        "output": new_output,
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id
    }

    # 3. Fetch existing data
    customer = supabase_client.table('customers').select("*").eq('customer_id', customer_id).single().execute()
    inps = customer.data.get('input_conversation') or []
    outs = customer.data.get('output_conversation') or []
    embs = customer.data.get('interaction_embeddings') or []
    metas = customer.data.get('interaction_metadata') or []  # This is now a list of JSON objects

    # Defensive: ensure all are lists
    if not isinstance(inps, list):
        inps = list(inps) if inps else []
    if not isinstance(outs, list):
        outs = list(outs) if outs else []
    if not isinstance(embs, list):
        embs = list(embs) if embs else []
    if not isinstance(metas, list):
        metas = list(metas) if metas else []

    # 4. Append new data
    updated_inputs = inps + [new_input]
    updated_outputs = outs + [new_output]
    updated_embs = embs + [embedding]  # list of lists of floats
    updated_metas = metas + [new_interaction_json]  # list of dicts (JSON)

    # --- Ensure every embedding is a list of floats ---
    updated_embs = [ensure_vector(e) for e in updated_embs]
    # --- Ensure every metadata is a dict ---
    updated_metas = [m if isinstance(m, dict) else json.loads(m) for m in updated_metas]

    # Debug: Print types and sample data before update
    print("DEBUG: Types and lengths before update:")
    print("updated_inputs:", type(updated_inputs), len(updated_inputs))
    print("updated_outputs:", type(updated_outputs), len(updated_outputs))
    print("updated_embs:", type(updated_embs), len(updated_embs))
    print("updated_metas:", type(updated_metas), len(updated_metas))
    print("Sample embedding (last):", updated_embs[-1], type(updated_embs[-1]))

    # 5. Save
    try:
        response = supabase_client.table('customers').update({
            'input_conversation': updated_inputs,
            'output_conversation': updated_outputs,
            'interaction_embeddings': updated_embs,  # list of lists of floats
            'interaction_metadata': updated_metas,    # list of dicts (JSON)
            'updated_at': datetime.datetime.now().isoformat()
        }).eq('customer_id', customer_id).execute()
        
        # 6. Send standardized interaction notification (do not depend on response.data)
        try:
            customer_name = 'Unknown Customer'
            try:
                if customer and getattr(customer, 'data', None):
                    customer_name = customer.data.get('customer_name', customer_name)
            except Exception:
                pass
            actor = get_actor_display(user_id)
            interaction_notification_ok = send_interaction_notification(
                customer_name=customer_name,
                customer_id=customer_id,
                actor=actor,
                input_text=new_input,
                output_text=new_output,
                timestamp=datetime.datetime.now()
            )
            if not interaction_notification_ok:
                print(
                    f"Warning: Interaction notification not delivered for "
                    f"customer_id={customer_id}."
                )
        except Exception:
            pass
        
        return response.data
    except Exception as e:
        print("Supabase update error:", e)
        st.error(f"Supabase update error: {e}")
        raise

def retrieve_relevant_interactions(customer_id: str, query: str, top_k: int = 3):
    """Retrieve the most relevant past interactions using vector similarity, returning full JSON objects."""
    try:
        # Use Gemini embeddings only; never call OpenAI here.
        query_embedding = gemini_embed(query)
        print("DEBUG: query_embedding from gemini_embed:", type(query_embedding))
        import json
        import numpy as np

        if isinstance(query_embedding, str):
            try:
                query_embedding = json.loads(query_embedding)
            except Exception:
                query_embedding = [float(query_embedding)]
        elif isinstance(query_embedding, (float, int)):
            query_embedding = [query_embedding]
        elif isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        customer = supabase_client.table('customers').select(
            "interaction_embeddings, interaction_metadata"
        ).eq('customer_id', customer_id).single().execute()

        if not customer.data:
            return []

        embs = customer.data.get('interaction_embeddings', [])
        metas = customer.data.get('interaction_metadata', [])
        if not embs or not metas:
            return []

        # Keep lists aligned and skip invalid shapes gracefully.
        min_len = min(len(embs), len(metas))
        embs_np = np.array(embs[:min_len], dtype=float)
        metas = metas[:min_len]
        if embs_np.shape[0] == 0:
            return []

        query_np = np.array(query_embedding, dtype=float)
        if query_np.ndim != 1:
            query_np = query_np.flatten()
        if embs_np.ndim != 2 or embs_np.shape[1] != query_np.shape[0]:
            print(
                "retrieve_relevant_interactions: embedding dimension mismatch "
                f"(stored={embs_np.shape}, query={query_np.shape})"
            )
            return []

        similarities = embs_np @ query_np / (
            np.linalg.norm(embs_np, axis=1) * np.linalg.norm(query_np) + 1e-8
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for i in top_indices:
            interaction_json = metas[i]
            if not isinstance(interaction_json, dict):
                try:
                    interaction_json = json.loads(interaction_json)
                except Exception:
                    interaction_json = {"input": "", "output": str(interaction_json)}
            interaction_json = dict(interaction_json)
            interaction_json['similarity'] = float(similarities[i])
            results.append(interaction_json)
        return results
    except Exception as e:
        print(f"retrieve_relevant_interactions failed for customer {customer_id}: {str(e)}")
        return []

def extract_file_content(file):
    """Extract content from different file types"""
    try:
        file_type = file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            # Handle PDF files
            pdf_reader = PdfReader(file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
            return content
            
        elif file_type == 'txt':
            # Handle text files
            return file.getvalue().decode("utf-8")
            
        elif file_type == 'docx':
            # Handle Word documents
            import docx  
            doc = docx.Document(file)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        raise Exception(f"Error extracting file content: {str(e)}")

def process_uploaded_file(file, customer_id: str):
    """Process uploaded file, extract content, and generate summary."""
    try:
        # Extract file content based on file type
        file_content = extract_file_content(file)

        # Create a summary of the file content
        summary_prompt = f"""Analyze the following document content for CRM purposes:
{file_content}

Format the analysis to include:
1. Key points and main topics
2. Important details and specifications
3. Any specific requirements or needs mentioned
4. Potential business opportunities
5. Relevant product matches (RDP, SBR, HPMC)
"""

        messages = [
            {"role": "system", "content": "You are a document analysis assistant specialized in chemical trading. Analyze the content for CRM purposes, focusing on business opportunities and product matches."},
            {"role": "user", "content": summary_prompt}
        ]

        # Get AI summary
        summary = gemini_chat(messages)

        # Instead of storing directly, return the content and summary
        # update_customer_interaction(
        #     customer_id,
        #     f"Uploaded file: {file.name}\nContent: {file_content}",
        #     f"File Analysis:\n{summary}"
        # )

        # Return both the file content and the generated summary
        return True, "Document analyzed successfully! Review analysis below and save.", file_content, summary
    except Exception as e:
        return False, f"Error processing file: {str(e)}", None, None

def render_update_interaction_ui(user_id: str):
    """Render the Update Interaction window UI."""

    # State 1: Customer Selection
    if 'selected_customer_for_update' not in st.session_state or st.session_state['selected_customer_for_update'] is None:
        st.subheader("Select Customer to Update")

        customers_dict = get_all_customer_names()
        if not customers_dict:
            st.warning("No customers found. Please create a customer first.")
            return

        sorted_customer_names = sorted(customers_dict.keys())

        col_dropdown, col_button = st.columns([0.7, 0.3])

        with col_dropdown:
            selected_customer_name = st.selectbox(
                "Select Customer",
                options=sorted_customer_names,
                key="update_interaction_customer_select"
            )

        with col_button:
            st.markdown("<br>", unsafe_allow_html=True) # Add a small space above the button
            select_button = st.button("Select", key="select_customer_button", type="primary")

        # Store the selected customer ID if the button is clicked and trigger rerun
        if select_button and selected_customer_name:
            selected_customer_id = customers_dict.get(selected_customer_name)
            if selected_customer_id:
                st.session_state['selected_customer_for_update'] = {
                    'name': selected_customer_name,
                    'id': selected_customer_id
                }
                st.rerun() # Rerun to show the interaction details

    # State 2, 3, 4: Interaction Details and Adding New Data
    else:
        selected_customer = st.session_state['selected_customer_for_update']
        customer_name = selected_customer['name']
        customer_id = selected_customer['id']

        st.subheader(f"Interactions with {customer_name}")

        # Display interaction history (Collapsible Cards)
        interactions = get_customer_interactions(customer_id)
        if interactions:
            for idx, interaction in enumerate(reversed(interactions)):
                with st.expander(f"{interaction['created_at']} | Interaction #{len(interactions)-idx}"):
                    st.markdown(f"**Input:** {interaction['interaction_input']}")
                    st.markdown(f"**AI Output:** {interaction['llm_output_summary']}")
                    # Delete button for this interaction
                    col_a, col_b = st.columns([0.2, 0.8])
                    with col_a:
                        # Map reversed display index back to original index in arrays
                        original_index = len(interactions) - 1 - idx
                        if st.button("🗑️ Delete", key=f"delete_interaction_{original_index}"):
                            success = delete_customer_interaction(customer_id, original_index)
                            if success:
                                st.success("Interaction deleted.")
                                st.rerun()
                            else:
                                st.error("Failed to delete interaction.")
            if st.button("Select Another Customer"):
                st.session_state['selected_customer_for_update'] = None
                if 'current_interaction_analysis' in st.session_state:
                    st.session_state['current_interaction_analysis'] = None
                if 'current_file_analysis' in st.session_state:
                    st.session_state['current_file_analysis'] = None
                st.rerun()
        else:
            st.info("No interactions found for this customer yet.")
            if st.button("Select Another Customer"):
                st.session_state['selected_customer_for_update'] = None
                if 'current_interaction_analysis' in st.session_state:
                    st.session_state['current_interaction_analysis'] = None
                if 'current_file_analysis' in st.session_state:
                    st.session_state['current_file_analysis'] = None
                st.rerun()

        st.markdown("\n---\n")

        # --- Section for adding New Interactions and Uploads ---
        st.subheader("Add New Interaction or Upload Document")

        # State 3: Display file analysis and Save/Cancel buttons if available in session state
        if 'current_file_analysis' in st.session_state and st.session_state['current_file_analysis'] is not None:
            file_analysis_data = st.session_state['current_file_analysis']
            st.subheader("📄 File Analysis Ready to Save")
            st.write(f"**File:** {file_analysis_data['file_name']}")
            st.markdown("**Extracted Content (Input):**")
            st.expander("View full content").markdown(f"```\n{file_analysis_data['file_content']}\n```")
            st.markdown("**AI Summary (Output):**")
            # Tabs for AI summary (if you want to split by sections, you can parse summary here)
            st.write(file_analysis_data['summary'])
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save File Interaction", key="save_file_interaction_button"):
                    llm_output = f"File Analysis:\n{file_analysis_data['summary']}"
                    interaction_input = f"Uploaded file: {file_analysis_data['file_name']}. Content summary: {file_analysis_data['file_content'][:200]}..."
                    result = update_customer_interaction(customer_id, interaction_input, llm_output, user_id)
                    if result:
                        st.success(f"File analysis for {file_analysis_data['file_name']} saved successfully!")
                        st.session_state['current_file_analysis'] = None
                        st.rerun()
                    else:
                        st.error("Failed to save file analysis")
            with col2:
                if st.button("❌ Cancel", key="cancel_file_interaction_button"):
                    st.session_state['current_file_analysis'] = None
                    st.rerun()
            st.markdown("\n---\n")

        # State 4: Display analysis and Save/Cancel buttons for text interaction if available in session state
        elif 'current_interaction_analysis' in st.session_state and st.session_state['current_interaction_analysis'] is not None:
            analysis_data = st.session_state['current_interaction_analysis']
            st.subheader("🤖 AI Analysis")
            # Tabs for AI output
            tabs = st.tabs(["Deal Analysis", "Sales Stage", "Next Action"])
            with tabs[0]:
                st.write(analysis_data['deal_analysis'])
            with tabs[1]:
                st.write(analysis_data['sales_stage_tracker'])
            with tabs[2]:
                st.write(analysis_data.get('next_action_str', analysis_data.get('suggest_next_action')))
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Interaction", key="save_interaction_button"):
                    llm_output = f"Deal Analysis:\n{analysis_data['deal_analysis']}\n\nSales Stage:\n{analysis_data['sales_stage_tracker']}\n\nNext Action:\n{analysis_data.get('next_action_str', analysis_data.get('suggest_next_action'))}"
                    result = update_customer_interaction(customer_id, analysis_data['new_interaction'], llm_output, user_id)
                    if result:
                        st.success("Interaction saved successfully!")
                        st.session_state['current_interaction_analysis'] = None
                        st.rerun()
                    else:
                        st.error("Failed to save interaction")
            with col2:
                if st.button("❌ Cancel", key="cancel_interaction_button"):
                    st.session_state['current_interaction_analysis'] = None
                    st.rerun()
            st.markdown("\n---\n")

        # State 5: Default state for adding new interactions/uploads
        else:
            st.subheader("✍️ New Interaction")
            st.caption("Add a new customer interaction and analyze it with AI, or ask any question about this customer.")
            new_interaction = st.text_area("📝 Enter new interaction details or any question", key="new_interaction_textarea")
            if st.button("💡 Analyze with AI", key="analyze_interaction_button") and new_interaction:
                # Meta-query detection for summarize (always use selected customer)
                if "summarize" in new_interaction.lower():
                    try:
                        summary = summarize_interactions_with_customer(customer_name, user_id)
                        st.session_state['current_interaction_analysis'] = {
                            'new_interaction': new_interaction,
                            'deal_analysis': summary,
                            'sales_stage_tracker': summary,
                            'next_action_str': summary
                        }
                    except RetryError as e:
                        root_error = e.last_attempt.exception() if e.last_attempt else e
                        print(f"Summarize exhausted retries: {str(root_error)}")
                        st.warning("Please wait 1 minute and try again.")
                        st.session_state['current_interaction_analysis'] = {
                            'new_interaction': new_interaction,
                            'deal_analysis': "Rate limit reached. Please wait 1 minute and try again.",
                            'sales_stage_tracker': "Rate limit reached. Please wait 1 minute and try again.",
                            'next_action_str': "Rate limit reached. Please wait 1 minute and try again."
                        }
                    except requests.exceptions.HTTPError as e:
                        error_msg = str(e)
                        st.error(f"❌ Gemini API Error: {error_msg}")
                        st.warning("⚠️ Please check your GEMINI_API_KEY and try again.")
                        st.session_state['current_interaction_analysis'] = {
                            'new_interaction': new_interaction,
                            'deal_analysis': f"Error: Could not summarize. {error_msg}",
                            'sales_stage_tracker': f"Error: {error_msg}",
                            'next_action_str': f"Error: {error_msg}"
                        }
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ Error summarizing interactions: {error_msg}")
                        st.session_state['current_interaction_analysis'] = {
                            'new_interaction': new_interaction,
                            'deal_analysis': f"Error: {error_msg}",
                            'sales_stage_tracker': f"Error: {error_msg}",
                            'next_action_str': f"Error: {error_msg}"
                        }
                    st.rerun()
                with st.spinner("API rate limit reached. Waiting for Google to allow the next request... this may take up to a minute."):
                    # 1. Retrieve relevant past interactions for context (always use selected customer)
                    relevant_interactions = retrieve_relevant_interactions(customer_id, new_interaction, top_k=3)
                    past_context = "No relevant past interactions found."
                    if relevant_interactions:
                        past_context = "Relevant Past Interactions (for context):\n"
                        for interaction in relevant_interactions:
                            past_context += f"- User: {interaction['input']}\\n- AI: {interaction['output']}\\n"
                    # --- Improved meta-query detection ---
                    is_question = new_interaction.strip().endswith('?')
                    is_summarize = "summarize" in new_interaction.lower()
                    is_latest = "latest interaction" in new_interaction.lower() or "last interaction" in new_interaction.lower()

                    if is_question or is_summarize or is_latest:
                        # Use open-ended RAG answer for questions and meta-queries
                        try:
                            open_answer = answer_any_query_with_rag(new_interaction, customer_id, user_id, top_k=3)
                            st.session_state['current_interaction_analysis'] = {
                                'new_interaction': new_interaction,
                                'deal_analysis': open_answer,
                                'sales_stage_tracker': open_answer,
                                'next_action_str': open_answer
                            }
                        except RetryError as e:
                            root_error = e.last_attempt.exception() if e.last_attempt else e
                            print(f"Query answer exhausted retries: {str(root_error)}")
                            st.warning("Please wait 1 minute and try again.")
                            st.session_state['current_interaction_analysis'] = {
                                'new_interaction': new_interaction,
                                'deal_analysis': "Rate limit reached. Please wait 1 minute and try again.",
                                'sales_stage_tracker': "Rate limit reached. Please wait 1 minute and try again.",
                                'next_action_str': "Rate limit reached. Please wait 1 minute and try again."
                            }
                        except requests.exceptions.HTTPError as e:
                            error_msg = str(e)
                            st.error(f"❌ Gemini API Error: {error_msg}")
                            st.warning("⚠️ Please check your GEMINI_API_KEY and try again.")
                            st.session_state['current_interaction_analysis'] = {
                                'new_interaction': new_interaction,
                                'deal_analysis': f"Error: Could not answer query. {error_msg}",
                                'sales_stage_tracker': f"Error: {error_msg}",
                                'next_action_str': f"Error: {error_msg}"
                            }
                        except Exception as e:
                            error_msg = str(e)
                            st.error(f"❌ Error answering query: {error_msg}")
                            st.session_state['current_interaction_analysis'] = {
                                'new_interaction': new_interaction,
                                'deal_analysis': f"Error: {error_msg}",
                                'sales_stage_tracker': f"Error: {error_msg}",
                                'next_action_str': f"Error: {error_msg}"
                            }
                    else:
                        # Always run classic sales analysis for normal sales interactions
                        try:
                            last_deal_block = interactions[-1]['llm_output_summary'] if interactions else ""
                            deal_analysis = analyze_deals_multi(new_interaction, past_context,last_deal_block)
                            last_stage_block = interactions[-1]['llm_output_summary'] if interactions else ""
                            stage_narrative = sales_stage_tracker(new_interaction, past_context, last_stage_block)
                            next_action_str = suggest_next_action(new_interaction, past_context, deal_analysis, stage_narrative)
                            st.session_state['current_interaction_analysis'] = {
                                'new_interaction': new_interaction,
                                'deal_analysis': deal_analysis,
                                'sales_stage_tracker': stage_narrative,
                                'next_action_str': next_action_str
                            }
                        except RetryError as e:
                            root_error = e.last_attempt.exception() if e.last_attempt else e
                            print(f"Classic analysis exhausted retries: {str(root_error)}")
                            st.warning("Please wait 1 minute and try again.")
                            st.session_state['current_interaction_analysis'] = {
                                'new_interaction': new_interaction,
                                'deal_analysis': "Rate limit reached. Please wait 1 minute and try again.",
                                'sales_stage_tracker': "Rate limit reached. Please wait 1 minute and try again.",
                                'next_action_str': "Rate limit reached. Please wait 1 minute and try again."
                            }
                        except requests.exceptions.HTTPError as e:
                            error_msg = str(e)
                            st.error(f"❌ Gemini API Error: {error_msg}")
                            st.warning("⚠️ Please check your GEMINI_API_KEY and try again. The analysis could not be completed.")
                            # Store a partial analysis with error message
                            st.session_state['current_interaction_analysis'] = {
                                'new_interaction': new_interaction,
                                'deal_analysis': f"Error: Could not analyze deals. {error_msg}",
                                'sales_stage_tracker': f"Error: Could not analyze sales stage. {error_msg}",
                                'next_action_str': f"Error: Could not suggest next action. {error_msg}"
                            }
                        except Exception as e:
                            error_msg = str(e)
                            st.error(f"❌ Unexpected error during analysis: {error_msg}")
                            st.warning("⚠️ Please try again. If the issue persists, contact support.")
                            # Store a partial analysis with error message
                            st.session_state['current_interaction_analysis'] = {
                                'new_interaction': new_interaction,
                                'deal_analysis': f"Error: {error_msg}",
                                'sales_stage_tracker': f"Error: {error_msg}",
                                'next_action_str': f"Error: {error_msg}"
                            }
                st.rerun()
            st.markdown("\n---\n")
            st.subheader("Upload Document")
            uploaded_file = st.file_uploader(
                "Upload a document (PDF, TXT, DOCX)",
                type=['pdf', 'txt', 'docx'],
                key="customer_file_upload"
            )
            if uploaded_file:
                if st.button("Process Document", key="process_uploaded_file_button"):
                    success, message, file_content, summary = process_uploaded_file(uploaded_file, customer_id)
                    if success:
                        st.success(message)
                        st.session_state['current_file_analysis'] = {
                            'file_name': uploaded_file.name,
                            'file_content': file_content,
                            'summary': summary
                        }
                        st.rerun()
                    else:
                        st.error(message)

        # --- Danger Zone: Delete Customer ---
        st.markdown("\n---\n")
        st.subheader("⚠️ Danger Zone")
        st.caption("Deleting a customer will remove all their interactions and cannot be undone.")
        confirm = st.checkbox("I understand this action is irreversible.", key="confirm_delete_customer")
        if st.button("🗑️ Delete Customer", key="delete_customer_button"):
            if confirm:
                with st.spinner("Deleting customer..."):
                    ok = delete_customer(customer_id)
                if ok:
                    st.success(f"Customer '{customer_name}' deleted.")
                    # Clear selection and any transient state
                    st.session_state['selected_customer_for_update'] = None
                    if 'current_interaction_analysis' in st.session_state:
                        st.session_state['current_interaction_analysis'] = None
                    if 'current_file_analysis' in st.session_state:
                        st.session_state['current_file_analysis'] = None
                    st.rerun()
                else:
                    st.error("Failed to delete customer. Please try again.")
            else:
                st.warning("Please confirm the irreversible action before deleting.")

def get_all_customer_data():
    """Fetch all customer data with their interactions"""
    try:
        response = supabase_client.table('customers').select('*').execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching customer data: {str(e)}")
        return []

def analyze_crm_data(query: str, user_id: str):
    """Analyze CRM data based on natural language query (RAG-ENABLED VERSION)"""
    customers = get_all_customer_data()
    
    context = "Customer Data:\n"
    for customer in customers:
        context += f"\nCustomer: {customer['customer_name']}\n"
        context += f"Display ID: {customer.get('display_id', 'N/A')}\n"
        context += f"Created: {customer.get('created_at', 'N/A')}\n"
        context += f"Last Updated: {customer.get('updated_at', 'N/A')}\n"
        if customer.get('input_conversation'):
            context += "\nRecent Interactions:\n"
            for i, (input_msg, output_msg) in enumerate(zip(
                customer['input_conversation'][-3:],
                customer['output_conversation'][-3:]
            )):
                context += f"\nInteraction {i+1}:\n"
                context += f"Input: {input_msg}\n"
                context += f"Output: {output_msg}\n"
    
    # Filter out empty/irrelevant memories
    relevant_memories = get_cached_memories(query, user_id)
    memories_str = "\n".join(
        f"- {entry['memory']}" for entry in relevant_memories["results"]
        if entry['memory'] and entry['memory'].strip() and entry['memory'].strip().lower() != "not specified"
    )

    # --- RAG: Retrieve most relevant interactions across all customers ---
    # Gather all interactions and embeddings
    import numpy as np
    all_interactions = []
    all_embeddings = []
    for customer in customers:
        embeddings = customer.get('interaction_embeddings', [])
        metas = customer.get('interaction_metadata', [])
        if not isinstance(embeddings, list) or not isinstance(metas, list):
            continue
        min_len = min(len(embeddings), len(metas))
        for i in range(min_len):
            all_embeddings.append(embeddings[i])
            all_interactions.append(metas[i])
    # Get query embedding
    try:
        query_embedding = gemini_embed(query)
        if isinstance(query_embedding, str):
            import json
            query_embedding = json.loads(query_embedding)
        query_np = np.array(query_embedding)
        embs_np = np.array(all_embeddings)
        if embs_np.shape[0] > 0:
            similarities = embs_np @ query_np / (np.linalg.norm(embs_np, axis=1) * np.linalg.norm(query_np) + 1e-8)
            top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5
            rag_context = "\nRAG: Most Relevant Past Interactions (All Customers):\n"
            for idx in top_indices:
                meta = all_interactions[idx]
                rag_context += f"\n- Customer: {meta.get('customer_name', 'N/A')}\n  Input: {meta.get('input', '')}\n  Output: {meta.get('output', '')}\n  Timestamp: {meta.get('timestamp', '')}\n  Similarity: {similarities[idx]:.2f}\n"
        else:
            rag_context = "\n(No relevant past interactions found for RAG)\n"
    except Exception as e:
        rag_context = f"\n(RAG retrieval error: {str(e)})\n"

    system_prompt = """You are a CRM data analyst specialized in chemical trading. Analyze the provided data and answer the user's query.
    Use the following guidelines:
    1. Be concise but comprehensive
    2. Structure your response in a clear, readable format
    3. Include relevant numbers and statistics when available
    4. Highlight important trends or patterns
    5. Suggest actionable insights when relevant
    \nAvailable Data:\n{context}\n\nRelevant Memories:\n{memories}\n\n{rag_context}\n\nFormat your response as:
    Analysis:
    [Your analysis here]
    \nKey Findings:
    - [Finding 1]
    - [Finding 2]
    ...
    \nRecommendations:
    - [Recommendation 1]
    - [Recommendation 2]
    ...
    \nDeals:
    - [Deal 1]
    - [Deal 2]
    """
    
    messages = [
        {"role": "system", "content": system_prompt.format(context=context, memories=memories_str, rag_context=rag_context)},
        {"role": "user", "content": query}
    ]
    try:
        return gemini_chat(messages)
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return f"OpenAI error: {e}"

def save_analysis_query(query: str, response: str, user_id: str):
    """Save the analysis query and response to the deals table"""
    try:
        data = {
            "input_log": query,
            "ai_response_log": response,
            "created_by": user_id,
            "created_at": datetime.datetime.now().isoformat()
        }
        api_response = supabase_client.table('deals').insert(data).execute()
        return api_response.data[0] if api_response.data else None
    except Exception as e:
        st.error(f"Error saving analysis query: {str(e)}")
        return None

def get_saved_queries(user_id: str):
    """Fetch saved analysis queries for the user"""
    try:
        response = supabase_client.table('deals').select('*').eq('created_by', user_id).order('created_at', desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching saved queries: {str(e)}")
        return []

def render_analysis_ui(user_id: str):
    """Render the Analysis & Reporting window UI"""
    st.title("📊 CRM Analysis & Reporting")
    st.write("Ask any question about your CRM data and get instant insights.")
    st.markdown("---")

    # Toggle for saved queries
    show_saved = st.checkbox("Show Saved Queries", key="show_saved_analysis_queries")

    if show_saved:
        saved_queries = get_saved_queries(user_id)
        if saved_queries:
            st.subheader("💾 Saved Queries")
            for query in saved_queries:
                with st.expander(f"Query from {query['created_at']}"):
                    st.write("Query:", query['input_log'])
                    st.write("Response:", query['ai_response_log'])
        else:
            st.info("No saved queries found.")

    st.markdown("---")

    # --- New Analysis Section ---
    st.subheader("📝 New Analysis")
    st.write("Type your question about CRM data and click the button to analyze with AI.")

    analysis_query = st.text_area("Enter your analysis query", key="new_analysis_query_input")

    if st.button("💡 Analyze with AI", key="analyze_crm_button") and analysis_query:
        with st.spinner("Analyzing CRM data..."):
            analysis_response = analyze_crm_data(analysis_query, user_id)
            # Store analysis response and query in session state for display and saving
            st.session_state['current_crm_analysis'] = {
                'query': analysis_query,
                'response': analysis_response
            }
            st.rerun() # Rerun to display analysis and save button

    # Display current analysis response and Save button if available in session state
    if 'current_crm_analysis' in st.session_state and st.session_state['current_crm_analysis'] is not None:
        analysis_data = st.session_state['current_crm_analysis']
        st.subheader("🤖 AI Analysis Result")
        st.write("**Query:**", analysis_data['query'])
        st.write("**Response:**")
        st.write(analysis_data['response'])

        if st.button("💾 Save Analysis", key="save_current_analysis_button"):
            saved_query_data = save_analysis_query(analysis_data['query'], analysis_data['response'], user_id)
            if saved_query_data:
                st.success("Analysis saved successfully!")
                # Clear the current analysis from session state after saving
                st.session_state['current_crm_analysis'] = None
                st.rerun() # Rerun to refresh the saved queries list (if shown)
            else:
                st.error("Failed to save analysis")

    # Clear the input area after analysis is triggered or saved (optional, can be adjusted)
    # if 'new_analysis_query_input' in st.session_state:
    #     del st.session_state['new_analysis_query_input']

    # --- Telegram Notification Section (hidden by default) ---
    if os.getenv('NOTIFICATION_UI_VISIBLE', 'false').lower() == 'true':
        st.markdown("---")
        st.subheader("📱 Telegram Notifications")
        
        if NOTIFICATION_ENABLED:
            st.success("✅ Telegram notifications are enabled")
            if TELEGRAM_BOT_TOKEN:
                st.info(f"🤖 Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...")
            if TELEGRAM_CHAT_ID:
                st.info(f"💬 Chat ID: {TELEGRAM_CHAT_ID}")
        else:
            st.warning("⚠️ Telegram notifications are disabled")
            st.info("To enable notifications, set NOTIFICATION_ENABLED=true in your .env file")
        
        if can_show_notification_test_ui():
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📤 Send Test Notification", key="test_telegram_notification"):
                    if NOTIFICATION_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        with st.spinner("Sending test notification..."):
                            test_message = "*🧪 LeanChems CRM Test Notification*\n\nThis is a test message to verify Telegram integration is working correctly.\n\n*Timestamp:* " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            success = send_telegram_message_sync(test_message)
                            if success:
                                st.success("✅ Test notification sent successfully!")
                            else:
                                st.error("❌ Failed to send test notification")
                    else:
                        st.error("❌ Telegram notifications not properly configured")
            
            with col2:
                if st.button("📊 Send Daily Summary Now", key="send_daily_summary_now"):
                    if NOTIFICATION_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        with st.spinner("Generating and sending daily summary..."):
                            summary = generate_daily_deal_summary()
                            success = send_telegram_message_sync(summary)
                            if success:
                                st.success("✅ Daily summary sent successfully!")
                                st.info("Summary preview:")
                                st.text(summary[:500] + "..." if len(summary) > 500 else summary)
                            else:
                                st.error("❌ Failed to send daily summary")
                    else:
                        st.error("❌ Telegram notifications not properly configured")
            
            with col3:
                if st.button("🔄 Send Deal Update Test", key="send_deal_update_test"):
                    if NOTIFICATION_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        with st.spinner("Sending deal update test..."):
                            test_deal_info = "Test Deal D-001: RDP 1000kg - Quote sent - Waiting for PO"
                            success = send_interaction_notification(
                                customer_name="Test Customer",
                                customer_id="test-customer-id",
                                actor="Test User",
                                input_text="Test interaction input",
                                output_text=test_deal_info,
                                timestamp=datetime.datetime.now()
                            )
                            if success:
                                st.success("✅ Deal update test sent successfully!")
                            else:
                                st.error("❌ Failed to send deal update test")
                    else:
                        st.error("❌ Telegram notifications not properly configured")
        
        st.subheader("⚙️ Notification Settings")
        st.info("""
        **📅 Daily Notifications:** Automatically sent every morning at 8:00 AM
        
        **🔄 Real-time Updates:** Sent immediately when:
        - New customers are created
        - Deal updates occur (quotes, orders, status changes)
        
        **📊 What's included in daily summary:**
        - Summary of all active deals
        - Customer count and deal status
        - Action items for the day
        - Quick tips and insights
        
        **🔔 What's included in real-time updates:**
        - New customer alerts with profile info
        - Deal progress updates with key details
        - Immediate action items
        
        **⚙️ To configure:**
        1. Create a Telegram bot via @BotFather
        2. Get your bot token
        3. Get your chat ID (send a message to your bot and check @userinfobot)
        4. Add to your .env file:
           ```
           TELEGRAM_BOT_TOKEN=your_bot_token_here
           TELEGRAM_CHAT_ID=your_chat_id_here
           NOTIFICATION_ENABLED=true
           NOTIFICATION_UI_VISIBLE=false
           ```
        """)

def check_documents_table():
    """Check if there are any documents in the documents table"""
    try:
        response = supabase_client.table('documents').select('count').execute()
        return len(response.data) if response.data else 0
    except Exception as e:
        st.error(f"Error checking documents table: {str(e)}")
        return 0

def test_api_connectivity():
    """Test API connectivity and return status"""
    results = {}
    
    # Test Gemini API
    if GEMINI_API_KEY:
        try:
            # Simple test request
            test_payload = {"text": "test"}
            headers = {"Content-Type": "application/json"}
            params = {"key": GEMINI_API_KEY}
            
            response = requests.post(
                GEMINI_EMBED_URL,
                params=params,
                headers=headers,
                data=json.dumps(test_payload),
                timeout=10
            )
            
            if response.status_code == 200:
                results['gemini'] = "✅ Working"
            else:
                results['gemini'] = f"❌ Error {response.status_code}: {response.text[:100]}"
        except Exception as e:
            results['gemini'] = f"❌ Connection failed: {str(e)}"
    else:
        results['gemini'] = "❌ No API key"
    
    # Test OpenAI API
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            response = openai_client.embeddings.create(
                input="test",
                model="text-embedding-3-small"
            )
            results['openai'] = "✅ Working"
        except Exception as e:
            results['openai'] = f"❌ Error: {str(e)}"
    else:
        results['openai'] = "❌ No API key"
    
    # Test Supabase
    try:
        response = supabase_client.table('customers').select('count').limit(1).execute()
        results['supabase'] = "✅ Working"
    except Exception as e:
        results['supabase'] = f"❌ Error: {str(e)}"
    
    return results

# --- RAG Test UI ---
def render_rag_test_ui(user_id):
    import streamlit as st
    st.title("🔍 RAG Test: Conversation Retrieval")
    st.markdown("---")
    # Check if conversations exist in the database
    try:
        response = supabase_client.table('conversation').select('count').execute()
        conv_count = len(response.data) if response.data else 0
    except Exception as e:
        st.error(f"Error checking conversation table: {str(e)}")
        conv_count = 0
    if conv_count == 0:
        st.warning("⚠️ No conversations found in the database. Please upload some test conversations first.")
        st.subheader("📝 Upload Test Conversation")
        test_content = st.text_area(
            "Enter test conversation content",
            value="Customer inquiry about RDP (Redispersible Polymer Powder) for dry-mix mortar applications. Need technical specifications and pricing for construction projects in Ethiopia.",
            key="test_conversation_content"
        )
        if st.button("📤 Upload Test Conversation", key="upload_test_conversation_button"):
            if test_content:
                with st.spinner("Uploading test conversation..."):
                    result = upload_test_conversation(test_content, user_id)
                    if result:
                        st.rerun()
            else:
                st.warning("Please enter some content for the test conversation.")
        return
    st.subheader("🔍 Test Conversation Retrieval")
    query = st.text_input("Enter a query to test conversation retrieval", key="rag_test_query")
    limit = st.number_input("Number of results to fetch", min_value=1, max_value=10, value=3, step=1, key="rag_test_limit")
    if st.button("🔍 Run RAG Test", key="run_rag_test_button") and query:
        with st.spinner("Fetching relevant conversations from Supabase..."):
            conversations = search_documents(query, user_id, limit=limit)
        if conversations:
            # Combine all retrieved conversation contents
            context = "\n\n".join(conv.get('content', '') for conv in conversations)
            # Build a prompt for Gemini
            prompt = (
                f"Context:\n{context}\n\n"
                f"User Query: {query}\n\n"
                "Based on the above context, provide a concise, insightful answer to the user's query. "
                "If the context is insufficient, say so."
            )
            # Call Gemini for analysis
            analyzed_answer = gemini_chat([
                {"role": "system", "content": "You are an expert CRM assistant."},
                {"role": "user", "content": prompt}
            ])
            st.subheader("🤖 Gemini's Analyzed Answer")
            st.write(analyzed_answer)
            # Optionally, still show the raw retrieved conversations for transparency
            with st.expander("Show Retrieved Conversations"):
                for i, conv in enumerate(conversations, 1):
                    st.markdown(f"**Conversation {i}:**")
                    st.code(conv.get('content', '')[:2000], language='text')
                    if conv.get('similarity'):
                        st.info(f"Similarity Score: {conv.get('similarity', 'N/A'):.4f}")
                    st.markdown("---")
        else:
            st.warning("No relevant conversations found for this query.")
    st.caption("This tool helps you debug and verify your RAG pipeline by directly inspecting what the retriever returns from your uploaded conversations.")

# --- Customer search and update interaction for dashboard tab ---
def render_choose_existing_ui(user_id):
    # Removed Customer Search Section
    # st.subheader("Customer Search")
    # search_query = st.text_input("Search customers", key="tab_customer_search")
    # customers = search_customers(search_query) if search_query else get_all_customer_names()
    # if customers:
    #     st.write("Found customers:")
    #     for customer in (customers if isinstance(customers, list) else customers.keys()):
    #         name = customer['customer_name'] if isinstance(customer, dict) else customer
    #         with st.expander(name):
    #             if isinstance(customer, dict):
    #                 st.write("Created:", customer.get('created_at', 'N/A'))
    #                 st.write("Last updated:", customer.get('updated_at', 'N/A'))
    #                 if customer.get('input_conversation'):
    #                     st.write("Recent conversations:")
    #                     for i, (input_msg, output_msg) in enumerate(zip(customer['input_conversation'][-3:], customer['output_conversation'][-3:]), 1):
    #                         st.write(f"Conversation {i}:")
    #                         st.write("Input:", input_msg)
    #                         st.write("Output:", output_msg)
    # else:
    #     st.write("No customers found.")

    st.subheader("Customer Management")

    # Removed All Interactions Table Section
    # st.subheader("📋 All Interactions Table")
    # interactions = get_customer_interactions(customers[selected_customer])
    # if interactions:
    #     df = pd.DataFrame(interactions)
    #     # Reorder columns for clarity
    #     cols = [c for c in ["created_at", "interaction_input", "llm_output_summary"] if c in df.columns]
    #     df = df[cols]
    #     df = df.rename(columns={
    #         "created_at": "Date",
    #         "interaction_input": "Input",
    #         "ai_output_summary": "AI Output"
    #     })
    #     st.dataframe(df, use_container_width=True)
    #     csv = df.to_csv(index=False).encode('utf-8')
    #     st.download_button(
    #         label="Export Interactions to CSV",
    #         data=csv,
    #         file_name=f"{selected_customer}_interactions.csv",
    #         mime='text/csv',
    #         key="export_interactions_csv"
    #     )
    # else:
    #     st.info("No interactions found for this customer.")
    # st.markdown("---")

    # Removed View Mode Section
    # st.subheader("🗂️ View Mode")
    # view_mode = st.radio(
    #     "View Mode",
    #     ["Full Chat Thread", "Summarized Insight"],
    #     horizontal=True,
    #     key="update_interaction_view_mode"
    # )

    # The customer selection is now handled inside render_update_interaction_ui
    # Fetch customer interactions - This is now done inside render_update_interaction_ui
    # interactions = get_customer_interactions(customers[selected_customer])

    # if view_mode == "Full Chat Thread":
    #     st.subheader("💬 Interaction History")
    #     for interaction in interactions:
    #         with st.expander(f"Interaction on {interaction['created_at']}"):
    #             st.write("Input:", interaction['interaction_input'])
    #             st.write("Analysis:", interaction['llm_output_summary'])
    # else:
        # Generate summary - This is now done inside render_update_interaction_ui
    #    ...
    # st.markdown("---")

    render_update_interaction_ui(user_id)

    # Utility to overlay data on the static template PDF

def generate_quote_with_items(
    template_path,
    output_path,
    customer_name,
    items,
    start_y=380,   # moved down from 365 → 380 for better spacing below headers
    row_height=20
):
    reader = PdfReader(template_path)
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(width, height))
    can.setFont("Helvetica", 12)

    # --- CUSTOMER NAME ---
    # Move further down and aligned left to start near "CUSTOMER :"
    customer_x = 61.45  # aligned with "CUSTOMER :" start
    customer_y_offset = 237.55 + 35  # increased from +20 to +35 to move it down a little more
    can.drawString(customer_x, height - customer_y_offset, customer_name)

    # --- Column X positions with proper spacing to prevent overlap ---
    x_name = 60      # Left-aligned with "NAME" header
    x_unit_price = 180  # Right-aligned with "UNIT PRICE" header
    x_quantity = 280  # Right-aligned with "QUANTITY" header  
    x_vat = 380      # Right-aligned with "VAT" header
    x_total = 480    # Right-aligned with "TOTAL PRICE" header

    # --- Items table ---
    y_position = start_y + 20  # moved down by 20pt to lower the table values
    for item in items:
        # Item name (left-aligned)
        can.drawString(x_name, height - y_position, str(item["name"]))
        
        # Numbers (right-aligned with proper column widths to prevent overlap)
        # Unit price - right align within a 80pt width
        unit_price_text = str(item["unit_price"])
        unit_price_width = can.stringWidth(unit_price_text, "Helvetica", 12)
        unit_price_x = x_unit_price + 80 - unit_price_width
        can.drawString(unit_price_x, height - y_position, unit_price_text)
        
        # Quantity - left align (changed from right align)
        quantity_text = str(item["quantity"])
        can.drawString(x_quantity, height - y_position, quantity_text)
        
        # VAT - right align within a 80pt width
        vat_text = str(item["vat"])
        vat_width = can.stringWidth(vat_text, "Helvetica", 12)
        vat_x = x_vat + 80 - vat_width
        can.drawString(vat_x, height - y_position, vat_text)
        
        # Total price - right align within a 100pt width
        total_text = str(item["total_price"])
        total_width = can.stringWidth(total_text, "Helvetica", 12)
        total_x = x_total + 100 - total_width
        can.drawString(total_x, height - y_position, total_text)
        
        y_position += row_height

    can.save()

    packet.seek(0)
    overlay_pdf = PdfReader(packet)
    output = PdfWriter()
    page.merge_page(overlay_pdf.pages[0])
    output.add_page(page)

    # Ensure output directory exists
    import os
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "wb") as f:
        output.write(f)

def render_quote_generation_ui(user_id):
    st.title("📝 Quote Generation")
    st.write("Generate a quote for a customer, including items you specify and deals from their interaction history.")



    # 1. Select customer
    customers_dict = get_all_customer_names()
    if not customers_dict:
        st.warning("No customers found. Please create a customer first.")
        return
    sorted_customer_names = sorted(customers_dict.keys())
    selected_customer_name = st.selectbox("Select Customer", options=sorted_customer_names, key="quote_customer_select")
    customer_id = customers_dict[selected_customer_name]

    # 2. Add items for the quote
    st.subheader("Add Items to Quote")
    if 'quote_items' not in st.session_state:
        st.session_state['quote_items'] = []
    with st.form(key="add_quote_item_form"):
        item_name = st.text_input("Item Name", key="quote_item_name")
        item_qty = st.number_input("Quantity", min_value=1, value=1, key="quote_item_qty")
        item_price = st.number_input("Unit Price", min_value=0.0, value=0.0, format="%.2f", key="quote_item_price")
        add_item = st.form_submit_button("Add Item")
        if add_item and item_name and item_price > 0:
            st.session_state['quote_items'].append({
                'item': item_name,
                'qty': item_qty,
                'price': item_price
            })
            formatted_price = format_currency_with_commas(item_price)
            st.success(f"Added {item_name} (x{item_qty}) at {formatted_price} each.")
            st.rerun()
    # Show current items
    if st.session_state['quote_items']:
        st.write("**Items in Quote:**")
        for idx, item in enumerate(st.session_state['quote_items']):
            formatted_price = format_currency_with_commas(item['price'])
            st.write(f"{idx+1}. {item['item']} - Qty: {item['qty']}, Price: {formatted_price}")
        if st.button("Clear Items"):
            st.session_state['quote_items'] = []
            st.rerun()
    else:
        st.info("No items added yet.")

    # 3. Extract deals from customer interactions
    st.subheader("Include Deals from Customer Interactions")
    deals = []
    interactions = get_customer_interactions(customer_id)
    if interactions:
        for interaction in interactions:
            summary = interaction.get('llm_output_summary', '')
            for line in summary.split('\n'):
                if any(keyword in line.lower() for keyword in ['deal', 'product', 'qty', 'price']):
                    deals.append(line.strip())
    if deals:
        st.write("**Extracted Deals:**")
        for deal in deals:
            st.write(f"- {deal}")
    else:
        st.info("No deals found in customer interactions.")

    # 4. Generate PDF
    if st.button("Generate Quote PDF"):
        # Prepare data for overlay
        invoice_number = "000079"  # Example, you can make this dynamic
        invoice_date = datetime.datetime.now().strftime("%d/%m/%Y")
        customer_address = "Kadisco Asian Paints Factory\nAddis Ababa, Ethiopia"  # Example, make dynamic if needed
        items = []
        for item in st.session_state['quote_items']:
            unit_price = float(item['price'])
            qty = float(item['qty'])
            vat = round(unit_price * qty * 0.15, 2)  # 15% VAT example
            total_price = round(unit_price * qty + vat, 2)
            items.append({
                'name': item['item'],
                'unit_price': f"{format_currency_with_commas(unit_price)} ETB",
                'quantity': f"{qty} KG",
                'vat': f"{format_currency_with_commas(vat)} ETB",
                'total_price': f"{format_currency_with_commas(total_price)} ETB"
            })
        notes = "We prioritize customer satisfaction. Our team of passionate skiers and snowboarders is dedicated to delivering exceptional service and ensuring your safety and enjoyment on the slopes."
        bank_details = "Beneficiary: Alhadi Maru Import and Export\nBank: Dashen Bank\nBank Branch: Bulgaria\nBank Account: 7981270984511"
        contact_info = "+251966274550"
        import io
        # Generate the quote PDF
        safe_name = selected_customer_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        output_path = f"temp_quote_{safe_name}.pdf"
        generate_quote_with_items(
            template_path="Kadisco PI.pdf",
            output_path=output_path,
            customer_name=selected_customer_name,
            items=items
        )
        
        # Read the generated file and provide download
        with open(output_path, "rb") as f:
            pdf_data = f.read()
        
        st.download_button(
            label="Download Quote PDF",
            data=pdf_data,
            file_name=f"quote_{selected_customer_name.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )

# --- Main CRM Dashboard with Tabs ---
def main_crm_dashboard(user_id, default_tab=None):
    st.title("CRM Dashboard")
    tab_titles = ["Create Customer", "Choose Existing", "Analysis & Chat", "Quote Generation" ]  # Removed RAG Test
    default_tab_index = 0 # Default to Create Customer
    if default_tab == 'manage':
        default_tab_index = 1
    elif default_tab == 'analysis':
        default_tab_index = 2
    elif default_tab == 'quote':
        default_tab_index = 3
    tabs = st.tabs(tab_titles)
    with tabs[default_tab_index]:
        if default_tab == 'create' or default_tab is None:
            render_customer_creation_ui_tab(user_id)
        elif default_tab == 'manage':
            st.write("Rendering Manage Existing Customers section...")
            render_choose_existing_ui(user_id)
        elif default_tab == 'analysis':
            st.write("Rendering Report, Analysis & Notification section...")
            render_analysis_ui(user_id)
        elif default_tab == 'quote':
            render_quote_generation_ui(user_id)

# --- Main Streamlit logic below ---
if st.session_state.get("logout_requested", False):
    st.session_state.logout_requested = False
    st.rerun()

# Initialize crm_view state if it doesn't exist
if 'crm_view' not in st.session_state:
    st.session_state.crm_view = None

# Sidebar: Only login/logout/profile
with st.sidebar:
    st.sidebar.title(" AI Powered CRM Chat ")
    # Add the logo here with error handling
    try:
        st.image("leanchems logo.png", width=150)
    except:
        st.markdown("### 🧠 AI Powered CRM")
    if not st.session_state.authenticated:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            st.subheader("Login")
            login_email = st.text_input("Email", key="sidebar_login_email")
            login_password = st.text_input("Password", type="password", key="sidebar_login_password")
            login_button = st.button("Login", key="sidebar_login_button", type="primary")
            if login_button:
                if login_email and login_password:
                    sign_in(login_email, login_password)
                else:
                    st.warning("Please enter both email and password.")
        with tab2:
            st.subheader("Sign Up")
            signup_name = st.text_input("Full Name", key="sidebar_signup_name")
            signup_email = st.text_input("Email", key="sidebar_signup_email")
            signup_password = st.text_input("Password", type="password", key="sidebar_signup_password")
            signup_button = st.button("Sign Up", key="sidebar_signup_button", type="primary")
            if signup_button:
                if signup_email and signup_password and signup_name:
                    response = sign_up(signup_email, signup_password, signup_name)
                    if response and response.user:
                        st.success("Sign up successful! Please check your email to confirm your account.")
                    else:
                        st.error("Sign up failed. Please try again.")
                else:
                    st.warning("Please fill in all fields.")
    else:
        user = st.session_state.user
        if user:
            # Get the user's full name from metadata, default to 'User' if not found
            full_name = user.user_metadata.get('full_name', 'User')
            # Display welcome message, centered and bold
            st.markdown(f"<div style='text-align: center;'><strong>Welcome, {full_name}!</strong></div>", unsafe_allow_html=True)
            st.button("Logout", on_click=sign_out, key="sidebar_logout_button", type="primary")

# Main tabbed dashboard for authenticated users
if st.session_state.authenticated and st.session_state.user:
    user_id = st.session_state.user.id
    
    if st.session_state.crm_view is None:
        st.title("CRM Dashboard")
        st.write("Select an action to get started.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Create New Customer", key="btn_create_customer", use_container_width=True, type="primary"):
                st.session_state.crm_view = 'create'
                st.rerun()
        with col2:
            if st.button("Manage Existing Customers", key="btn_manage_customer", use_container_width=True, type="primary"):
                st.session_state.crm_view = 'manage'
                st.rerun()
        with col3:
            if st.button("Report, Analysis & Notification", key="btn_analyze_crm", use_container_width=True, type="primary"):
                st.session_state.crm_view = 'analysis'
                st.rerun()
        with col4:
            if st.button("Quote Generation", key="btn_quote_generation", use_container_width=True, type="primary"):
                st.session_state.crm_view = 'quote'
                st.rerun()

    else:
        # Render the content of the selected view without tabs
        st.title("CRM Dashboard") # Keep the main title
        user_id = st.session_state.user.id # Ensure user_id is available

        # Add a button to go back to the initial selection view
        st.markdown("---") # Add a separator
       
        if st.button("Back to Main Menu", key="btn_back_to_menu"):
            st.session_state.crm_view = None
            st.rerun()

        # Now render the section-specific content
        if st.session_state.crm_view == 'create':
            render_customer_creation_ui_tab(user_id)
        elif st.session_state.crm_view == 'manage':
            
            render_choose_existing_ui(user_id)
        elif st.session_state.crm_view == 'analysis':
           
            render_analysis_ui(user_id)
        elif st.session_state.crm_view == 'quote':
            render_quote_generation_ui(user_id)
        

else:
    # Apply custom styling
    st.title("Welcome to LeanChems AI CRM Chat")
    st.write("Login or sign up to unlock powerful AI features for managing customer interactions, gaining insights, and driving growth.")
   
    # Feature highlights
    st.subheader("Features")
    col1, col2, col3 = st.columns(3)

    # Define Lottie animation URLs
    lottie_profiling = load_lottieurl("https://lottie.host/732c38ca-e084-4a94-a8c7-c2c1324b9700/WzO5X7h1a4.json") # Example URL, replace with actual
    lottie_relationships = load_lottieurl("https://lottie.host/embed/59a97f54-3a5c-494d-9d0f-b5504f6b308e/H6m7G94hUj.json") # Example URL, replace with actual
    lottie_reporting = load_lottieurl("https://lottie.host/8616b5b7-31f9-44c2-a110-76692c499332/oXqA8f8z07.json") # Example URL, replace with actual

    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if lottie_profiling:
            st_lottie(lottie_profiling, height=150, key="profiling_animation")
        st.markdown("#### ✨ Ideal Customer Profiling")
        st.write("Leverage AI to build detailed profiles and understand your most valuable customers.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if lottie_relationships:
            st_lottie(lottie_relationships, height=150, key="relationships_animation")
        st.markdown("#### 🤝 Develop & Manage Relationships")
        st.write("Track interactions, gain conversation insights, and nurture customer relationships effectively.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if lottie_reporting:
            st_lottie(lottie_reporting, height=150, key="reporting_animation")
        st.markdown("#### 📈 Advanced Reporting & Alerts")
        st.write("Get data-driven insights and receive timely alerts to stay ahead of opportunities.")
        st.markdown('</div>', unsafe_allow_html=True)

def upload_pdf_to_documents(pdf_path: str, user_id: str = "default_user"):
    # 1. Read PDF content
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # 2. Get embedding for the content (using OpenAI)
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    
    # 3. Store in Supabase
    data = {
        "content": text,
        "embedding": embedding,
        "metadata": {
            "filename": Path(pdf_path).name,
            "user_id": user_id,
            "source": "product_pdf"
        }
    }
    supabase_client.table("documents").insert(data).execute()
    print(f"Uploaded {pdf_path} to documents table.")

def upload_test_conversation(content: str, user_id: str = "default_user"):
    """Upload a test conversation to the conversation table for RAG testing"""
    try:
        # Generate embedding for the content
        embedding = gemini_embed(content)
        
        # Store in Supabase conversation table
        data = {
            "content": content,
            "embedding": embedding,
            "metadata": {
                "user_id": user_id,
                "source": "test_upload",
                "type": "conversation"
            }
        }
        
        response = supabase_client.table("conversation").insert(data).execute()
        
        if response.data:
            st.success(f"Successfully uploaded test conversation with ID: {response.data[0]['id']}")
            return response.data[0]
        else:
            st.error("Failed to upload test conversation")
            return None
            
    except Exception as e:
        st.error(f"Error uploading test conversation: {str(e)}")
        return None

def answer_any_query_with_rag(user_query, customer_id, user_id, top_k=3):
    relevant_interactions = retrieve_relevant_interactions(customer_id, user_query, top_k=top_k)
    context = ""
    if relevant_interactions:
        context = "Relevant Past Interactions:\n"
        for interaction in relevant_interactions:
            context += f"- User: {interaction['input']}\n- AI: {interaction['output']}\n"
    else:
        context = "No relevant past interactions found."
    system_prompt = f"""You are a helpful CRM assistant. Use the relevant past interactions below to answer the user's question.\n\n{context}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    return gemini_chat(messages)

# Initialize notification scheduler
try:
    start_notification_scheduler()
except Exception as e:
    print(f"Failed to start notification scheduler: {str(e)}")

# Update the main execution block to use the new sidebar
if __name__ == "__main__":
    # This section won't run in Streamlit
    pass

# --- TEST: Simulate daily notification logic ---
def test_daily_notification_simulation():
    """Simulate daily notification logic for empty and non-empty customer lists."""
    import types

    # Backup original get_all_customer_data
    orig_get_all_customer_data = globals().get('get_all_customer_data')

    # Case 1: No customers
    def fake_get_all_customer_data_empty():
        print("[TEST] get_all_customer_data() returns [] (no customers)")
        return []
    globals()['get_all_customer_data'] = fake_get_all_customer_data_empty
    print("\n[TEST] --- Simulating daily notification with NO customers ---")
    send_daily_notification()

    # Case 2: Some customers with deals
    def fake_get_all_customer_data_some():
        print("[TEST] get_all_customer_data() returns sample customers")
        return [
            {
                'customer_name': 'Sika Abysinia Ethiopia',
                'interaction_metadata': [
                    {'output': 'CURRENT DEALS:\n| Deal_ID | Product | Qty | Price | Stage | Progress | Last_Update |\n|---|---|---|---|---|---|---|\n| D-001 | RDP | N/A | N/A | Open | RFQ received | 2025-10-29 |\nCLOSED DEALS:\n| Deal_ID | Product | Outcome | Qty | Price | Progress |\n|---|---|---|---|---|---|\n| D-002 | HPMC | Won | 1000 kg | $5000 | Sample delivered |'}
                ]
            },
            {
                'customer_name': 'SOLVOCHEM',
                'interaction_metadata': [
                    {'output': 'CURRENT DEALS:\n| Deal_ID | Product | Qty | Price | Stage | Progress | Last_Update |\n|---|---|---|---|---|---|---|\n| D-003 | THINNER MIX | 500 kg | $2000 | InProcess | Quote sent | 2025-10-29 |'}
                ]
            }
        ]
    globals()['get_all_customer_data'] = fake_get_all_customer_data_some
    print("\n[TEST] --- Simulating daily notification with SOME customers ---")
    send_daily_notification()

    # Restore original
    if orig_get_all_customer_data:
        globals()['get_all_customer_data'] = orig_get_all_customer_data
    print("\n[TEST] --- Simulation complete ---\n")