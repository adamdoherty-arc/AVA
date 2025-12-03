"""
Notifications Router - Telegram and other notification services
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/notifications",
    tags=["notifications"]
)

# Get Telegram bot token from environment
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


class TelegramMessage(BaseModel):
    chat_id: str
    message: str


class TelegramResponse(BaseModel):
    success: bool
    message_id: int | None = None
    error: str | None = None


@router.post("/telegram", response_model=TelegramResponse)
async def send_telegram_message(request: TelegramMessage):
    """
    Send a message to a Telegram chat.

    Requires TELEGRAM_BOT_TOKEN environment variable to be set.
    Users can get their chat_id by messaging @userinfobot on Telegram.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable."
        )

    if not request.chat_id:
        raise HTTPException(status_code=400, detail="chat_id is required")

    if not request.message:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={
                "chat_id": request.chat_id,
                "text": request.message,
                "parse_mode": "HTML"
            })

            data = response.json()

            if data.get("ok"):
                return TelegramResponse(
                    success=True,
                    message_id=data.get("result", {}).get("message_id")
                )
            else:
                error = data.get("description", "Unknown error")
                logger.error(f"Telegram API error: {error}")
                return TelegramResponse(success=False, error=error)

    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
        return TelegramResponse(success=False, error=str(e))


@router.post("/telegram/alert-pick")
async def send_pick_alert(
    chat_id: str,
    sport: str,
    matchup: str,
    pick: str,
    confidence: float,
    ev: float,
    reasoning: str = ""
):
    """
    Send a formatted AI pick alert to Telegram.
    """
    emoji_map = {
        "NFL": "🏈",
        "NBA": "🏀",
        "NCAAF": "🏈",
        "NCAAB": "🏀",
        "MLB": "⚾",
        "NHL": "🏒"
    }

    emoji = emoji_map.get(sport.upper(), "🎯")

    message = f"""
{emoji} <b>AI Pick Alert!</b>

<b>Sport:</b> {sport}
<b>Game:</b> {matchup}
<b>Pick:</b> {pick}
<b>Confidence:</b> {confidence:.0f}%
<b>Expected Value:</b> +{ev:.1f}%

{f'<i>{reasoning}</i>' if reasoning else ''}

<i>Powered by AVA Sports AI</i>
"""

    return await send_telegram_message(TelegramMessage(
        chat_id=chat_id,
        message=message.strip()
    ))


@router.post("/telegram/parlay-alert")
async def send_parlay_alert(
    chat_id: str,
    legs: list[dict],
    combined_probability: float,
    payout_multiplier: float,
    recommendation: str
):
    """
    Send a parlay suggestion alert to Telegram.
    """
    legs_text = "\n".join([
        f"• {leg.get('pick', 'N/A')} ({leg.get('game', 'Unknown')})"
        for leg in legs
    ])

    message = f"""
🎰 <b>Parlay Alert!</b>

<b>Type:</b> {recommendation}
<b>Legs:</b>
{legs_text}

<b>Combined Prob:</b> {combined_probability*100:.1f}%
<b>Payout:</b> {payout_multiplier:.2f}x

<i>Powered by AVA Sports AI</i>
"""

    return await send_telegram_message(TelegramMessage(
        chat_id=chat_id,
        message=message.strip()
    ))


@router.get("/telegram/test")
async def test_telegram_connection(chat_id: str):
    """
    Test Telegram bot connection by sending a test message.
    """
    return await send_telegram_message(TelegramMessage(
        chat_id=chat_id,
        message="✅ AVA Sports Bot connected successfully!\n\nYou will receive AI pick alerts here."
    ))
