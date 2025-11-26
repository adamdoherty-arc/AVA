from fastapi import APIRouter
from typing import Optional
from datetime import datetime, timedelta
import random

router = APIRouter(prefix="/api/subscriptions", tags=["subscriptions"])


@router.get("")
async def get_subscription():
    """Get current subscription - alias for /current used by SubscriptionManagement page"""
    return await get_current_subscription()


@router.get("/current")
async def get_current_subscription():
    """Get current subscription details"""
    return {
        "plan": "Professional",
        "status": "active",
        "started_at": "2024-10-01T00:00:00Z",
        "next_billing": (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d"),
        "price": 49.99,
        "billing_cycle": "monthly",
        "features": [
            "Unlimited options scans",
            "Real-time options flow",
            "AI trade recommendations",
            "Sports betting analytics",
            "Priority support",
            "API access"
        ],
        "limits": {
            "api_calls_per_day": 10000,
            "api_calls_used": 2345,
            "watchlist_symbols": 100,
            "watchlist_used": 45,
            "alerts": 50,
            "alerts_used": 12
        }
    }

@router.get("/plans")
async def get_available_plans():
    """Get available subscription plans"""
    return {
        "plans": [
            {
                "id": "free",
                "name": "Free",
                "price": 0,
                "billing_cycle": "monthly",
                "features": [
                    "5 options scans per day",
                    "Basic portfolio tracking",
                    "Community support"
                ],
                "limits": {
                    "api_calls_per_day": 100,
                    "watchlist_symbols": 10,
                    "alerts": 3
                }
            },
            {
                "id": "starter",
                "name": "Starter",
                "price": 19.99,
                "billing_cycle": "monthly",
                "features": [
                    "50 options scans per day",
                    "Basic options flow",
                    "Portfolio analytics",
                    "Email support"
                ],
                "limits": {
                    "api_calls_per_day": 1000,
                    "watchlist_symbols": 25,
                    "alerts": 10
                }
            },
            {
                "id": "professional",
                "name": "Professional",
                "price": 49.99,
                "billing_cycle": "monthly",
                "popular": True,
                "features": [
                    "Unlimited options scans",
                    "Real-time options flow",
                    "AI trade recommendations",
                    "Sports betting analytics",
                    "Priority support",
                    "API access"
                ],
                "limits": {
                    "api_calls_per_day": 10000,
                    "watchlist_symbols": 100,
                    "alerts": 50
                }
            },
            {
                "id": "enterprise",
                "name": "Enterprise",
                "price": 199.99,
                "billing_cycle": "monthly",
                "features": [
                    "Everything in Professional",
                    "Dedicated support",
                    "Custom integrations",
                    "Multi-user accounts",
                    "White-label options",
                    "SLA guarantee"
                ],
                "limits": {
                    "api_calls_per_day": 100000,
                    "watchlist_symbols": 1000,
                    "alerts": 500
                }
            }
        ]
    }

@router.get("/usage")
async def get_usage_stats():
    """Get subscription usage statistics"""
    return {
        "period": "November 2024",
        "api_calls": {
            "total": 2345,
            "limit": 10000,
            "by_endpoint": [
                {"endpoint": "/options/chain", "calls": 890},
                {"endpoint": "/scanner/scan", "calls": 567},
                {"endpoint": "/sports/games", "calls": 445},
                {"endpoint": "/chat/message", "calls": 234},
                {"endpoint": "Other", "calls": 209}
            ]
        },
        "features_used": [
            {"feature": "Options Scanner", "usage_count": 156},
            {"feature": "AI Chat", "usage_count": 89},
            {"feature": "Sports Analytics", "usage_count": 67},
            {"feature": "Alerts", "usage_count": 45}
        ],
        "daily_usage": [
            {"date": "2024-11-20", "calls": 234},
            {"date": "2024-11-21", "calls": 456},
            {"date": "2024-11-22", "calls": 567},
            {"date": "2024-11-23", "calls": 345},
            {"date": "2024-11-24", "calls": 743}
        ]
    }

@router.get("/billing-history")
async def get_billing_history():
    """Get billing history"""
    return {
        "invoices": [
            {
                "id": "inv-001",
                "date": "2024-11-01",
                "amount": 49.99,
                "status": "paid",
                "plan": "Professional",
                "payment_method": "Visa ****4242"
            },
            {
                "id": "inv-002",
                "date": "2024-10-01",
                "amount": 49.99,
                "status": "paid",
                "plan": "Professional",
                "payment_method": "Visa ****4242"
            },
            {
                "id": "inv-003",
                "date": "2024-09-01",
                "amount": 19.99,
                "status": "paid",
                "plan": "Starter",
                "payment_method": "Visa ****4242"
            }
        ]
    }

@router.post("/upgrade")
async def upgrade_plan(plan_id: str):
    """Upgrade subscription plan"""
    return {
        "status": "success",
        "message": f"Successfully upgraded to {plan_id} plan",
        "effective_date": datetime.now().isoformat()
    }

@router.post("/cancel")
async def cancel_subscription():
    """Cancel subscription"""
    return {
        "status": "success",
        "message": "Subscription cancelled",
        "effective_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
        "note": "You will retain access until the end of your billing period"
    }

@router.get("/payment-methods")
async def get_payment_methods():
    """Get saved payment methods"""
    return {
        "methods": [
            {
                "id": "pm-001",
                "type": "card",
                "brand": "Visa",
                "last4": "4242",
                "exp_month": 12,
                "exp_year": 2025,
                "is_default": True
            },
            {
                "id": "pm-002",
                "type": "card",
                "brand": "Mastercard",
                "last4": "5555",
                "exp_month": 6,
                "exp_year": 2026,
                "is_default": False
            }
        ]
    }

@router.post("/payment-methods")
async def add_payment_method(card_token: str):
    """Add a new payment method"""
    return {
        "status": "success",
        "message": "Payment method added",
        "method_id": f"pm-{random.randint(100, 999)}"
    }

@router.delete("/payment-methods/{method_id}")
async def delete_payment_method(method_id: str):
    """Delete a payment method"""
    return {
        "status": "success",
        "message": f"Payment method {method_id} deleted"
    }
