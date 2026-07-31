from langchain_core.tools import tool
import re
import json

@tool
def parse_financial_document(text: str) -> str:
    """
    Parses financial text to extract key metrics like stock prices, quantities, and SIP details.
    Returns a JSON string of the extracted data.
    """
    # Regex patterns for common financial terms
    data = {
        "stock_transactions": [],
        "sip_details": {}
    }
    
    # Robust regex extraction for POC
    # 1. Stocks: Works for "500 shares of Reliance at 2200 and sold at 2650", "bought Reliance 500 at 2200, sold for 2650", etc.
    
    # Look for quantity + shares + name + at + buy_price (+ sold + [optional word] + at/for + sell_price)
    # Pattern: [qty] shares of [name] at [price] ... sold [optional] at [price]
    stock_pattern = r"(\d+)\s+shares\s+of\s+([\w\s]+?)\s+at\s+([\d,]+(?:\.\d+)?)(?:.*?sold\s+(?:[\w\s]+?\s+)?(?:at|for)\s+([\d,]+(?:\.\d+)?))?"
    
    stock_matches = re.finditer(stock_pattern, text, re.IGNORECASE)
    for m in stock_matches:
        data["stock_transactions"].append({
            "quantity": int(m.group(1).replace(",", "")),
            "stock_name": m.group(2).strip(),
            "buy_price": float(m.group(3).replace(",", "")),
            "sell_price": float(m.group(4).replace(",", "")) if m.group(4) else None
        })
        
    # 2. SIP: "SIP of 20,000 for 10 years at a 12%"
    sip_match = re.search(r"SIP\s+of\s+([\d,]+(?:\.\d+)?)\s+for\s+(\d+)\s+years\s+at\s+(?:a\s+)?([\d,]+(?:\.\d+)?)%?", text, re.IGNORECASE)
    if sip_match:
        data["sip_details"] = {
            "monthly_amount": float(sip_match.group(1).replace(",", "")),
            "duration_years": int(sip_match.group(2)),
            "expected_return": float(sip_match.group(3).replace(",", ""))
        }

    return json.dumps({"extracted_data": data, "status": "success"})

@tool
def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
    """Calculates the Compound Annual Growth Rate (CAGR)."""
    if years <= 0 or start_value <= 0: return 0.0
    return ((end_value / start_value) ** (1 / years)) - 1

@tool
def calculate_sip_future_value(monthly_investment: float, annual_return_rate: float, years: int) -> float:
    """Calculates the Future Value of a Mutual Fund SIP."""
    i = (annual_return_rate / 100) / 12
    n = years * 12
    # Formula: P * [((1 + i)^n - 1) / i] * (1 + i)
    return monthly_investment * ((((1 + i) ** n) - 1) / i) * (1 + i)

@tool
def calculate_stock_returns(buy_price: float, current_price: float, quantity: int) -> dict:
    """Calculates absolute returns and percentage gain for a stock investment."""
    investment = buy_price * quantity
    current_value = current_price * quantity
    absolute_return = current_value - investment
    percentage_gain = (absolute_return / investment) * 100 if investment > 0 else 0
    return {
        "investment": investment,
        "current_value": current_value,
        "absolute_return": absolute_return,
        "percentage_gain": percentage_gain
    }

@tool
def get_mutual_fund_info(fund_name: str) -> str:
    """Mock tool to simulate fetching mutual fund data like expense ratio and 1Y/3Y returns."""
    return f"Data for {fund_name}: 3Y Return: 15.4%, Expense Ratio: 0.75%, Category: Equity Large Cap"

@tool
def search_knowledge_base(query: str, agent_id: str = "default") -> str:
    """
    Searches the agent's attached knowledge base for relevant information.
    Use this to find specific product rules, formulas, or guidelines not provided in the prompt.
    """
    from utils.knowledge import get_agent_kb
    kb = get_agent_kb(agent_id)
    if not kb:
        return "No knowledge base attached to this agent."
    
    return kb.search(query)
