"""
Data Validation Utilities for Options Analysis
Provides comprehensive validation and quality checks for stock and options data
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta


class DataValidator:
    """Validates stock and options data for quality and completeness"""
    
    def __init__(self) -> None:
        self.warnings = []
        self.errors = []
    
    def validate_stock_data(self, stock_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate stock data is reasonable and complete
        
        Args:
            stock_data: Dict with stock information
            
        Returns:
            Tuple of (is_valid, list of warnings/errors)
        """
        self.warnings = []
        self.errors = []
        
        symbol = stock_data.get('symbol', 'UNKNOWN')
        current_price = stock_data.get('current_price', 0)
        market_cap = stock_data.get('market_cap', 0)
        high_52w = stock_data.get('high_52week', 0)
        low_52w = stock_data.get('low_52week', 0)
        volume = stock_data.get('volume', 0)
        pe_ratio = stock_data.get('pe_ratio', 0)
        
        # Critical validations (errors)
        if current_price <= 0:
            self.errors.append(f"❌ {symbol}: Price is missing or invalid (${current_price})")
        
        if current_price > 10000:  # Unreasonable price
            self.errors.append(f"⚠️ {symbol}: Price seems unusually high (${current_price:.2f})")
        
        # Data completeness warnings
        if market_cap == 0:
            self.warnings.append(f"⚠️ {symbol}: Market cap not available")
        
        if high_52w == 0 or low_52w == 0:
            self.warnings.append(f"⚠️ {symbol}: 52-week high/low not available")
        
        if volume == 0:
            self.warnings.append(f"⚠️ {symbol}: Volume data not available")
        
        # Data consistency checks
        if high_52w > 0 and low_52w > 0:
            if high_52w < low_52w:
                self.errors.append(f"❌ {symbol}: 52-week high ({high_52w}) < low ({low_52w})")
            
            if current_price > high_52w * 1.1:  # 10% tolerance
                self.warnings.append(f"⚠️ {symbol}: Current price (${current_price:.2f}) exceeds 52-week high (${high_52w:.2f})")
            
            if current_price < low_52w * 0.9:  # 10% tolerance
                self.warnings.append(f"⚠️ {symbol}: Current price (${current_price:.2f}) below 52-week low (${low_52w:.2f})")
        
        # P/E ratio validation
        if pe_ratio < 0:
            self.warnings.append(f"⚠️ {symbol}: Negative P/E ratio ({pe_ratio:.2f})")
        elif pe_ratio > 1000:
            self.warnings.append(f"⚠️ {symbol}: Unusually high P/E ratio ({pe_ratio:.2f})")
        
        is_valid = len(self.errors) == 0
        all_issues = self.errors + self.warnings
        
        return is_valid, all_issues
    
    def validate_options_data(self, options_data: Dict[str, Any], stock_price: float) -> Tuple[bool, List[str]]:
        """
        Validate options data is reasonable
        
        Args:
            options_data: Dict with options information
            stock_price: Current stock price for validation
            
        Returns:
            Tuple of (is_valid, list of warnings/errors)
        """
        self.warnings = []
        self.errors = []
        
        strike_price = options_data.get('strike_price', 0)
        premium = options_data.get('premium', 0)
        dte = options_data.get('dte', 0)
        delta = options_data.get('delta', 0)
        iv = options_data.get('iv', 0)
        
        # Critical validations
        if strike_price <= 0:
            self.errors.append(f"❌ Strike price is invalid (${strike_price:.2f})")
        
        if premium <= 0:
            self.errors.append(f"❌ Premium is invalid (${premium:.2f})")
        
        if dte <= 0 or dte > 365:
            self.errors.append(f"❌ Days to expiration is invalid ({dte} days)")
        
        # Strike price validation
        if stock_price > 0:
            strike_pct = (strike_price / stock_price) * 100 if stock_price > 0 else 0
            
            # For puts, strike should be below current price
            if delta < 0:  # Put option
                if strike_price > stock_price * 1.1:  # More than 10% above
                    self.warnings.append(f"⚠️ Put strike (${strike_price:.2f}) is {strike_pct:.1f}% of current price - may be too high")
            
            # Premium validation
            premium_pct = (premium / stock_price) * 100 if stock_price > 0 else 0
            if premium_pct > 50:  # Premium > 50% of stock price
                self.warnings.append(f"⚠️ Premium (${premium:.2f}) is {premium_pct:.1f}% of stock price - unusually high")
        
        # Delta validation
        if abs(delta) > 1.0:
            self.errors.append(f"❌ Delta ({delta:.2f}) is outside valid range (-1.0 to 1.0)")
        
        # IV validation
        if iv > 5.0:  # 500% IV
            self.errors.append(f"❌ IV ({iv*100:.1f}%) is unreasonably high")
        elif iv > 2.0:  # 200% IV
            self.warnings.append(f"⚠️ IV ({iv*100:.1f}%) is very high")
        elif iv < 0.01:  # 1% IV
            self.warnings.append(f"⚠️ IV ({iv*100:.1f}%) is very low")
        
        is_valid = len(self.errors) == 0
        all_issues = self.errors + self.warnings
        
        return is_valid, all_issues
    
    def validate_data_freshness(self, last_updated: Optional[datetime], max_age_hours: int = 24) -> Tuple[bool, str]:
        """
        Validate data freshness
        
        Args:
            last_updated: Timestamp of last update
            max_age_hours: Maximum age in hours before data is considered stale
            
        Returns:
            Tuple of (is_fresh, status_message)
        """
        if last_updated is None:
            return False, "⚠️ Data freshness unknown"
        
        age = datetime.now() - last_updated
        age_hours = age.total_seconds() / 3600
        
        if age_hours > max_age_hours:
            return False, f"🔴 Data is {age_hours:.1f} hours old (stale)"
        elif age_hours > max_age_hours / 2:
            return True, f"🟡 Data is {age_hours:.1f} hours old (recent)"
        else:
            return True, f"🟢 Data is {age_hours:.1f} hours old (fresh)"
    
    def get_data_quality_score(self, stock_data: Dict[str, Any], options_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Calculate overall data quality score (0-100)
        
        Args:
            stock_data: Stock data dict
            options_data: Optional options data dict
            
        Returns:
            Quality score from 0-100
        """
        score = 100
        
        # Stock data completeness
        if stock_data.get('current_price', 0) == 0:
            score -= 30
        if stock_data.get('market_cap', 0) == 0:
            score -= 10
        if stock_data.get('high_52week', 0) == 0 or stock_data.get('low_52week', 0) == 0:
            score -= 10
        if stock_data.get('volume', 0) == 0:
            score -= 5
        
        # Options data completeness
        if options_data:
            if options_data.get('strike_price', 0) == 0:
                score -= 15
            if options_data.get('premium', 0) == 0:
                score -= 10
            if options_data.get('dte', 0) == 0:
                score -= 5
        
        return max(0, score)


def display_data_validation(stock_data: Dict[str, Any], options_data: Optional[Dict[str, Any]] = None):
    """
    Display data validation results in Streamlit
    
    Args:
        stock_data: Stock data dict
        options_data: Optional options data dict
    """
    validator = DataValidator()
    
    # Validate stock data
    stock_valid, stock_issues = validator.validate_stock_data(stock_data)
    
    # Validate options data if provided
    options_valid = True
    options_issues = []
    if options_data:
        stock_price = stock_data.get('current_price', 0)
        options_valid, options_issues = validator.validate_options_data(options_data, stock_price)
    
    # Display validation results
    all_issues = stock_issues + options_issues
    is_valid = stock_valid and options_valid
    
    if all_issues:
        with st.expander("🔍 Data Quality Check", expanded=len(all_issues) > 0):
            quality_score = validator.get_data_quality_score(stock_data, options_data)
            
            # Quality score indicator
            if quality_score >= 90:
                st.success(f"✅ Data Quality: {quality_score}/100 (Excellent)")
            elif quality_score >= 70:
                st.info(f"ℹ️ Data Quality: {quality_score}/100 (Good)")
            elif quality_score >= 50:
                st.warning(f"⚠️ Data Quality: {quality_score}/100 (Fair)")
            else:
                st.error(f"❌ Data Quality: {quality_score}/100 (Poor)")
            
            # Display issues
            if validator.errors:
                st.error("**Errors:**")
                for error in validator.errors:
                    st.write(error)
            
            if validator.warnings:
                st.warning("**Warnings:**")
                for warning in validator.warnings:
                    st.write(warning)
    else:
        st.success("✅ Data validation passed")


def create_refresh_button(key: str = "refresh_data"):
    """
    Create a refresh button for data
    
    Args:
        key: Unique key for the button
        
    Returns:
        True if button was clicked
    """
    return st.button("🔄 Refresh Data", key=key, help="Force refresh from data sources")

