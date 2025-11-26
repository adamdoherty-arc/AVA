# UI Style Guide - Magnus Trading Dashboard

**Last Updated:** November 22, 2025
**Status:** ✅ **ACTIVE - ALL DEVELOPERS MUST FOLLOW**

---

## 🚫 **CRITICAL RULES - NO EXCEPTIONS**

### Rule #1: NO Horizontal Lines
**NEVER use horizontal dividers in any Streamlit page**

❌ **FORBIDDEN:**
```python
st.markdown("---")
st.divider()
```

✅ **ALLOWED:**
- Use whitespace (empty `st.write()` or just spacing)
- Use section headers with emojis
- Use expanders for logical grouping
- Use columns for visual separation

**Reasoning:**
- Horizontal lines create visual clutter
- They break the flow of the dashboard
- Modern UI design uses whitespace instead
- User preference confirmed multiple times

**Files Cleaned:** All 20+ page files as of Nov 22, 2025

---

## 📐 **Layout Guidelines**

### Spacing & Whitespace
```python
# Good spacing
st.title("💼 Page Title")
st.caption("Brief description")

# First section
st.markdown("### 🎯 Section One")
# content here

# Second section (just use spacing, no divider!)
st.markdown("### 📊 Section Two")
# content here
```

### Section Headers
```python
# Always use emojis for visual hierarchy
st.markdown("### 🎯 Main Section")
st.markdown("#### 📊 Subsection")
```

### Expanders for Grouping
```python
# Use expanders instead of dividers to separate content
with st.expander("📊 Advanced Options", expanded=False):
    # content
    pass

with st.expander("🔧 Settings", expanded=False):
    # content
    pass
```

---

## 🎨 **Color Scheme**

### Profit/Loss Colors
```python
# Text colors for P/L (NEVER use background colors)
PROFIT_COLOR = "#00AA00"  # Green text
LOSS_COLOR = "#DD0000"    # Red text

# Usage in dataframes
def highlight_pl(row):
    if pl_val > 0:
        return 'color: #00AA00; font-weight: bold'
    elif pl_val < 0:
        return 'color: #DD0000; font-weight: bold'
```

### Status Indicators
```python
# Use emoji + text for status
st.success("✅ Success message")
st.error("❌ Error message")
st.warning("⚠️ Warning message")
st.info("ℹ️ Info message")
```

---

## 📊 **Data Display**

### Tables
```python
# Always use styled dataframes for financial data
styled_df = df.style.apply(highlight_pl, axis=1)
st.dataframe(styled_df, hide_index=True, use_container_width=True)
```

### Metrics
```python
# Use columns for metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Label", "$1,234.56", delta="+5%")
```

---

## 🔄 **Refresh & Auto-Update**

### Auto-Refresh Controls (Standard Pattern)
```python
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    auto_refresh = st.checkbox("🔄 Auto-Refresh", value=False)
with col2:
    refresh_freq = st.selectbox(
        "Frequency",
        ["30s", "1m", "2m", "5m"],
        index=1,
        label_visibility="collapsed"
    )
with col3:
    if st.button("🔄 Refresh Now", type="primary"):
        st.rerun()
```

---

## 📱 **Responsive Design**

### Column Ratios
```python
# For different screen sections
col1, col2 = st.columns([2, 1])  # 2:1 ratio
col1, col2, col3 = st.columns([1, 1, 1])  # Equal
col1, col2 = st.columns([3, 1])  # Wide left, narrow right
```

---

## 🚀 **Performance Rules**

### Rate Limiting
```python
# ALWAYS use rate-limited wrappers for Robinhood API
@rate_limit("robinhood", tokens=1, timeout=30)
def get_data_rate_limited():
    return rh.some_api_call()
```

### Database Caching
```python
# Prefer database queries over repeated API calls
# Use TradeHistorySyncService for historical data
```

---

## 📝 **Code Comments**

### Section Markers
```python
# === MAIN SECTION ===
# Use triple equals for major sections

# === SUBSECTION ===
# Also use for important subsections

# Regular comments for implementation details
```

---

## ✅ **Pre-Commit Checklist**

Before committing any UI changes:

- [ ] NO `st.markdown("---")` anywhere
- [ ] NO `st.divider()` anywhere
- [ ] Proper emoji usage in headers
- [ ] Consistent color scheme for P/L
- [ ] Rate limiting on all API calls
- [ ] Responsive column layouts
- [ ] Proper error handling
- [ ] User-friendly messages

---

## 🛠️ **Enforcement**

**Automated Check:**
```bash
# Run this before committing
grep -r 'st\.markdown("---")' *.py && echo "❌ FAIL: Horizontal lines found!" || echo "✅ PASS: No horizontal lines"
```

**Code Review:**
- All PRs must be checked for horizontal lines
- Violations must be fixed before merge

---

## 📚 **Examples**

### Good Page Structure
```python
import streamlit as st

st.title("💼 Page Title")
st.caption("Brief description")

# Controls
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    option = st.selectbox("Option", ["A", "B"])

# Main content section
st.markdown("### 🎯 Main Content")
# content here

# Secondary section (NO DIVIDER!)
st.markdown("### 📊 Details")
with st.expander("📈 Advanced", expanded=False):
    # detailed content
    pass
```

---

**Questions?** Contact the development team or refer to this guide.

**Violations?** Fix immediately and update this guide if needed.
