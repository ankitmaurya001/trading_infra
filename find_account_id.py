#!/usr/bin/env python3
"""
Helper script to find your cTrader Account ID
The Account ID is a NUMERIC value, not a username
"""

print("=" * 70)
print("Finding Your cTrader Account ID")
print("=" * 70)

print("\n📋 The Account ID (ctidTraderAccountId) is a NUMERIC value")
print("   Examples: 12345678, 98765432, 1234567")
print("   NOT a username like 'ankitmaurya001'")
print("\n🔍 Where to find it:")
print("\n   1. FTMO Dashboard:")
print("      - Log into your FTMO account")
print("      - Go to 'My Accounts' or 'Account Details'")
print("      - Look for 'Account Number' or 'Account ID'")
print("      - It's usually 6-8 digits")
print("\n   2. cTrader Platform:")
print("      - Open cTrader (web or desktop)")
print("      - Go to Settings → Account")
print("      - Look for 'Account ID' or 'Account Number'")
print("      - It's a numeric value")
print("\n   3. cTrader Account Statement:")
print("      - Download your account statement")
print("      - The account ID is usually at the top")
print("\n   4. FTMO Email:")
print("      - Check your FTMO welcome email")
print("      - Account details are usually included")
print("\n💡 Common formats:")
print("   - 6 digits: 123456")
print("   - 7 digits: 1234567")
print("   - 8 digits: 12345678")
print("\n❌ NOT:")
print("   - Username: 'ankitmaurya001'")
print("   - Email: 'user@example.com'")
print("   - Broker name: 'FTMO'")
print("\n" + "=" * 70)
print("Once you find the numeric Account ID, update config.py:")
print("   CTRADER_ACCOUNT_ID = 12345678  # Your numeric account ID")
print("=" * 70)

