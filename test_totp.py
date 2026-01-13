#!/usr/bin/env python3
"""
Test TOTP generation to verify TOTP key is working correctly.
This helps diagnose if TOTP is the issue with authentication.
"""

import onetimepass as otp
import time
from datetime import datetime
import config as cfg

def test_totp():
    """Test TOTP generation."""
    print("=" * 60)
    print("  TOTP TEST")
    print("=" * 60)
    
    totp_key = cfg.KITE_TOTP_KEY
    print(f"\n📋 TOTP Key: {totp_key}")
    print(f"   (First 10 chars: {totp_key[:10]}...)")
    
    # Generate current TOTP
    current_totp = otp.get_totp(totp_key)
    # Convert to string for display and validation
    current_totp_str = str(current_totp).zfill(6)  # Ensure 6 digits with leading zeros if needed
    print(f"\n🔐 Current TOTP: {current_totp_str}")
    print(f"   Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show TOTP for next few intervals
    print(f"\n📊 TOTP for next 3 intervals:")
    for i in range(3):
        # TOTP changes every 30 seconds
        # We can't easily predict future TOTPs, but we can show current
        if i == 0:
            print(f"   Now: {current_totp_str}")
        else:
            print(f"   +{i*30}s: (will change)")
    
    # Verify TOTP format
    if len(current_totp_str) == 6 and current_totp_str.isdigit():
        print(f"\n✅ TOTP format is correct (6 digits)")
    else:
        print(f"\n❌ TOTP format is incorrect!")
        print(f"   Expected: 6 digits")
        print(f"   Got: {len(current_totp_str)} chars - {current_totp_str}")
    
    # Check if TOTP key format is correct (base32)
    try:
        import base64
        # TOTP keys are base32 encoded
        # Try to decode to verify format
        base32_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
        if all(c in base32_chars for c in totp_key.upper()):
            print(f"✅ TOTP key format looks correct (base32)")
        else:
            print(f"⚠️  TOTP key format might be incorrect")
            print(f"   Should contain only: A-Z, 2-7")
    except Exception as e:
        print(f"⚠️  Could not verify TOTP key format: {e}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Verify this TOTP matches what you see in your authenticator app")
    print(f"   2. If it doesn't match, your TOTP key might be wrong or out of sync")
    print(f"   3. If account is locked, unlock it first via Kite Web")
    print(f"   4. If TOTP is wrong, regenerate it in Kite → Settings → API → TOTP")
    
    return current_totp

if __name__ == "__main__":
    test_totp()

