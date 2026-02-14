#!/usr/bin/env python3
"""
Get the list of cTrader accounts linked to your access token.
This will show you the correct ctidTraderAccountId to use.
"""

import sys
import time
import threading

try:
    from ctrader_open_api import Client, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAErrorRes
    )
    from ctrader_open_api.protobuf import Protobuf
    from twisted.internet import reactor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install with: pip install ctrader-open-api")
    sys.exit(1)

from config import (
    CTRADER_CLIENT_ID,
    CTRADER_CLIENT_SECRET,
    CTRADER_ACCESS_TOKEN,
    CTRADER_DEMO
)

def main():
    print("=" * 60)
    print("cTrader Account Finder")
    print("=" * 60)
    
    if not CTRADER_CLIENT_ID or not CTRADER_CLIENT_SECRET:
        print("❌ Missing CTRADER_CLIENT_ID or CTRADER_CLIENT_SECRET in config.py")
        sys.exit(1)
    
    if not CTRADER_ACCESS_TOKEN:
        print("❌ Missing CTRADER_ACCESS_TOKEN in config.py")
        print("   Run get_ctrader_token.py first to get your access token")
        sys.exit(1)
    
    print(f"\n📋 Using credentials:")
    print(f"   Client ID: {CTRADER_CLIENT_ID[:10]}...")
    print(f"   Access Token: {CTRADER_ACCESS_TOKEN[:20]}...")
    print(f"   Mode: {'Demo' if CTRADER_DEMO else 'Live'}")
    
    # Select endpoint based on mode
    host = EndPoints.PROTOBUF_DEMO_HOST if CTRADER_DEMO else EndPoints.PROTOBUF_LIVE_HOST
    
    accounts_found = []
    done_event = threading.Event()
    
    def on_error(failure):
        print(f"\n❌ Error: {failure}")
        done_event.set()
        try:
            reactor.callFromThread(reactor.stop)
        except:
            pass
    
    def account_list_callback(result):
        """Handle account list response"""
        try:
            message = Protobuf.extract(result)
            
            if isinstance(message, ProtoOAErrorRes):
                error_msg = getattr(message, 'description', 'Unknown error')
                error_code = getattr(message, 'errorCode', 'Unknown')
                print(f"\n❌ Error getting accounts: Code={error_code}, Message={error_msg}")
            else:
                # Get accounts from response
                if hasattr(message, 'ctidTraderAccount'):
                    print(f"\n✅ Found {len(message.ctidTraderAccount)} account(s):\n")
                    print("-" * 60)
                    
                    for i, account in enumerate(message.ctidTraderAccount, 1):
                        account_id = account.ctidTraderAccountId
                        is_live = getattr(account, 'isLive', None)
                        broker_title = getattr(account, 'brokerTitleShort', 'Unknown')
                        deposit_currency = getattr(account, 'depositCurrency', 'Unknown')
                        
                        accounts_found.append(account_id)
                        
                        print(f"   Account {i}:")
                        print(f"   ├── ctidTraderAccountId: {account_id}  ← USE THIS IN config.py")
                        print(f"   ├── Type: {'Live' if is_live else 'Demo'}")
                        print(f"   ├── Broker: {broker_title}")
                        print(f"   └── Currency: {deposit_currency}")
                        print()
                    
                    print("-" * 60)
                    print("\n📝 Update your config.py with the correct account ID:")
                    print(f"   CTRADER_ACCOUNT_ID = {accounts_found[0]}")
                    
                    if len(accounts_found) > 1:
                        print(f"\n   (Multiple accounts found. Choose the one you want to use)")
                else:
                    print("\n⚠️  No accounts found linked to this access token")
                    print("   Make sure you authorized with the correct cTrader ID")
        except Exception as e:
            print(f"\n❌ Error processing response: {e}")
        finally:
            done_event.set()
            try:
                client.stopService()
            except:
                pass
            try:
                reactor.callFromThread(reactor.stop)
            except:
                pass
    
    def app_auth_callback(result):
        """Handle application auth response"""
        try:
            message = Protobuf.extract(result)
            
            if isinstance(message, ProtoOAErrorRes):
                error_msg = getattr(message, 'description', 'Unknown error')
                error_code = getattr(message, 'errorCode', 'Unknown')
                print(f"\n❌ Application auth error: Code={error_code}, Message={error_msg}")
                done_event.set()
                try:
                    reactor.callFromThread(reactor.stop)
                except:
                    pass
                return
            
            print("✅ Application authenticated")
            print("📥 Fetching account list...")
            
            # Get account list
            req = ProtoOAGetAccountListByAccessTokenReq()
            req.accessToken = CTRADER_ACCESS_TOKEN
            deferred = client.send(req)
            deferred.addCallbacks(account_list_callback, on_error)
        except Exception as e:
            print(f"\n❌ Error in app auth: {e}")
            done_event.set()
    
    def on_connect(client_obj):
        """Handle connection"""
        print("✅ Connected to cTrader")
        print("🔐 Authenticating application...")
        
        app_auth = ProtoOAApplicationAuthReq()
        app_auth.clientId = CTRADER_CLIENT_ID
        app_auth.clientSecret = CTRADER_CLIENT_SECRET
        deferred = client.send(app_auth)
        deferred.addCallbacks(app_auth_callback, on_error)
    
    def on_disconnect(client_obj, reason):
        """Handle disconnection"""
        if not done_event.is_set():
            print(f"\n⚠️  Disconnected: {reason}")
        done_event.set()
    
    # Create client and connect
    print(f"\n🔌 Connecting to cTrader {'Demo' if CTRADER_DEMO else 'Live'}...")
    client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
    client.setConnectedCallback(on_connect)
    client.setDisconnectedCallback(on_disconnect)
    
    # Run in thread
    def run_reactor():
        try:
            client.startService()
            reactor.run(installSignalHandlers=False)
        except Exception as e:
            print(f"Reactor error: {e}")
    
    thread = threading.Thread(target=run_reactor, daemon=True)
    thread.start()
    
    # Wait for completion
    done_event.wait(timeout=30)
    
    if not done_event.is_set():
        print("\n⏱️  Timeout waiting for response")
    
    # Cleanup
    try:
        reactor.callFromThread(reactor.stop)
    except:
        pass
    
    time.sleep(0.5)
    
    return accounts_found

if __name__ == "__main__":
    accounts = main()
    if accounts:
        print(f"\n🎯 Use one of these account IDs: {accounts}")
    else:
        print("\n❌ No accounts found")

