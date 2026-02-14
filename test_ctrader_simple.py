#!/usr/bin/env python3
"""Simple cTrader connection test - minimal code to debug the flow."""

import time
import threading

from ctrader_open_api import Client, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOASymbolsListReq,
    ProtoOAErrorRes
)
from ctrader_open_api.protobuf import Protobuf
from twisted.internet import reactor

from config import (
    CTRADER_CLIENT_ID,
    CTRADER_CLIENT_SECRET,
    CTRADER_ACCESS_TOKEN,
    CTRADER_ACCOUNT_ID,
    CTRADER_DEMO
)

print("=" * 60)
print("Simple cTrader Connection Test")
print("=" * 60)
print(f"Client ID: {CTRADER_CLIENT_ID[:15]}...")
print(f"Account ID: {CTRADER_ACCOUNT_ID}")
print(f"Mode: {'Demo' if CTRADER_DEMO else 'Live'}")
print("=" * 60)

# Select endpoint
host = EndPoints.PROTOBUF_DEMO_HOST if CTRADER_DEMO else EndPoints.PROTOBUF_LIVE_HOST
print(f"Host: {host}")

done = threading.Event()
symbols_found = []

def on_error(failure):
    print(f"❌ Error: {failure}")
    done.set()
    try:
        reactor.callFromThread(reactor.stop)
    except:
        pass

def on_symbols(result):
    print("📥 Got symbol list response")
    try:
        message = Protobuf.extract(result)
        if isinstance(message, ProtoOAErrorRes):
            print(f"❌ Symbol list error: {message.errorCode} - {message.description}")
        else:
            print(f"✅ Found {len(message.symbol)} symbols")
            for sym in message.symbol[:10]:
                print(f"   - {sym.symbolName} (ID: {sym.symbolId})")
                symbols_found.append(sym.symbolName)
    except Exception as e:
        print(f"❌ Error parsing symbols: {e}")
    finally:
        done.set()
        try:
            client.stopService()
        except:
            pass
        try:
            reactor.callFromThread(reactor.stop)
        except:
            pass

def on_account_auth(result):
    print("📥 Got account auth response")
    try:
        message = Protobuf.extract(result)
        if isinstance(message, ProtoOAErrorRes):
            print(f"❌ Account auth error: {message.errorCode} - {message.description}")
            done.set()
            try:
                reactor.callFromThread(reactor.stop)
            except:
                pass
            return
        
        print("✅ Account authenticated")
        print("📤 Requesting symbol list...")
        
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = CTRADER_ACCOUNT_ID
        req.includeArchivedSymbols = False
        d = client.send(req)
        d.addCallbacks(on_symbols, on_error)
    except Exception as e:
        print(f"❌ Error in account auth: {e}")
        done.set()

def on_app_auth(result):
    print("📥 Got app auth response")
    try:
        message = Protobuf.extract(result)
        if isinstance(message, ProtoOAErrorRes):
            print(f"❌ App auth error: {message.errorCode} - {message.description}")
            done.set()
            try:
                reactor.callFromThread(reactor.stop)
            except:
                pass
            return
        
        print("✅ Application authenticated")
        print("📤 Authenticating account...")
        
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = CTRADER_ACCOUNT_ID
        req.accessToken = CTRADER_ACCESS_TOKEN
        d = client.send(req)
        d.addCallbacks(on_account_auth, on_error)
    except Exception as e:
        print(f"❌ Error in app auth: {e}")
        done.set()

def on_connect(client_obj):
    print("✅ Connected!")
    print("📤 Authenticating application...")
    
    req = ProtoOAApplicationAuthReq()
    req.clientId = CTRADER_CLIENT_ID
    req.clientSecret = CTRADER_CLIENT_SECRET
    d = client.send(req)
    d.addCallbacks(on_app_auth, on_error)

def on_disconnect(client_obj, reason):
    print(f"⚠️ Disconnected: {reason}")
    done.set()

# Create client
client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
client.setConnectedCallback(on_connect)
client.setDisconnectedCallback(on_disconnect)

def run():
    try:
        client.startService()
        reactor.run(installSignalHandlers=False)
    except Exception as e:
        print(f"Reactor error: {e}")

print("\n🔌 Connecting...")
thread = threading.Thread(target=run, daemon=True)
thread.start()

done.wait(timeout=30)

if symbols_found:
    print(f"\n✅ Success! Found symbols including: {symbols_found[:5]}")
else:
    print("\n❌ No symbols retrieved")

time.sleep(0.5)

