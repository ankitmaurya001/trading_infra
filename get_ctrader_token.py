#!/usr/bin/env python3
"""
Helper script to get cTrader Access Token and Account ID
This script walks you through the OAuth 2.0 flow to get your access token.
"""

import os
import sys
import requests
import webbrowser
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

# Import from config
import config as cfg

class TokenHandler(BaseHTTPRequestHandler):
    """HTTP handler to catch the OAuth redirect"""
    
    def do_GET(self):
        """Handle the OAuth callback"""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        if 'code' in params:
            self.server.auth_code = params['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body>
                    <h1>Authorization Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """)
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body>
                    <h1>Authorization Failed</h1>
                    <p>No authorization code received. Please try again.</p>
                </body>
                </html>
            """)
    
    def log_message(self, format, *args):
        """Suppress server logs"""
        pass

def get_access_token(client_id, client_secret, redirect_uri='http://localhost:8080'):
    """
    Get access token using OAuth 2.0 authorization code flow
    
    Steps:
    1. Open browser to authorization URL
    2. User authorizes and gets redirected with code
    3. Exchange code for access token
    """
    print("=" * 70)
    print("cTrader OAuth 2.0 Token Generator")
    print("=" * 70)
    
    # Step 1: Build authorization URL
    auth_url = (
        f"https://id.ctrader.com/my/settings/openapi/grantingaccess/"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope=trading"
        f"&product=web"
    )
    
    print(f"\n📋 Configuration:")
    print(f"   Client ID: {client_id[:20]}...")
    print(f"   Redirect URI: {redirect_uri}")
    
    print(f"\n🔐 Step 1: Opening browser for authorization...")
    print(f"   URL: {auth_url}")
    print(f"\n   Please:")
    print(f"   1. Log in to your cTrader account in the browser")
    print(f"   2. Authorize the application")
    print(f"   3. You'll be redirected back automatically")
    
    # Start local HTTP server to catch redirect
    server = HTTPServer(('localhost', 8080), TokenHandler)
    server.auth_code = None
    
    # Start server in a thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Open browser
    webbrowser.open(auth_url)
    
    # Wait for authorization code (max 5 minutes)
    print(f"\n⏳ Waiting for authorization... (this may take a few minutes)")
    timeout = 300  # 5 minutes
    start_time = time.time()
    
    while server.auth_code is None:
        if time.time() - start_time > timeout:
            print("\n❌ Timeout: No authorization code received")
            server.shutdown()
            return None, None
        time.sleep(1)
    
    auth_code = server.auth_code
    server.shutdown()
    
    print(f"\n✅ Authorization code received!")
    print(f"   Code: {auth_code[:20]}...")
    
    # Step 2: Exchange code for access token
    print(f"\n🔄 Step 2: Exchanging authorization code for access token...")
    
    token_url = "https://openapi.ctrader.com/apps/token"
    token_params = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri
    }
    
    try:
        response = requests.get(token_url, params=token_params)
        response.raise_for_status()
        
        token_data = response.json()
        
        access_token = token_data.get('accessToken')
        refresh_token = token_data.get('refreshToken')
        expires_in = token_data.get('expiresIn', 3600)
        
        if not access_token:
            print(f"\n❌ Error: No access token in response")
            print(f"   Response: {token_data}")
            return None, None
        
        print(f"\n✅ Access token obtained!")
        print(f"   Access Token: {access_token[:30]}...")
        print(f"   Refresh Token: {refresh_token[:30] if refresh_token else 'N/A'}...")
        print(f"   Expires in: {expires_in} seconds ({expires_in/3600:.1f} hours)")
        
        return access_token, refresh_token
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error exchanging code for token: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return None, None

def get_account_list(access_token):
    """
    Get list of accounts associated with the access token
    
    Returns:
        list: List of account dictionaries with ctidTraderAccountId
    """
    print(f"\n📋 Step 3: Fetching account list...")
    
    # We need to use cTrader Open API SDK for this
    try:
        from ctrader_open_api import Client as CTraderClient, TcpProtocol, EndPoints
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAApplicationAuthReq,
            ProtoOAGetAccountListByAccessTokenReq,
            ProtoOAAccountListRes
        )
        from twisted.internet import reactor
        import threading
        import time
    except ImportError:
        print("\n❌ cTrader Open API SDK not installed!")
        print("   Install with: pip install ctrader-open-api")
        return []
    
    accounts = []
    error_occurred = False
    
    def on_connect(result):
        """Handle connection"""
        try:
            # Authenticate application (we need client_id and secret for this)
            app_auth = ProtoOAApplicationAuthReq()
            app_auth.clientId = cfg.CTRADER_CLIENT_ID
            app_auth.clientSecret = cfg.CTRADER_CLIENT_SECRET
            client.sendMessage(app_auth)
            
            # Get account list
            acc_list_req = ProtoOAGetAccountListByAccessTokenReq()
            acc_list_req.accessToken = access_token
            client.sendMessage(acc_list_req)
        except Exception as e:
            nonlocal error_occurred
            error_occurred = True
            print(f"   Error in connection: {e}")
    
    def on_account_list(message):
        """Handle account list response"""
        nonlocal accounts
        try:
            if hasattr(message, 'ctidTraderAccount'):
                for acc in message.ctidTraderAccount:
                    accounts.append({
                        'account_id': acc.ctidTraderAccountId,
                        'broker_name': acc.brokerName,
                        'account_type': acc.accountType,
                        'deposit_asset_id': acc.depositAssetId,
                        'is_live': acc.accountType == 1  # 1 = LIVE, 0 = DEMO
                    })
        except Exception as e:
            print(f"   Error processing account list: {e}")
        finally:
            client.disconnect()
            try:
                if reactor.running:
                    reactor.stop()
            except:
                pass
    
    def on_error(error):
        """Handle errors"""
        nonlocal error_occurred
        error_occurred = True
        print(f"   Connection error: {error}")
        client.disconnect()
        try:
            if reactor.running:
                reactor.stop()
        except:
            pass
    
    # Create client (try demo first, then live)
    for demo_mode in [True, False]:
        host = EndPoints.PROTOBUF_DEMO_HOST if demo_mode else EndPoints.PROTOBUF_LIVE_HOST
        port = EndPoints.PROTOBUF_PORT
        
        client = CTraderClient(host, port, TcpProtocol)
        client.setCallback(on_account_list)
        client.setErrorCallback(on_error)
        
        # Run reactor in thread
        reactor_thread = threading.Thread(target=reactor.run, kwargs={'installSignalHandlers': 0}, daemon=True)
        reactor_thread.start()
        time.sleep(0.5)
        
        # Connect
        d = client.connect()
        d.addCallback(on_connect)
        
        # Wait for response
        reactor_thread.join(timeout=10)
        
        try:
            if reactor.running:
                reactor.stop()
        except:
            pass
        
        if accounts or error_occurred:
            break
    
    return accounts

def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("cTrader Access Token & Account ID Generator")
    print("=" * 70)
    
    # Get credentials from config
    client_id = cfg.CTRADER_CLIENT_ID
    client_secret = cfg.CTRADER_CLIENT_SECRET
    
    if not client_id or not client_secret:
        print("\n❌ Missing CLIENT_ID or CLIENT_SECRET in config.py")
        print("   Please add them to config.py first")
        return
    
    # Get access token
    access_token, refresh_token = get_access_token(client_id, client_secret)
    
    if not access_token:
        print("\n❌ Failed to get access token")
        return
    
    # Get account list
    accounts = get_account_list(access_token)
    
    if not accounts:
        print("\n⚠️  Could not fetch account list automatically")
        print("\n📝 Manual steps:")
        print("   1. After authorization, check your cTrader account")
        print("   2. Find your Account ID (ctidTraderAccountId)")
        print("   3. Add it to config.py as CTRADER_ACCOUNT_ID")
    else:
        print(f"\n✅ Found {len(accounts)} account(s):")
        for i, acc in enumerate(accounts, 1):
            print(f"\n   Account {i}:")
            print(f"      Account ID: {acc['account_id']}")
            print(f"      Broker: {acc.get('broker_name', 'N/A')}")
            print(f"      Type: {'LIVE' if acc['is_live'] else 'DEMO'}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ Summary - Add these to config.py:")
    print("=" * 70)
    print(f"\nCTRADER_ACCESS_TOKEN = '{access_token}'")
    if refresh_token:
        print(f"CTRADER_REFRESH_TOKEN = '{refresh_token}'")
    if accounts:
        print(f"CTRADER_ACCOUNT_ID = {accounts[0]['account_id']}  # {accounts[0].get('broker_name', '')} {'LIVE' if accounts[0]['is_live'] else 'DEMO'}")
    else:
        print(f"CTRADER_ACCOUNT_ID = <YOUR_ACCOUNT_ID>  # Add manually")
    print(f"CTRADER_DEMO = {not accounts[0]['is_live'] if accounts else 'True'}  # True for demo, False for live")
    
    print("\n" + "=" * 70)
    print("💡 Note: Access tokens expire. Use refresh_token to get new tokens.")
    print("=" * 70)

if __name__ == "__main__":
    main()

