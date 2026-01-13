# BINANCE_API_KEY = "BaDszdWXKghkBsGKIDyXT5r29ZIQDe5SVRfOMin7i8D0IaypQkaPIYVFTixupJYp"
# BINANCE_SECRET_KEY = "UKhB3R38TJjV1Tq8E6lK9jgqFerd3KgzCGwym9WpLXCpn5TfSbizAC2mha1N3emr"

#testnet
BINANCE_API_KEY = "XOUiHfWOjRjgpPk083uqYpR1kpwZe8c5IUTBYyHLTTu5CnbQpjGYARaRdHeGpCi9"
BINANCE_SECRET_KEY = "aN33R5C6BAqatoUVx13z0IEIELOFh6fIyGLFB0zCwoci9p7ypdJm5EBzlnbHZ481"


# Your Kite Connect API credentials
KITE_API_KEY = "wu80p2aelj2d73v5"
KITE_API_SECRET = "xorznn9fcocx1xww4uflrqprsorrij4t"

# Your Zerodha login credentials (for automated authentication)
KITE_USERNAME = "KMX177"  # Replace with your actual username
KITE_PASSWORD = "Krishna298!"  # Replace with your actual password
#KITE_TOTP_KEY = "N6RGEW7E5VBDTGBLFOONFU3KOQZGR27G"  # Replace with your actual TOTP key
KITE_TOTP_KEY = "2DP2PLIZCIZQTPEYSB7Z6AL7QB2YUVKX"

KITE_CREDENTIALS={"username":KITE_USERNAME, 
             "password" : KITE_PASSWORD,
            "api_key":KITE_API_KEY,
             "api_secret": KITE_API_SECRET,
            "totp_key": KITE_TOTP_KEY}

KITE_EXCHANGE = "MCX"

# ============================================================================
# MA Optimization - Neighborhood-Aware Scoring Configuration
# ============================================================================
# Neighborhood radius in normalized space (0-1 range)
# "auto": Dynamically calculate based on grid step sizes (RECOMMENDED)
#         Automatically adapts to include immediate neighbors regardless of grid spacing
# 0.1-0.15: Very tight neighborhood (few neighbors, very local)
# 0.2-0.3: Moderate neighborhood
# 0.4-0.5: Large neighborhood (many neighbors, broader region)
NEIGHBORHOOD_RADIUS = "auto"  # Recommended: auto-adapts to grid spacing

# Multiplier for auto radius calculation (how many "steps" to include as neighbors)
# 1.5: Include immediate diagonal neighbors (8 neighbors in regular grid)
# 2.0: Include next layer of neighbors (up to 24 neighbors)
# 2.5: Include wider neighborhood
NEIGHBORHOOD_RADIUS_MULTIPLIER = 1.5

# Distance weight power for inverse distance weighting
# Higher = closer neighbors get exponentially more weight
# 1.0: Linear inverse distance (moderate preference)
# 2.0: Squared inverse distance (strong preference, recommended)
# 3.0: Cubic inverse distance (very strong preference)
DISTANCE_WEIGHT_POWER = 2.0

# Score weighting for neighborhood-aware calculation
# own_score_weight: Weight for the point's own score (0-1)
# neighborhood_weight: Weight for neighborhood average score (0-1)
# negative_penalty_weight: Weight for penalty from negative scores nearby (0-1)
OWN_SCORE_WEIGHT = 0.5
NEIGHBORHOOD_WEIGHT = 0.5
NEGATIVE_PENALTY_WEIGHT = 0.1