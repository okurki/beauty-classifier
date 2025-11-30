#!/bin/bash

# Test script to verify feedback impact on predictions

echo "=== Testing Feedback Impact on Celebrity Matcher ==="
echo ""

# 1. Login
echo "1. Logging in..."
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"name":"admin","password":"admin"}' | jq -r '.token')

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
    echo "❌ Login failed!"
    exit 1
fi
echo "✅ Login successful"
echo ""

# 2. Create first inference (need a test image)
echo "2. Creating first inference..."
echo "⚠️  Please provide a test image path:"
read -p "Enter image path (or press Enter to skip): " IMAGE_PATH

if [ -z "$IMAGE_PATH" ]; then
    echo "Skipping test - no image provided"
    echo ""
    echo "To test manually:"
    echo "1. Upload an image: curl -X POST http://localhost:8000/ml/users/me/inference -H 'Authorization: Bearer $TOKEN' -F 'image=@your_photo.jpg'"
    echo "2. Note the inference_id and celebrity IDs"
    echo "3. Give feedback: curl -X POST http://localhost:8000/ml/users/me/feedback -H 'Authorization: Bearer $TOKEN' -H 'Content-Type: application/json' -d '{\"inference_id\": 1, \"celebrity_id\": 3, \"feedback_type\": \"like\"}'"
    echo "4. Upload the SAME image again and compare results!"
    exit 0
fi

# First inference
RESPONSE1=$(curl -s -X POST http://localhost:8000/ml/users/me/inference \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@$IMAGE_PATH")

echo "First prediction:"
echo "$RESPONSE1" | jq '.celebrities[] | {name: .name, id: .id}'
echo ""

INFERENCE_ID=$(echo "$RESPONSE1" | jq -r '.id')
CELEBRITY_ID=$(echo "$RESPONSE1" | jq -r '.celebrities[2].id')  # 3rd celebrity

if [ -z "$CELEBRITY_ID" ] || [ "$CELEBRITY_ID" = "null" ]; then
    echo "❌ Could not get celebrity ID"
    exit 1
fi

echo "3. Giving LIKE feedback to celebrity ID: $CELEBRITY_ID"
FEEDBACK=$(curl -s -X POST http://localhost:8000/ml/users/me/feedback \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"inference_id\": $INFERENCE_ID, \"celebrity_id\": $CELEBRITY_ID, \"feedback_type\": \"like\"}")

echo "✅ Feedback submitted"
echo ""

# 4. Second inference with SAME image
echo "4. Creating second inference with SAME image..."
sleep 2
RESPONSE2=$(curl -s -X POST http://localhost:8000/ml/users/me/inference \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@$IMAGE_PATH")

echo "Second prediction (after LIKE):"
echo "$RESPONSE2" | jq '.celebrities[] | {name: .name, id: .id}'
echo ""

# 5. Compare
echo "=== Comparison ==="
echo "Celebrity that received LIKE (ID: $CELEBRITY_ID):"
echo "Before:"
echo "$RESPONSE1" | jq ".celebrities[] | select(.id == $CELEBRITY_ID)"
echo "After:"
echo "$RESPONSE2" | jq ".celebrities[] | select(.id == $CELEBRITY_ID)"
echo ""

# 6. RL Stats
echo "=== RL Agent Stats ==="
curl -s -X GET http://localhost:8000/ml/rl/stats \
  -H "Authorization: Bearer $TOKEN" | jq '.'

echo ""
echo "✅ Test complete!"

