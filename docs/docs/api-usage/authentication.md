# Authentication

To obtain and use API tokens:

1. Log in to the CKAN platform
2. Navigate to "Profile" > "AI/ML API Tokens"
3. Specify token name, scopes, and expiration
4. Click "Create New Token"
5. Copy the generated token (displayed only once)
6. Use the token in API requests:

```bash
curl -X POST \
  -H "Authorization: Bearer <your_token_here>" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "c1ccccc1O"}' \
  https://api.example.com/v1/ai/evaluate
```