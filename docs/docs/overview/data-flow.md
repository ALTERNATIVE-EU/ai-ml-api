# Data Flow

![Data Flow Diagram](data_flow.png)

The data flow process involves several key steps, outlined below:

1. **Client Request**
    - The client application sends an API request to the system. This request includes an authentication token and the necessary data or parameters for the API endpoint.

2. **Istio Service Mesh**
    - The request first passes through the Istio Service Mesh.

3. **Envoy Filter Interception**
    - The custom Envoy Filter, written in Go, intercepts the request and begins processing.

4. **Public Key Retrieval**
    - The Envoy Filter fetches and caches the public key from Keycloak, which will be used to verify the authentication token.

5. **Token Verification**
    - The Envoy Filter verifies the token's signature using the fetched public key. It also checks the token's validity, expiration date, and any other claims.

6. **Role and Permission Check**
    - After verifying the token, the Envoy Filter checks the user's roles and permissions encoded within the token.

7. **Revoked Token Check**
    - The Envoy Filter fetches and caches revoked tokens from the database, then checks if the token is revoked. If revoked, it responds with an unauthorized message to the client.

8. **Forward Valid Request**
    - If the token is valid and the user has the necessary roles and permissions, the Envoy Filter forwards the request to the AI/ML API Server.

9. **ML Model Interaction**
    - The API Server interacts with the appropriate machine learning (ML) model(s) based on the functionality requested.

10. **ML Model Processing**
    - The ML model(s) process the input data and generate the requested output, such as predictions, classifications, or analyses.

11. **API Response**
    - The API Server packages the output from the ML model(s) into an appropriate response format and sends it back to the client application through the secure communication channel.