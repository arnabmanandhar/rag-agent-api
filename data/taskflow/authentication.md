# TaskFlow API Authentication

## Access tokens
TaskFlow uses bearer tokens for API requests. Create a token in the workspace console and send it in the `Authorization: Bearer <token>` header. Tokens belong to one workspace and inherit the permissions assigned when created.

## Token rotation
Tokens can be rotated from the console. The replacement token is shown once. During rotation, both old and new tokens work for a ten-minute overlap; revoke the old token after updating integrations. Store tokens in a secret manager, never in source control.

## Request signing for callbacks
Webhook deliveries are authenticated separately with an HMAC-SHA256 signature. See the webhook guide for verification details; an API bearer token is not used to authenticate callback requests.
