# TaskFlow API Error Handling

## Error response format
Errors use a JSON object with `error.code`, `error.message`, and `request_id`. The message is safe to show to an operator; the request ID can be given to support. Do not retry a request merely because its message contains the word “temporary.”

## Common status codes
HTTP 400 means a malformed parameter; correct the request before retrying. HTTP 401 means the bearer token is missing, invalid, or revoked. HTTP 403 means the token is valid but lacks permission. HTTP 404 means the resource does not exist or is not visible to that workspace.

## Conflict and server errors
HTTP 409 indicates a state conflict, such as updating a task with an outdated version; fetch the latest task and reconcile. HTTP 429 indicates a rate limit. HTTP 500 and 503 indicate a service-side failure and may be retried using bounded exponential backoff with jitter. Include `Idempotency-Key` for safe retries of task creation.
