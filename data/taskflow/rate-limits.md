# TaskFlow API Rate Limits

## Standard limits
Each workspace may make 120 requests per minute. Limits are measured over a rolling 60-second window and shared by all tokens in that workspace. A request rejected for rate limiting returns HTTP 429 and does not consume a task identifier.

## Rate-limit headers
Responses include `RateLimit-Limit`, `RateLimit-Remaining`, and `RateLimit-Reset`. Reset is a Unix timestamp in seconds. On HTTP 429, clients should wait until reset and then retry with exponential backoff and random jitter.

## Burst behavior
The service permits a short burst of up to 20 requests above the steady rate. Repeated bursts may be throttled earlier. Bulk imports should use the asynchronous batch endpoint rather than parallel single-task requests.
