# TaskFlow Webhooks

## Delivery events
Workspaces can subscribe to `task.created`, `task.updated`, and `task.completed`. Each event contains an event ID, event type, creation timestamp, and a task snapshot. Events are delivered as HTTPS POST requests with a JSON body.

## Signature verification
The `X-TaskFlow-Signature` header contains a timestamp and an HMAC-SHA256 digest made with the endpoint signing secret. Compute the digest over the exact request body and compare it with a constant-time comparison. Reject timestamps more than five minutes from local time to reduce replay risk.

## Retries and ordering
TaskFlow retries failed deliveries up to six times over 24 hours with increasing delays. Delivery is at least once, so consumers must deduplicate by event ID. Events for the same task can arrive out of order; use the task version field to ignore stale updates.
