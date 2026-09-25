# TaskFlow Python SDK

## Installation and client setup
Install the fictional package `taskflow-client` from your organization's approved package registry. Create a client with `TaskFlowClient(token=...)`; pass the token from an environment variable or secret manager rather than embedding it in code.

## Creating and listing tasks
Call `client.tasks.create(title="Review release plan")` to create a task. List tasks with `client.tasks.list(limit=50)`. The returned page exposes `items` and `next_cursor`; pass the cursor to another list call to continue.

## Retries and timeouts
The client uses a 10-second connection timeout and a 30-second total request timeout by default. Configure these when constructing the client. Automatic retries are limited to connection failures and HTTP 503, with at most three attempts. The SDK does not automatically retry task creation unless an idempotency key is supplied.
