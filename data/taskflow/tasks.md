# TaskFlow Tasks

## Creating a task
Create a task with `POST /v1/tasks` and a JSON body containing a non-empty `title`. Optional fields include `description`, `assignee_id`, and `due_at` in ISO 8601 UTC format. The response returns the task with its server-generated ID and version 1.

## Updating a task
Update a task with `PATCH /v1/tasks/{task_id}`. Include the current `version` value. A successful update increments the version. If another client changed the task first, the API returns HTTP 409 and the client should fetch the latest representation before applying its change.

## Idempotent creation
For task creation that may be retried, send a unique `Idempotency-Key` header. TaskFlow stores the result for 24 hours. Reusing a key with the same request returns the original result; using it with a different request returns HTTP 409.
