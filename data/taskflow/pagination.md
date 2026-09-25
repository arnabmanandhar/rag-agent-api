# TaskFlow API Pagination

## Cursor parameters
List endpoints accept `limit` and `cursor`. The default limit is 25 and the maximum is 100. The response includes an `items` array and a `next_cursor` string, which is null when no further page exists.

## Continuing a listing
Pass `next_cursor` unchanged as the `cursor` parameter in the next request. Cursors are opaque, expire after one hour, and are scoped to the original endpoint and filters. If a cursor expires, restart the listing from the first page.

## Consistent traversal
Results are ordered by creation time, newest first. A listing is not a snapshot: items created while paging may appear on later pages. For a stable export, provide `created_before` with a fixed timestamp on the first request and reuse it on every page.
