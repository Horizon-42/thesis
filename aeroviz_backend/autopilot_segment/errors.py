"""What the live executor's endpoint answers when it does not fly: stdlib only, so the HTTP layer maps them to a status
without importing torch."""


class RequestRefused(ValueError):
    """The request is not one the Training view can make: a field missing or of the wrong kind, a column that is not a
    column, an airport that is not an airport code, a step that says no word of it (HTTP 400)."""


class NotListed(LookupError):
    """A set or a flight the request names that the airport's Training files do not list (HTTP 404)."""
