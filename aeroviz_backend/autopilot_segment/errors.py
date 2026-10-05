"""What the live executor's endpoint answers when it does not fly: stdlib only, so the HTTP layer maps them to a status
without importing torch."""


class RequestRefused(ValueError):
    """The request is not one the Training view can make: a field missing or of the wrong kind, a column that is not a
    column, an airport that is not an airport code, a step that says no word of it (HTTP 400)."""


class Superseded(RuntimeError):
    """A newer request from the same page came in: this one is not flown, or stops flying (HTTP 409) — the page has
    already dropped it, and the backend flies one segment at a time, so it would only hold the newer one up."""


class NotListed(LookupError):
    """A set or a flight the request names that the airport's Training files do not list (HTTP 404)."""


class ExecutorDiffers(RuntimeError):
    """The live executor does not fly a set's flight as the set was exported: the flight is farther from the artefact's
    stored flown states than the executor conformance's bound (D73, A43) — the executor code is not the code that
    exported the set (HTTP 500 with its reason: it stops a listed flight)."""
