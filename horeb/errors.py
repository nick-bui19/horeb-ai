class HorebError(Exception):
    pass


class InvalidReferenceError(HorebError):
    pass


class EmptyPassageError(HorebError):
    pass


class CitationOutOfRangeError(HorebError):
    pass


class AnalysisFailedError(HorebError):
    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response
