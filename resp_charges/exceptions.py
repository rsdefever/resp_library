class RESPLibraryError(Exception):
    """Base class for all non-trivial errors raised by `RESP Library`"""


class ChargesNotFoundError(Exception):
    """Error raised when charges are not found for a molecule"""


class SMILESConversionError(Exception):
    """Converting SMILES to other format"""
