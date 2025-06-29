
"""
Custom exceptions
"""

class CryptoDataException(Exception):
    """Base exception for all custom exceptions"""
    pass

class CollectorException(CryptoDataException):
    """Collector-related exceptions"""
    pass

class ValidationException(CryptoDataException):
    """Data validation exceptions"""
    pass

class ProcessingException(CryptoDataException):
    """Data processing exceptions"""
    pass

class DatabaseException(CryptoDataException):
    """Database-related exceptions"""
    pass

class MessageBusException(CryptoDataException):
    """Message bus exceptions"""
    
