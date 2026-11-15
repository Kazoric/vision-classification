from .registry import SSL_METHODS, build_ssl_method, register_ssl_method

from . import byol
from . import fcmae
from . import simsiam

__all__ = [
    "SSL_METHODS",
    "build_ssl_method",
    "register_ssl_method",
]