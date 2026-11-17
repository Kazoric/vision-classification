from typing import Any, Callable, Type, Dict

SSL_METHODS: Dict[str, type] = {}

def register_ssl_method(name: str):
    def deco(cls):
        SSL_METHODS[name] = cls
        return cls
    return deco
 
 
def build_ssl_method(name: str, backbone, **params) -> Any:
    if name not in SSL_METHODS:
        raise KeyError(f"Méthode SSL '{name}' inconnue. Disponibles : {list(SSL_METHODS)}")
    return SSL_METHODS[name](backbone, **params)