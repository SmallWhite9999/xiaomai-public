"""Python conversion toolkit. Tensor dependencies are loaded only when requested."""
__all__=('StorageBackend','StorageError')
def __getattr__(name):
    if name=='StorageBackend':
        from .backend import StorageBackend
        return StorageBackend
    if name=='StorageError':
        from .kernels import StorageError
        return StorageError
    raise AttributeError(name)
