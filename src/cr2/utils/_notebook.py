def running_in_notebook() -> bool:
    """Evaluate if code is running in an IPython notebook."""
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
