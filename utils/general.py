def init_seed(seed: int):
    pass


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        return False
