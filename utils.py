class dotdict(dict):
    def __getattr__(self, name):
        if name == '__getstate__':
            return super().__getattr__(name)  # Call the superclass method
        else:
            return self[name]