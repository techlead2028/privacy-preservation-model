import pickle
def serialize_update(update):
    return pickle.dumps(update)
def deserialize_update(data):
    return pickle.loads(data)