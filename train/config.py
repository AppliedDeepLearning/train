import tensorflow as tf

_is_training_eager = False
_COLLECTION = 'training'


def training(*arguments, **keywords):
    fn = _training_eager if tf.executing_eagerly() else _training
    return fn(*arguments, **keywords)


def init_training(*arguments, **keywords):
    fn = _init_training_eager if tf.executing_eagerly() else _init_training
    return fn(*arguments, **keywords)


def _training_eager(value=None):
    global _is_training_eager

    if value is None:
        return _is_training_eager

    _is_training_eager = bool(value)


def _init_training_eager():
    pass


def _training(value=None, session=None):
    _init_training()

    if value is None:
        return tf.get_collection(_COLLECTION)[0]

    if session is None:
        session = tf.get_default_session()
    value = int(value) + 1
    tf.get_collection(_COLLECTION)[value].eval(session=session)


def _init_training():
    if len(tf.get_collection(_COLLECTION)) == 0:
        v = tf.Variable(False, trainable=False, name='is_training')
        tf.add_to_collection(_COLLECTION, v)
        tf.add_to_collection(_COLLECTION, tf.assign(v, False, name='set_training_false'))
        tf.add_to_collection(_COLLECTION, tf.assign(v, True, name='set_training_true'))
