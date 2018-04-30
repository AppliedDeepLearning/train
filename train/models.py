import tensorflow as tf
import estimator as E

from .config import training, init_training


class TrainingHook(tf.train.SessionRunHook):

    def begin(self):
        init_training()

    def after_create_session(self, session, _):
        training(True, session=session)

    def end(self, session):
        training(False, session=session)


class NotTrainingHook(TrainingHook):

    def after_create_session(self, session, _):
        training(False, session=session)


def get_hooks():
    return {
        E.TRAIN: [TrainingHook()],
        E.EVAL: [NotTrainingHook()],
        E.PREDICT: [NotTrainingHook()],
    }


class Estimator(E.Estimator):

    def _get_hooks(self):
        return get_hooks()


class Model(E.Model):

    def _get_hooks(self):
        return get_hooks()
