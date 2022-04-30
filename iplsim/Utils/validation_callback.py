import tensorflow as tf


class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=1):
        """
        :param validation_sets:
        list of (generator, name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 2:
                (validation_generator,
                 validation_set_name) = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(validation_generator,
                                          verbose=self.verbose)

            for metric, result in zip(self.model.metrics_names, results):
                valuename = validation_set_name + '_' + metric
                self.history.setdefault(valuename, []).append(result)
