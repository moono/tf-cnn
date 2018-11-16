import os
import glob
import shutil
import tensorflow as tf
from tensorflow.python.estimator.canned import metric_keys


def _loss_smaller(best_eval_result, current_eval_result):
    """Compares two evaluation results and returns true if the 2nd one is smaller.

    Both evaluation results should have the values for MetricKeys.LOSS, which are
    used for comparison.

    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.

    Returns:
      True if the loss of current_eval_result is smaller; otherwise, False.

    Raises:
      ValueError: If input eval result is None or no loss is available.
    """
    default_key = metric_keys.MetricKeys.LOSS
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')

    return best_eval_result[default_key] > current_eval_result[default_key]


class BestCheckpointExporter(tf.estimator.Exporter):
    def __init__(self,
                 name='best_checkpoint_exporter',
                 compare_fn=_loss_smaller,
                 num_to_keep=3):
        self._name = name
        self._compare_fn = compare_fn
        self._num_to_keep = max(num_to_keep, 1)
        self._event_file_pattern = 'eval/*.tfevents.*'
        self._model_dir = None
        self._best_eval_result = None
        self._best_checkpoints_path = None
        self._best_checkpoint_steps = list()

    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        export_result = None

        # initial result
        if self._model_dir != estimator.model_dir:
            # Loads best metric from event files.
            tf.logging.info('Loading best metric from event files.')

            # reset
            self._best_checkpoint_steps = list()

            # get model_dir
            self._model_dir = estimator.model_dir

            # form best checkpoint path for the first time
            self._best_checkpoints_path = os.path.join(self._model_dir, 'best_checkpoints')
            if not os.path.exists(self._best_checkpoints_path):
                os.makedirs(self._best_checkpoints_path)

            # find first best result == current result
            full_event_file_pattern = os.path.join(self._model_dir, self._event_file_pattern)
            self._best_eval_result = self._get_best_eval_result(full_event_file_pattern)

            # copy current checkpoint
            self._copy_checkpoint(checkpoint_path)

        # compare every time new eval result comes in and save if neccessary
        elif self._compare_fn(self._best_eval_result, eval_result):
            tf.logging.info('Performing best model copying.')
            self._best_eval_result = eval_result

            # remove old files if needed
            if self._count_number_of_checkpoint_files() >= self._num_to_keep:
                tf.logging.info('Removing oldest best checkpoint file.')
                self._remove_last_best_checkpoint_files()

            # copy current checkpoint
            self._copy_checkpoint(checkpoint_path)

        return export_result

    def _get_best_eval_result(self, event_files):
        if not event_files:
            return None

        best_eval_result = None
        for event_file in tf.gfile.Glob(os.path.join(event_files)):
            for event in tf.train.summary_iterator(event_file):
                if event.HasField('summary'):
                    event_eval_result = {}
                    for value in event.summary.value:
                        if value.HasField('simple_value'):
                            event_eval_result[value.tag] = value.simple_value
                    if event_eval_result:
                        if best_eval_result is None or self._compare_fn(best_eval_result, event_eval_result):
                            best_eval_result = event_eval_result
        return best_eval_result

    def _copy_checkpoint(self, checkpoint_path):
        # add to list
        current_step = self._extract_checkpoint_step(checkpoint_path)
        self._best_checkpoint_steps.append(current_step)

        related_files = glob.glob('{:s}.*'.format(checkpoint_path))
        for file in related_files:
            shutil.copy(file, self._best_checkpoints_path)
        return

    def _count_number_of_checkpoint_files(self):
        related_files = glob.glob(os.path.join(self._best_checkpoints_path, '*.ckpt*'))
        total_cnt = len(related_files) // 3
        return total_cnt

    def _remove_last_best_checkpoint_files(self):
        last_best_checkpoint_step = self._best_checkpoint_steps[0]

        # remove related file
        related_files = glob.glob(os.path.join(self._best_checkpoints_path,
                                               '*.ckpt-{:d}.*'.format(last_best_checkpoint_step)))
        for file in related_files:
            os.remove(file)

        # update list
        self._best_checkpoint_steps.pop(0)
        return

    @staticmethod
    def _extract_checkpoint_step(checkpoint_path):
        fn_only = os.path.basename(checkpoint_path)
        step_str = fn_only.split('-')[-1]
        return int(step_str)
