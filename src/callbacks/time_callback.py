import time

from transformers import TrainerCallback


class TimeLoggerCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.last_logged_epoch = -1
        self.epoch_durations = {}  # Optional: store durations per epoch

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None:
            duration = time.time() - self.epoch_start_time
            epoch = int(state.epoch)
            self.epoch_durations[epoch] = round(duration, 2)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.epoch is not None:
            epoch = int(state.epoch)
            if epoch in self.epoch_durations and epoch != self.last_logged_epoch:
                logs["epoch_time_sec"] = self.epoch_durations[epoch]
                self.last_logged_epoch = epoch
