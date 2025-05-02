from tensorflow.keras.callbacks import TensorBoard
import datetime
import os

def get_tensorboard_callback(log_dir_base):
    # Generate a unique timestamped log directory
    log_dir = os.path.join(log_dir_base, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return TensorBoard(log_dir=log_dir, histogram_freq=1)
