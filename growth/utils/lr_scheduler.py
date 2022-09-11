import tensorflow as tf

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    CosineSchedule Class
    """
    def __init__(self, train_steps=16000, warmup_steps=3000, lr=3e-4, decay=0.0001, offset=3e-4):
        """
        """
        super().__init__()
        # self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        self.lr = lr
        self.decay = decay
        self.offset = offset

    def __call__(self, step_num):
        """
        """
        # state = tf.cast(step_num <= self.warmup_steps,tf.float32)
        # self.lr = self.lr * (1 - self.decay * (1-state.eval())) + state.eval() * 1e-7
        # lr =  tf.cast(self.lr,tf.float32) + self.offset
        return self.offset + self.decay * tf.math.cos(3.14*tf.cast(step_num, tf.float32)/self.train_steps)