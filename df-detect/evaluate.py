import tensorflow as tf
from video_loader import DeepFakeTransformer






if __name__ == "__main__":

    val_root = '../data/source/train_val_sort/val/*/*.mp4'
    resize_shape = (224,224)
    sequence_len = 30
    prefetch_num = 10
    batch_size=3

    val_ds = tf.data.Dataset.list_files(val_root)
    val_transformer = DeepFakeTransformer(resize_shape=resize_shape, seq_length=sequence_len)
    val_ds = val_ds.map(lambda x: val_transformer.transform_map(x)).batch(batch_size).prefetch(prefetch_num)


    metrics = [tf.keras.metrics.AUC(),
               tf.keras.metrics.FalseNegatives(),
               tf.keras.metrics.FalsePositives(),
               tf.keras.metrics.TrueNegatives(),
               tf.keras.metrics.TruePositives()]

    model = tf.keras.models.load_model('models/0307_1000_steps_01_reg.h5')


    model.compile(metrics = metrics)

    model.evaluate(val_ds)