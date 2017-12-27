# -*- coding: utf-8 -*-
import tensorflow as tf


#for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
#    example = tf.train.Example()
#    example.ParseFromString(serialized_example)
#
#    image = example.features.feature['image'].bytes_list.value
#    label = example.features.feature['label'].int64_list.value
#    # 可以做一些预处理之类的
#    print (image, label)
def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label
#以下为测试代码
#img, label = read_and_decode("train.tfrecords")
#
##使用shuffle_batch可以随机打乱输入
#img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                batch_size=30, capacity=2000,
#                                                min_after_dequeue=1000)
#init = tf.initialize_all_variables()
#
#with tf.Session() as sess:
#    sess.run(init)
#    threads = tf.train.start_queue_runners(sess=sess)
#    for i in range(3):
#        val, l= sess.run([img_batch, label_batch])
#        #我们也可以根据需要对val， l进行处理
#        #l = to_categorical(l, 12) 
#        print(val.shape, l)