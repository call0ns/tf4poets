import os, sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
# image_path = sys.argv[1]

def load_graph():
    # Unpersists graph from file
    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def main(_):
    print(FLAGS.buckets)
    print(FLAGS.output_labels)
    print(FLAGS.output_graph)

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(FLAGS.output_labels)]
    
    load_graph()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        image_dir = FLAGS.buckets
        image_files = tf.gfile.ListDirectory(image_dir)
        for image_name in image_files:
            image_path = os.path.join(image_dir, image_name)
            print(image_path)
            # Read in the image_data
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    # tf.app.flags.DEFINE_string("image_dir", "", "input data path")
    tf.app.flags.DEFINE_string("buckets", "", "input data path")
    tf.app.flags.DEFINE_string("output_graph", "", "input data path")
    tf.app.flags.DEFINE_string("output_labels", "", "input data path")

    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.app.run(main=main)
