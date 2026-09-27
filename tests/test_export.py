"""The exported .tflite files have a fixed batch of 1 (the browser runtime cannot resize inputs)."""
import numpy as np
import pytest

from sim2chip.stages.export import fixed_batch

tf = pytest.importorskip("tensorflow")


def test_fixed_batch_keeps_the_model_and_its_outputs():
    inp = tf.keras.Input((8, 8, 3))
    model = tf.keras.Model(inp, tf.keras.layers.Conv2D(2, 3, padding="same", activation="softmax")(inp))
    blob = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    x = np.random.default_rng(0).random((1, 8, 8, 3)).astype(np.float32)
    outs = []
    for b in (blob, fixed_batch(blob)):
        it = tf.lite.Interpreter(model_content=b)
        it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        it.set_tensor(i["index"], x)
        it.invoke()
        outs.append((list(i["shape_signature"]), list(o["shape_signature"]), it.get_tensor(o["index"])))
    assert outs[0][0][0] == -1 and outs[1][0] == [1, 8, 8, 3] and outs[1][1] == [1, 8, 8, 2]
    np.testing.assert_array_equal(outs[0][2], outs[1][2])
