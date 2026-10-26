"""Print input/output tensor shapes for an ONNX model."""
import onnx

m = onnx.load("models/yolo11n.onnx")
for i in m.graph.input:
    print("in :", i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
for o in m.graph.output:
    print("out:", o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
