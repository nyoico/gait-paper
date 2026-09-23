from collections import Counter

import onnx


model = onnx.load(
    "deploy/student_qkd_qat_qdq.onnx"
)

onnx.checker.check_model(model)

op_counts = Counter(
    node.op_type
    for node in model.graph.node
)

print("QuantizeLinear:", op_counts["QuantizeLinear"])
print("DequantizeLinear:", op_counts["DequantizeLinear"])
print("Conv:", op_counts["Conv"])
print("MatMul:", op_counts["MatMul"])
print("Gemm:", op_counts["Gemm"])

if op_counts["QuantizeLinear"] == 0:
    raise RuntimeError(
        "No QuantizeLinear nodes were found. "
        "QAT symbolic export was not applied."
    )

if op_counts["DequantizeLinear"] == 0:
    raise RuntimeError(
        "No DequantizeLinear nodes were found."
    )

print("QAT QDQ ONNX validation: OK")
