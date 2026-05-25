import onnx
import sys

def main():
    model_path = sys.argv[1]
    model = onnx.load(model_path)
    op_types = {node.op_type for node in model.graph.node}
    print("Ops in model:", op_types)

if __name__ == "__main__":
    main()
