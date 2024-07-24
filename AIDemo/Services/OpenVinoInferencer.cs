
using OpenVinoSharp;

namespace AIDemo.Services
{
    internal class OpenVinoInferencer : Inferencer
    {
        private const string OPEN_VINO_MODEL_BIN_PATH = "D:/Workspace/Study/anomalib-demo/results/Padim/MVTec/bottle/latest/weights/openvino/model.bin";
        public OpenVinoInferencer(string modelPath)
        {

            using Core core = new Core();
            using var model = core.read_model(modelPath,OPEN_VINO_MODEL_BIN_PATH);
            using CompiledModel compiled_model = core.compile_model(model, "AUTO");
            var inputs = compiled_model.inputs();
            var outputs = compiled_model.outputs();
            var input = compiled_model.get_input(0);
            var output = compiled_model.get_output(0);
            var shape = input.get_partial_shape();
            var isStatic = shape.is_static();
            using InferRequest infer_request = compiled_model.create_infer_request();
            using Tensor input_tensor = infer_request.get_tensor("input");

            infer_request.infer();
            using Tensor output_tensor = infer_request.get_tensor("output0");
        }
    }
}
