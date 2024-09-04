using AIDemo.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Models;
using OpenCvSharp;
using Services.Extension;

namespace AIDemo.Services
{
    internal class OnnxInferencer : Inferencer
    {
        private readonly MetaData metaData;
        private readonly InferenceSession session;

        public OnnxInferencer(string modelPath, MetaData metaData)
        {
            var gpuSessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0);
            modelPath += "/model.onnx";
            session = new InferenceSession(modelPath, gpuSessionOptions);
            this.metaData = metaData;
        }

        public override PredictResult Predict(string imgPath)
        {
            Mat imgSrc = Cv2.ImRead(imgPath, ImreadModes.Color);
            int inputWidth = imgSrc.Width;
            int inputHeight = imgSrc.Height;
            var input = GetDenseTensorFromMat(imgSrc, inputWidth, inputHeight);

            var inputMeta = session.InputMetadata;
            var inputName = inputMeta.First().Key;
            var inputDims = inputMeta.First().Value.Dimensions;

            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(inputName, input) };

            Mat colorMap = new();
            Mat colorMixMap = new();

            using var results = session.Run(inputs);

            Tensor<float> segmentationImage = results[0].AsTensor<float>();
            var outputTensor = results[0].AsTensor<float>();
            var outputDims = outputTensor.Dimensions;
            int width = outputDims[2];
            int height = outputDims[3];

            Mat floatMap = new Mat(height, width, MatType.CV_32FC1);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var index = y * width + x;
                    float pixelValue = segmentationImage[0, 0, 0, index];
                    floatMap.Set(y, x, pixelValue);
                }
            }

            var maxFloat = segmentationImage.Max();
            var (anomalyMap, score) = Normalize(maxFloat, metaData, floatMap);

            double minDouble = 0;
            double maxDouble = 0;
            var beta = float.NaN;
            var alpha = 0;
            if (float.IsNaN(beta))
            {
                floatMap.MinMaxIdx(out minDouble, out maxDouble);
                beta = (float)minDouble;
            }

            if (alpha == 0)
            {
                colorMap = floatMap.Normalize(0, 255, NormTypes.MinMax, (int)MatType.CV_8UC1);
            }
            else
            {
                floatMap.ConvertTo(colorMap, MatType.CV_8UC1, alpha, beta);
            }

            Cv2.Resize(floatMap, floatMap, dsize: new Size(inputWidth, inputHeight));

            Mat predMask = new();
            Cv2.Threshold(floatMap, predMask, metaData.pixel_threshold, 255, ThresholdTypes.Binary);
            predMask.ConvertTo(predMask, MatType.CV_8UC1);

            Cv2.Resize(colorMap, colorMap, dsize: new Size(inputWidth, inputHeight));
            Cv2.ApplyColorMap(colorMap, colorMap, ColormapTypes.Jet);
            Cv2.AddWeighted(imgSrc, 0.5, colorMap, 0.5, 0, colorMixMap);

            var res = new PredictResult(imgSrc, null)
            {
                HeatMap = colorMixMap,
                Score = Math.Round(score, 2),
                IsNormal = maxFloat < metaData.image_threshold,
                Mask = predMask
            };
            return res;
        }

        public DenseTensor<float> PreProcessToDenseTensor(Mat src, int tensorWidth, int tensorHeight)
        {
            // Ensure the image is in the correct format (Mat) and expand dimensions if necessary
            Mat processedImage = new Mat();
            Cv2.Resize(src, processedImage, new Size(tensorWidth, tensorHeight));

            // Normalize the image to the range [0, 1]
            //processedImage.ConvertTo(processedImage, MatType.CV_32F, 1.0 / 255);

            if (processedImage.Dims == 3)
            {
                // Expand dimensions
                processedImage = processedImage.Reshape(0, 1);
            }

            if (processedImage.Channels() == 3)
            {
                // Convert HWC to CHW format
                Cv2.CvtColor(processedImage, processedImage, ColorConversionCodes.BGR2RGB);
                processedImage = processedImage.PermuteAxes(0, 3, 1, 2);
            }

            // Convert Mat to float array
            float[] imageData = new float[processedImage.Total()];
            processedImage.GetArray(out imageData);
            // Create a DenseTensor from the float array

            var shape = new int[] { processedImage.Size(0), processedImage.Size(1), processedImage.Size(2), processedImage.Size(3) };
            var dstTensor = new DenseTensor<float>(imageData, shape);
            return dstTensor;
        }

        private static DenseTensor<float> GetDenseTensorFromMat(Mat src, int tensorWidth, int tensorHeight)
        {
            using Mat dst = new();
            var dstTensor = new DenseTensor<float>([1, 3, tensorHeight, tensorWidth]);
            Cv2.Resize(src, dst, new Size(tensorWidth, tensorHeight));

            for (int y = 0; y < tensorHeight; y++)
            {
                for (int x = 0; x < tensorWidth; x++)
                {
                    Vec3b color = dst.At<Vec3b>(y, x);
                    dstTensor[0, 0, y, x] = color.Item2 / 255f;
                    dstTensor[0, 1, y, x] = color.Item1 / 255f;
                    dstTensor[0, 2, y, x] = color.Item0 / 255f;
                }
            }

            dst.Dispose();
            return dstTensor;
        }
    }
}