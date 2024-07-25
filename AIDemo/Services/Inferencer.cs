using AIDemo.Models;
using Models;
using OpenCvSharp;

namespace AIDemo.Services
{
    internal class Inferencer
    {
        public virtual PredictResult Predict(string imgPath)
        {
            throw new NotImplementedException();
        }

        public virtual PredictResult Predict(Mat mat)
        {
            throw new NotImplementedException();
        }

        protected (Mat? anomalyMap, double predScore) Normalize(double predScore, MetaData metadata, Mat anomalyMap)
        {
            // Normalize anomaly maps if provided
            if (anomalyMap != null)
            {
                anomalyMap = NormalizeMinMax(anomalyMap, metadata.pixel_threshold, metadata.min, metadata.max);
            }

            // Normalize predScores
            predScore = NormalizeMinMax(predScore, metadata.image_threshold, metadata.min, metadata.max);

            return (anomalyMap, predScore);
        }

        private double NormalizeMinMax(double value, double threshold, double min, double max)
        {
            if (value < threshold)
                return min;
            else if (value > threshold)
                return max;
            else
                return (value - threshold) / (max - min) + min;
        }

        private Mat NormalizeMinMax(Mat mat, double threshold, double min, double max)
        {
            Mat result = new Mat();
            Cv2.Normalize(mat, result, min, max, NormTypes.MinMax);
            result = result * (max - min) + min; // Rescale to original range
            return result;
        }
    }
}