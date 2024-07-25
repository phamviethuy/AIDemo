using OpenCvSharp;

namespace Models
{
    internal class PredictResult
    {
        public PredictResult(Mat img, Mat anomaly)
        {
            Image = img;
            Anomaly = anomaly;
        }

        public Mat Anomaly { get; }
        public Mat HeatMap { get; set; }
        public Mat Mask { get; set; }
        public Mat Image { get; }
        public double Score { get; set; }
        public Mat Segmentation { get; set; }
        public string Label { get => IsNormal ? "Normal" : "Abnormal"; }
        public bool IsNormal { get; set; }
    }
}