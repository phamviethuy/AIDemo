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
        public Mat Image { get; }
        public bool IsNormal { get; set; }
        public string Label { get => IsNormal ? "Normal" : "Abnormal"; }
        public Mat Mask { get; set; }
        public double Score { get; set; }
        public Mat Segmentation { get => MarkBoundaries(Image, Mask); }

        private Mat MarkBoundaries(Mat image, Mat predMask)
        {
            // Find contours in the prediction mask
            HierarchyIndex[] hierarchy = [];
            Point[][] contours;
            Cv2.FindContours(predMask, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            // Clone the original image to draw boundaries
            Mat segmentations = image.Clone();

            // Draw each contour on the image with the specified color and thickness
            Scalar color = new Scalar(0, 0, 255); // Red color (BGR format)
            int thickness = 1; // Thickness of the boundary

            foreach (var contour in contours)
            {
                Cv2.DrawContours(segmentations, new[] { contour }, -1, color, thickness);
            }

            return segmentations;
        }
    }
}